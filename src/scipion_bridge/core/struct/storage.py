"""Unified Arrow-backed storage engine for Struct and Set data structures."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import awkward as ak
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.compute as _pc

pc: Any = _pc

from .exceptions import UninitializedFieldError
from .schema import (
    Entry,
    SetEntryBase,
    ArrayEntryBase,
    SchemaEntry,
    SchemaSetEntry,
)
from .utils.arrow_utils import (
    RaggedArrayView,
)
from .key_path import IndexType, KeyPath


class _BaseStorage(abc.ABC):
    """
    Abstract base storage class managing offset tracking and reading and writing
    of data.

    Schema-backed objects like `Struct` and `Set` (aka. frontend) use
    implementations of this class to manage their backend storage. The frontend
    uses the schema traced from the Python types to compute a _query_ of a
    schema entry and path.

    Scipion Bridge uses three implementations of this class:
    1. StagingEngine: A numpy/Awkward Array backend used to initialize
    types in local storage
    2. ArrowEngine: An immutable representation based on Apache Arrow which
    can be efficiently transferred over the network. It is used to exchange data
    in a distributed setting or when crossing Python interpreter bounds
    3. StorageView: Not another backend, but rather a reference for subslices

    """

    def __init__(
        self,
        root: KeyPath = KeyPath(),
        parent: Optional["_BaseStorage"] = None,
    ) -> None:

        self.parent = parent
        self.root: KeyPath = root

    def view(self, path: KeyPath) -> StorageView:
        assert path.path[:len(self.root)] == self.root.path

        return StorageView(
            path,
            parent=self,
        )

    def append(self, name: str) -> StorageView:
        """Create a child storage view targeting a nested attribute, struct field, or column.

        Args:
            name: The field, attribute, or column name to append to the path.

        Returns:
            StorageView: A view into the parent storage rooted at the appended path.
        """

        return self.view(
            self.root.append(name)
        )

    def narrow_index(self, index: int, length: Optional[int] = None) -> StorageView:
        """Create a child storage view focused on a single scalar item along the active dimension.

        Args:
            index: Relative integer index along the terminal dimension.
            length: Optional sequence length. If omitted, defaults to self.get_length().

        Returns:
            StorageView: A view focused on the indexed element.
        """
        effective_len = length if length is not None else self.get_length()
        return self.view(
            self.root.narrow_index(index, length=effective_len)
        )

    def narrow_slice(self, index: slice, length: Optional[int] = None) -> StorageView:
        """Create a child storage view restricted to a sub-slice along the active dimension.

        Args:
            index: Relative slice to compose with the current terminal slice.
            length: Optional sequence length. If omitted, defaults to self.get_length().

        Returns:
            StorageView: A view restricted to the sub-sliced window.
        """
        effective_len = length if length is not None else self.get_length()
        return self.view(
            self.root.narrow_slice(index, length=effective_len)
        )

    def narrow_indices(
        self,
        indices: Union[Sequence[int], NDArray[np.integer]],
        length: Optional[int] = None,
    ) -> StorageView:
        """Create a child storage view restricted to specified integer indices along the active dimension."""
        effective_len = length if length is not None else self.get_length()
        return self.view(
            self.root.narrow_indices(indices, length=effective_len)
        )

    def narrow_mask(
        self,
        mask: Union[Sequence[bool], NDArray[np.bool_]],
        length: Optional[int] = None,
    ) -> StorageView:
        """Create a child storage view filtered by a boolean mask along the active dimension."""
        effective_len = length if length is not None else self.get_length()
        return self.view(
            self.root.narrow_mask(mask, length=effective_len)
        )

    @property
    def is_view(self) -> bool:
        return self.parent is not None

    @property
    def root_storage(self) -> "_BaseStorage":
        if self.parent is None:
            return self
        
        return self.parent.root_storage


    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        """Return active sequence length under key/root, or None if uninitialized."""
        return None

    @abc.abstractmethod
    def read(
        self,
        key: KeyPath,
        entry: Entry,
    ) -> Any:
        """Read data for the specified schema entry at this storage's index."""

        ...

    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
    ) -> None:
        """Write data for the specified schema entry at this storage's index."""

        ...

    def concat(
        self,
        others: Sequence["_BaseStorage"],
        entry: Entry,
    ) -> "_BaseStorage":
        """Concatenate this storage with other compatible storages along axis 0."""
        target_cls = type(self.root_storage)
        result_storage = target_cls()
        all_storages = [self, *others]

        assert isinstance(entry, SchemaSetEntry) or isinstance(entry, SchemaEntry)

        for path, target_entry in entry.schema.tree_iter():
            if not isinstance(target_entry, ArrayEntryBase):
                continue

            try:
                buffers = [
                    st.read(st.root.extend(path), target_entry)
                    for st in all_storages
                ]
            except UninitializedFieldError:
                init_mask = []
                for st in all_storages:
                    try:
                        st.read(st.root.extend(path), target_entry)
                        init_mask.append(True)
                    except UninitializedFieldError:
                        init_mask.append(False)
                if not any(init_mask):
                    continue
                raise UninitializedFieldError(
                    f"Cannot concatenate sets: field '{path}' is initialized in some sets but not others."
                )

            if target_entry.is_static:
                concatenated = np.concatenate(buffers, axis=0)
            else:
                concatenated = ak.concatenate(buffers, axis=0)

            result_storage.write(
                result_storage.root.extend(path),
                target_entry,
                concatenated,
            )

        return result_storage


class StorageView(_BaseStorage):
    """
    A reference to some other storage.

    Schema-based types like `Struct` or `Set` are internally backed by an array
    storage (e.g. numpy or Arrow).
    
    """

    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        target_key = key if key is not None else self.root
        return self.root_storage.get_length(target_key)

    def read(self, key: KeyPath, entry: Entry) -> Any:
        return self.root_storage.read(key, entry)

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        return self.root_storage.write(key, entry, data)



class StagingEngine(_BaseStorage):
    """Mutable in-memory storage engine backed by NumPy and Awkward Arrays."""

    def __init__(
        self,
        root: KeyPath = KeyPath(),
        parent: Optional[_BaseStorage] = None,
    ) -> None:
        super().__init__(root=root, parent=parent)
        self._data: Dict[Tuple[str, ...], Union[np.ndarray, ak.Array]] = {}
        self._chunks: Dict[Tuple[str, ...], List[Any]] = {}

    @staticmethod
    def _decompose(key: KeyPath) -> Tuple[Tuple[str, ...], Tuple[IndexType, ...]]:
        """Extracts field tuple (excluding 'root') and corresponding index tuple."""
        path = key.path
        if not path or path[0] != "root":
            raise ValueError("KeyPath must start with a 'root' component")

        return path[1:], key.indices

    @staticmethod
    def _compute_index(index_tuple: Tuple[IndexType, ...]) -> Tuple[IndexType, ...]:
        """Strip trailing slice(None) no-ops to form a clean index tuple."""
        match index_tuple:
            case (*rest, slice() as trailing_slice) if trailing_slice == slice(None):
                return StagingEngine._compute_index(tuple(rest))

            case _:
                return index_tuple

    @staticmethod
    def _get_item_from_index(
        container: Any, effective_idx: Tuple[IndexType, ...],
    ) -> Any:
        curr = container
        for idx in effective_idx:
            if curr is None:
                return None
            match idx:
                case int(i):
                    curr = curr[i]
                case slice() as s:
                    curr = curr[s]
                case _:
                    curr = curr[idx]
        return curr

    def _materialize_chunks(
        self,
        field_key: Tuple[str, ...],
        entry: Entry,
    ) -> Union[np.ndarray, ak.Array]:
        chunks = self._chunks[field_key]
        valid_chunks = [c for c in chunks if c is not None]

        match valid_chunks:
            case []:
                arr = ak.Array([None] * len(chunks))

            case _ if not isinstance(valid_chunks[0], (np.ndarray, ak.Array, Sequence)):
                arr = ak.Array(chunks)

            case _ if self._is_uniform_numpy(valid_chunks, chunks):
                arr = self._stack_uniform_numpy(valid_chunks, chunks, entry)

            case _:
                arr = self._stack_ragged(valid_chunks, chunks)

        self._data[field_key] = arr
        del self._chunks[field_key]
        return arr

    @staticmethod
    def _is_uniform_numpy(
        valid_chunks: List[Any],
        chunks: List[Any],
    ) -> bool:
        """Check if all valid chunks are numpy arrays with identical shapes and no None gaps."""
        first = valid_chunks[0]
        return (
            all(isinstance(c, np.ndarray) for c in valid_chunks)
            and isinstance(first, np.ndarray)
            and all(c.shape == first.shape for c in valid_chunks)
            and all(c is not None for c in chunks)
        )

    @staticmethod
    def _stack_uniform_numpy(
        valid_chunks: List[Any],
        chunks: List[Any],
        entry: Entry,
    ) -> Union[np.ndarray, ak.Array]:
        """Stack uniform-shape numpy chunks into a single buffer."""
        stacked = np.stack(chunks, axis=0)
        return stacked if entry.is_static else ak.Array(stacked)

    @staticmethod
    def _stack_ragged(
        valid_chunks: List[Any],
        chunks: List[Any],
    ) -> ak.Array:
        """Concatenate ragged chunks into a single Awkward Array, masking None slots."""
        flat = ak.concatenate(valid_chunks, axis=0)
        lengths = [len(c) if c is not None else 0 for c in chunks]
        unflat = ak.unflatten(flat, lengths, axis=0)

        if any(c is None for c in chunks):
            mask = [c is not None for c in chunks]
            return ak.mask(unflat, mask)

        return unflat

    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        """Return the active sequence length of data stored under key or root."""
        target_key = key if key is not None else self.root
        prefix_fields, index_tuple = self._decompose(target_key)
        effective_idx = self._compute_index(index_tuple)

        for field_key, buffer in self._data.items():
            if field_key[: len(prefix_fields)] == prefix_fields:
                if not effective_idx:
                    return len(buffer)
                return len(buffer[effective_idx])

        for field_key, chunks in self._chunks.items():
            if field_key[: len(prefix_fields)] == prefix_fields:
                if not effective_idx:
                    return len(chunks)
                val = self._get_item_from_index(chunks, effective_idx)
                return len(val) if val is not None else 0

        return None

    def read(self, key: KeyPath, entry: Entry) -> Any:
        field_key, index_tuple = self._decompose(key)
        effective_idx = self._compute_index(index_tuple)

        # Guard: pending chunks with scalar index → return directly
        if field_key in self._chunks:
            if effective_idx and all(
                isinstance(idx, int) for idx in effective_idx
            ):
                val = self._get_item_from_index(
                    self._chunks[field_key], effective_idx,
                )
                if val is None:
                    raise UninitializedFieldError(
                        f"Element at '{key}' has not been initialized.",
                    )
                return val
            self._materialize_chunks(field_key, entry)

        if field_key not in self._data:
            raise UninitializedFieldError(
                f"Field '{key}' has not been initialized.",
            )

        buffer = self._data[field_key]

        if not effective_idx:
            return self._read_full_column(buffer, entry)

        return self._read_indexed(buffer, effective_idx, key)

    def _read_full_column(
        self,
        buffer: Union[np.ndarray, ak.Array],
        entry: Entry,
    ) -> Any:
        """Read an entire column, wrapping dynamic entries as RaggedArrayView when possible."""
        if not (
            isinstance(entry, ArrayEntryBase)
            and not entry.is_static
            and isinstance(buffer, (ak.Array, np.ndarray))
        ):
            return buffer

        try:
            ak_arr = (
                buffer
                if isinstance(buffer, ak.Array)
                else ak.Array(buffer)
            )
            pa_arr = ak.to_arrow(ak_arr, extensionarray=False)
            if isinstance(pa_arr, pa.ChunkedArray):
                pa_arr = pa_arr.combine_chunks()
            if isinstance(
                pa_arr,
                (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray),
            ):
                return RaggedArrayView(pa_arr, entry.dtype)
        except (pa.ArrowInvalid, ValueError, TypeError):
            pass

        return buffer

    @staticmethod
    def _read_indexed(
        buffer: Union[np.ndarray, ak.Array],
        effective_idx: Tuple[IndexType, ...],
        key: KeyPath,
    ) -> Any:
        """Read a single indexed element, raising on uninitialized slots."""
        val = buffer[effective_idx]
        match val:
            case None:
                raise UninitializedFieldError(
                    f"Element at '{key}' has not been initialized.",
                )
            case _ if isinstance(val, ak.Array) and len(val) == 0 and ak.is_none(val):
                raise UninitializedFieldError(
                    f"Element at '{key}' has not been initialized.",
                )
            case _:
                return val

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        assert isinstance(entry, ArrayEntryBase)
        self._validate_dtype(data, entry.dtype, key)

        field_key, index_tuple = self._decompose(key)
        effective_idx = self._compute_index(index_tuple)

        if not effective_idx:
            return self._write_full_column(field_key, entry, data, key)

        if entry.is_static:
            return self._write_indexed_static(
                field_key, entry, effective_idx, data, key,
            )

        return self._write_indexed_dynamic(field_key, entry, effective_idx, data)

    def _write_full_column(
        self,
        field_key: Tuple[str, ...],
        entry: ArrayEntryBase,
        data: Any,
        key: KeyPath,
    ) -> None:
        """Unbounded full-column write with capacity and cross-field length validation."""
        if isinstance(entry, SetEntryBase):
            data_len = len(data) if hasattr(data, "__len__") else 1
            if entry.capacity is not None and data_len > entry.capacity:
                raise ValueError(
                    f"Shape mismatch for key '{key}': length {data_len} exceeds capacity {entry.capacity}.",
                )

            if entry.capacity is None and __debug__ == True:
                container_prefix = field_key[:-1]
                for other_key, other_buffer in self._data.items():
                    if other_key[:-1] == container_prefix and len(other_key) == len(field_key):
                        if data_len != len(other_buffer):
                            raise ValueError(
                                f"Length mismatch for key '{key}': data length {data_len} does not match existing column '{other_key[-1]}' length {len(other_buffer)}.",
                            )
                        break

        self._chunks.pop(field_key, None)
        self._data[field_key] = self._allocate_buffer(data, entry, key)

    def _write_indexed_static(
        self,
        field_key: Tuple[str, ...],
        entry: ArrayEntryBase,
        effective_idx: Tuple[IndexType, ...],
        data: Any,
        key: KeyPath,
    ) -> None:
        """Indexed write on a static (fixed-shape) field with auto-allocation and expansion."""
        if field_key not in self._data:
            outer_dims = self._infer_outer_dims(entry, effective_idx)
            int_shape = cast(Tuple[int, ...], entry.shape)
            total_shape = (*outer_dims, *int_shape)
            buffer = np.zeros(
                total_shape,
                dtype=entry.dtype,
            )
            buffer[effective_idx] = data
            self._data[field_key] = buffer
            return

        buffer = self._data[field_key]
        assert isinstance(buffer, np.ndarray)
        buffer = self._expand_numpy_if_needed(
            buffer,
            effective_idx,
            field_key,
        )
        if self._fits_numpy_buffer(buffer, effective_idx, data):
            buffer[effective_idx] = data
            return

        raise ValueError(
            f"Shape mismatch writing to static field '{key}'.",
        )

    def _write_indexed_dynamic(
        self,
        field_key: Tuple[str, ...],
        entry: ArrayEntryBase,
        effective_idx: Tuple[IndexType, ...],
        data: Any,
    ) -> None:
        """Indexed write on a dynamic field, staged as chunks for deferred materialization."""
        if field_key in self._data:
            existing = self._data.pop(field_key)
            if isinstance(existing, ak.Array):
                self._chunks[field_key] = cast(list, ak.to_list(existing))
            else:
                self._chunks[field_key] = list(existing)

        if field_key not in self._chunks:
            outer_dims = self._infer_outer_dims(entry, effective_idx)
            self._chunks[field_key] = self._build_nested_list(outer_dims)

        chunks = self._chunks[field_key]
        match effective_idx:
            case (int(idx), *_) if idx >= len(chunks):
                chunks.extend([None] * (idx + 1 - len(chunks)))
            case _:
                pass

        self._traverse_list_update(chunks, effective_idx, data)

    def _validate_dtype(self, data: Any, target_dtype: Any, key: KeyPath) -> None:
        match data:
            case ak.Array():
                return

            case list() | tuple() if len(data) > 0:
                first = data[0]
                if isinstance(first, ak.Array):
                    return
                first_dtype = getattr(first, "dtype", None)
                if first_dtype is not None:
                    if first_dtype != object and not np.can_cast(
                        first_dtype, target_dtype, casting="same_kind",
                    ):
                        raise TypeError(
                            f"Cannot cast data of dtype '{first_dtype}' to field '{key}' dtype '{target_dtype}'.",
                        )
                    return

            case _:
                pass

        data_dtype = getattr(data, "dtype", None)
        if data_dtype is None:
            try:
                data_dtype = np.asarray(data).dtype
            except (ValueError, TypeError):
                return

        if data_dtype != object and not np.can_cast(
            data_dtype, target_dtype, casting="same_kind",
        ):
            raise TypeError(
                f"Cannot cast data of dtype '{data_dtype}' to field '{key}' dtype '{target_dtype}'.",
            )

    def _allocate_buffer(
        self, data: Any, entry: Entry, key: KeyPath,
    ) -> Union[np.ndarray, ak.Array]:
        assert isinstance(entry, ArrayEntryBase)
        cap = entry.capacity if isinstance(entry, SetEntryBase) else None

        match data:
            case list() | tuple() if len(data) > 0:
                return self._allocate_from_sequence(data, entry, cap, key)

            case ak.Array():
                if entry.is_static:
                    raise ValueError(f"Shape mismatch for static field '{key}'.")
                return data

            case _:
                return self._allocate_from_scalar_or_array(data, entry, cap, key)

    def _allocate_from_sequence(
        self,
        data: Any,
        entry: ArrayEntryBase,
        cap: Optional[int],
        key: KeyPath,
    ) -> Union[np.ndarray, ak.Array]:
        """Allocate a buffer from a list or tuple of elements."""
        is_static = entry.is_static

        all_np = all(isinstance(c, np.ndarray) for c in data)
        if all_np and all(c.shape == data[0].shape for c in data):
            stacked = np.stack(data, axis=0)
            if cap is not None and cap > len(data):
                buffer = np.zeros(
                    (cap, *stacked.shape[1:]),
                    dtype=entry.dtype,
                )
                buffer[: len(data)] = stacked
                return buffer if is_static else ak.Array(buffer)
            return stacked if is_static else ak.Array(stacked)

        if all(isinstance(c, (np.ndarray, ak.Array)) for c in data):
            flat = ak.concatenate(data, axis=0)
            lengths = [len(c) for c in data]
            unflat = ak.unflatten(flat, lengths, axis=0)
            if cap is not None and cap > len(data):
                return ak.pad_none(unflat, cap, axis=0)
            return unflat

        try:
            arr = np.asarray(data, dtype=entry.dtype)
            if arr.dtype != object:
                if cap is not None and cap > len(data):
                    buf = np.zeros(
                        (cap, *arr.shape[1:]),
                        dtype=entry.dtype,
                    )
                    buf[: len(data)] = arr
                    return buf if is_static else ak.Array(buf)
                return arr if is_static else ak.Array(arr)
        except (ValueError, TypeError):
            pass

        return self._allocate_from_scalar_or_array(data, entry, cap, key)

    def _allocate_from_scalar_or_array(
        self,
        data: Any,
        entry: ArrayEntryBase,
        cap: Optional[int],
        key: KeyPath,
    ) -> Union[np.ndarray, ak.Array]:
        """Allocate a buffer from a scalar, ndarray, or Awkward Array."""
        if isinstance(data, ak.Array):
            if entry.is_static:
                raise ValueError(f"Shape mismatch for static field '{key}'.")
            return data

        try:
            arr = np.asarray(data, dtype=entry.dtype)
        except (ValueError, TypeError):
            arr = None

        if arr is not None and arr.dtype != object:
            if entry.is_static:
                return self._coerce_static_shape(arr, entry.shape, key)
            return arr

        if entry.is_static:
            raise ValueError(f"Shape mismatch for static field '{key}'.")

        return ak.Array(data)

    @staticmethod
    def _coerce_static_shape(
        arr: np.ndarray,
        entry_shape: Tuple[Optional[int], ...],
        key: KeyPath,
    ) -> np.ndarray:
        """Validate and adjust array shape to match static entry dimensions."""
        match (arr.shape, entry_shape):
            case ((), (1,)):
                return arr.reshape(1)

            case ((), ()):
                return arr.reshape(())

            case ((n,), (1,)) if n != 1:
                return arr.reshape(-1, 1)

            case (shape, target) if (
                len(shape) >= len(target)
                and shape[len(shape) - len(target):] == target
            ):
                return arr

            case _:
                raise ValueError(
                    f"Shape mismatch for static field '{key}': expected {entry_shape} for element.",
                )

    def _fits_numpy_buffer(self, buffer: np.ndarray, effective_idx: Tuple[IndexType, ...], data: Any) -> bool:
        try:
            arr_data = np.asarray(data, dtype=buffer.dtype)
            if arr_data.dtype == object:
                return False
            
            target_shape = buffer[effective_idx].shape
            np.broadcast_shapes(target_shape, arr_data.shape)
            return True
        except (ValueError, TypeError, IndexError):
            return False


    def _infer_outer_dims(
        self, entry: Entry, effective_idx: Tuple[IndexType, ...]
    ) -> Tuple[int, ...]:
        outer_dims: List[int] = []
        for i, idx in enumerate(effective_idx):
            dim_cap = (
                entry.capacity
                if i == 0 and isinstance(entry, SetEntryBase) and entry.capacity is not None
                else 0
            )
            match idx:
                case int(n):
                    dim_cap = max(dim_cap, n + 1)
                case slice() as s if s.stop is not None:
                    dim_cap = max(dim_cap, s.stop)
                case np.ndarray() as arr:
                    dim_cap = max(dim_cap, len(arr))
                case Sequence() as seq:
                    dim_cap = max(dim_cap, len(seq))

            outer_dims.append(dim_cap)
        return tuple(outer_dims)

    @staticmethod
    def _build_nested_list(dims: Sequence[int]) -> list:
        if len(dims) <= 1:
            return [None] * (dims[0] if dims else 0)
        return [StagingEngine._build_nested_list(dims[1:]) for _ in range(dims[0])]

    def _expand_numpy_if_needed(
        self,
        buffer: np.ndarray,
        effective_idx: Tuple[IndexType, ...],
        field_key: Tuple[str, ...],
    ) -> np.ndarray:
        new_shape = list(buffer.shape)
        expanded = False
        for i, idx in enumerate(effective_idx):
            match idx:
                case int(n) if n >= new_shape[i]:
                    new_shape[i] = n + 1
                    expanded = True
                case slice() as s if s.stop is not None and s.stop > new_shape[i]:
                    new_shape[i] = s.stop
                    expanded = True
                case _:
                    pass

        if expanded:
            new_buffer = np.zeros(tuple(new_shape), dtype=buffer.dtype,)
            slices = tuple(slice(0, s) for s in buffer.shape)
            new_buffer[slices] = buffer
            self._data[field_key] = new_buffer
            return new_buffer

        return buffer


    def _traverse_list_update(self, lst: list, effective_idx: Tuple[IndexType, ...], data: Any) -> None:
        match effective_idx:
            case (np.ndarray() as arr,):
                for i, d in zip(arr, data):
                    lst[int(i)] = d

            case (single_idx,):
                lst[single_idx] = data

            case (slice() as s, *rest):
                for i, d in zip(range(*s.indices(len(lst))), data):
                    self._traverse_list_update(lst[i], tuple(rest), d)

            case (np.ndarray() as arr, *rest):
                for i, d in zip(arr, data):
                    self._traverse_list_update(lst[int(i)], tuple(rest), d)

            case (int() as idx, *rest):
                self._traverse_list_update(lst[idx], tuple(rest), data)



# class ArrayStorage(_BaseStorage):
#     """Unified Arrow-backed storage engine delegating to polymorphic _StorageEngine implementations."""

#     def __init__(
#         self,
#         schema: Schema,
#         capacity: Optional[int] = None,
#         record_batch: Optional[pa.RecordBatch] = None,
#         path: KeyPath = ("root",),
#         offset: Union[Offset, Sequence[IndexType]] = (),
#         root_entry: Optional[SchemaEntry] = None,
#     ) -> None:
#         super().__init__(schema=schema, parent=None, path=path, offset=Offset(offset))
#         self._schema = schema

#         if root_entry is not None:
#             resolved_root_entry = root_entry
#         elif capacity is not None:
#             resolved_root_entry = SchemaSetEntry(schema=schema, capacity=capacity)
#         else:
#             resolved_root_entry = SchemaEntry(schema=schema)

#         if record_batch is not None:
#             self._engine: _StorageEngine = _ArrowEngine(
#                 schema, record_batch, root_entry=resolved_root_entry
#             )
#         else:
#             self._engine = _StagingEngine(
#                 schema, capacity=capacity, root_entry=resolved_root_entry
#             )

#     @property
#     def root_entry(self) -> SchemaEntry:
#         return self._engine.root_entry

#     @property
#     def capacity(self) -> Optional[int]:
#         return self._engine.capacity

#     @property
#     def is_frozen(self) -> bool:
#         return self._engine.is_frozen

#     def to_record_batch(self) -> pa.RecordBatch:
#         """Freeze and compile the storage into an immutable Arrow RecordBatch."""
#         batch = self._engine.to_record_batch()
#         if not self._engine.is_frozen:
#             # Transition to immutable Arrow engine and drop staging buffers
#             self._engine = _ArrowEngine(
#                 self._schema, batch, root_entry=self._engine.root_entry
#             )
#         return batch

#     @classmethod
#     def from_record_batch(
#         cls,
#         batch: pa.RecordBatch,
#         schema: Schema,
#         root_entry: Optional[SchemaEntry] = None,
#     ) -> "ArrayStorage":
#         """Construct an ArrayStorage directly wrapping a frozen Arrow RecordBatch."""
#         return cls(
#             schema=schema,
#             capacity=len(batch),
#             record_batch=batch,
#             root_entry=root_entry,
#         )