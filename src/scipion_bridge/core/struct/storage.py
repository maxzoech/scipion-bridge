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
    Schema,
    Entry,
    SetEntryBase,
    ArrayEntryBase,
    ArrayEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
    SchemaEntry,
    SchemaSetEntry,
)
from .utils.arrow_utils import (
    RaggedArrayView,
    build_tensor_array,
    build_ragged_array,
    build_multidim_ragged_array,
    _read_ragged,
    _get_nested_arrow_field,
    _slice_arrow_array,
    _arrow_to_numpy,
)
from .key_path import IndexType, KeyPath


class _StorageEngine(abc.ABC):
    """Abstract interface for polymorphic storage engines."""

    @abc.abstractmethod
    def read(
        self, key: KeyPath, entry: Entry,
    ) -> Any: ...

    @abc.abstractmethod
    def write(
        self, key: KeyPath, entry: Entry, data: Any,
    ) -> None: ...

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch: ...

    @abc.abstractmethod
    def __contains__(self, key: KeyPath) -> bool: ...

    @property
    @abc.abstractmethod
    def root_entry(self) -> SchemaEntry: ...

    @property
    @abc.abstractmethod
    def is_frozen(self) -> bool: ...


class _BaseStorage(abc.ABC):
    """
    Abstract base storage class managing offset tracking and reading and writing
    of data.

    Schema-backed objects like `Struct` and `Set` (aka. frontend) use
    implementations of this class to manage their backend storage. The frontend
    uses the schema traced from the Python types to compute a _query_ of a
    schema entry and path.

    Scipion Bridge implements uses three implementation of this class:
    1. StagingStorage: A numpy/akward array implemented used to initialize
    types in local storage
    2. ArrowStorage: An immutable representation based on Apache Arrow which
    can be efficiently transferred over the network. It is used exchange data
    in a distributed setting or when crossing Python interpeter bounds
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

    def narrow_index(self, index: int) -> StorageView:
        """Create a child storage view focused on a single scalar item along the active dimension.

        Args:
            index: Relative integer index along the terminal dimension.

        Returns:
            StorageView: A view focused on the indexed element.
        """
        return self.view(
            self.root.narrow_index(index)
        )

    def narrow_slice(self, index: slice) -> StorageView:
        """Create a child storage view restricted to a sub-slice along the active dimension.

        Args:
            index: Relative slice to compose with the current terminal slice.

        Returns:
            StorageView: A view restricted to the sub-sliced window.
        """

        return self.view(
            self.root.narrow_slice(index)
        )

    def narrow_indices(
        self,
        indices: Union[Sequence[int], NDArray[np.integer]],
        length: Optional[int] = None,
    ) -> StorageView:
        """Create a child storage view restricted to specified integer indices along the active dimension."""
        return self.view(
            self.root.narrow_indices(indices, length=length)
        )

    def narrow_mask(
        self,
        mask: Union[Sequence[bool], NDArray[np.bool_]],
        length: Optional[int] = None,
    ) -> StorageView:
        """Create a child storage view filtered by a boolean mask along the active dimension."""
        return self.view(
            self.root.narrow_mask(mask, length=length)
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

    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        """Return the active sequence length of data stored under key or root."""
        target_key = key if key is not None else self.root
        prefix_fields, index_tuple = self._decompose(target_key)
        effective_idx = self._compute_index(index_tuple)

        for field_key, buffer in self._data.items():
            if field_key[:len(prefix_fields)] == prefix_fields:
                if not effective_idx:
                    return len(buffer)
                return len(buffer[effective_idx])

        return None

    def read(self, key: KeyPath, entry: Entry) -> Any:
        field_key, index_tuple = self._decompose(key)
        if field_key not in self._data:
            raise UninitializedFieldError(f"Field '{key}' has not been initialized.")
            
        buffer = self._data[field_key]
        effective_idx = self._compute_index(index_tuple)
        
        if not effective_idx:
            return buffer
            
        return buffer[effective_idx]

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        assert isinstance(entry, ArrayEntryBase)

        self._validate_dtype(data, entry.dtype, key)

        field_key, index_tuple = self._decompose(key)
        effective_idx = self._compute_index(index_tuple)

        # 1. Full-column write (unbounded)
        if not effective_idx:
            if isinstance(entry, SetEntryBase):
                data_len = len(data) if hasattr(data, "__len__") else 1
                if entry.capacity is not None and data_len > entry.capacity:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': length {data_len} exceeds capacity {entry.capacity}."
                    )

                if entry.capacity is None and __debug__ == True:
                    container_prefix = field_key[:-1]
                    for other_key, other_buffer in self._data.items():
                        
                        if other_key[:-1] == container_prefix and len(other_key) == len(field_key):
                            if data_len != len(other_buffer):
                                raise ValueError(
                                    f"Length mismatch for key '{key}': data length {data_len} does not match existing column '{other_key[-1]}' length {len(other_buffer)}."
                                )
                            break

            self._data[field_key] = self._allocate_buffer(data, entry, key)
            return

        # 2. Indexed write
        if field_key not in self._data:
            raise UninitializedFieldError(
                f"Cannot perform indexed write on uninitialized field '{key}'."
            )

        buffer = self._data[field_key]

        if isinstance(buffer, np.ndarray):
            if self._fits_numpy_buffer(buffer, effective_idx, data):
                buffer[effective_idx] = data
            elif not entry.is_static:
                self._data[field_key] = self._promote_and_update(buffer, effective_idx, data)
            else:
                raise ValueError(f"Shape mismatch writing to static field '{key}'.")
        else:
            if entry.is_static:
                raise ValueError(f"Shape mismatch writing to static field '{key}'.")
            self._data[field_key] = self._update_ak_array(buffer, effective_idx, data)

    def _validate_dtype(self, data: Any, target_dtype: Any, key: KeyPath) -> None:
        if isinstance(data, ak.Array):
            return
            
        data_dtype = getattr(data, "dtype", None)
        if data_dtype is None:
            data_dtype = np.asarray(data).dtype
            
        if data_dtype != object and not np.can_cast(data_dtype, target_dtype, casting="safe"):
            raise TypeError(
                f"Cannot cast data of dtype '{data_dtype}' to field '{key}' dtype '{target_dtype}'."
            )
        
    def _allocate_buffer(self, data: Any, entry: Entry, key: KeyPath) -> Union[np.ndarray, ak.Array]:
        assert isinstance(entry, ArrayEntryBase)
        is_static = entry.is_static
        
        if not isinstance(data, ak.Array):
            try:
                arr = np.asarray(data, dtype=entry.dtype)
                if arr.dtype != object:
                    if is_static:
                        entry_shape = entry.shape
                        if entry_shape == (1,) and len(arr.shape) == 1:
                            arr = arr.reshape(-1, 1)
                        elif len(arr.shape) < len(entry_shape) or arr.shape[len(arr.shape)-len(entry_shape):] != entry_shape:
                            raise ValueError(f"Shape mismatch for static field '{key}': expected {entry_shape} for element.")
                    return arr
            except (ValueError, TypeError):
                pass
                
        if is_static:
            raise ValueError(f"Shape mismatch for static field '{key}'.")
            
        return data if isinstance(data, ak.Array) else ak.Array(data)

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

    def _promote_and_update(self, buffer: np.ndarray, effective_idx: Tuple[IndexType, ...], data: Any) -> ak.Array:
        if len(effective_idx) == 1:
            lst = list(buffer)
        else:
            lst = buffer.tolist()
            
        self._traverse_list_update(lst, effective_idx, data)
        return ak.Array(lst)

    def _update_ak_array(self, buffer: ak.Array, effective_idx: Tuple[IndexType, ...], data: Any) -> ak.Array:
        lst = cast(list, ak.to_list(buffer))
        self._traverse_list_update(lst, effective_idx, data)
        
        return ak.Array(lst)

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