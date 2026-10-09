"""Storage engines for Struct and Set data structures."""

from __future__ import annotations

import abc
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import awkward as ak
import numpy as np
from numpy.typing import NDArray

from .buffers import (
    Column,
    as_regular_numpy,
    assign_rows,
    broadcastable,
    check_castable,
    coerce_static_shape,
    concat_columns,
    expand_to_fit,
    expanded_shape,
    has_numpy_layout,
    infer_outer_dims,
    nested_rows,
    pad_to_capacity,
    split_rows,
    stack_rows,
    stack_sequence,
)
from .exceptions import UninitializedFieldError
from .schema import (
    Entry,
    SetEntryBase,
    ArrayEntryBase,
    SchemaEntry,
    SchemaSetEntry,
)
from .key_path import IndexType, KeyPath

FieldKey = Tuple[str, ...]


class _BaseStorage(abc.ABC):
    """
    Abstract base storage class managing offset tracking and reading and writing
    of data.

    Schema-backed objects like `Struct` and `Set` (aka. frontend) use
    implementations of this class to manage their backend storage. The frontend
    uses the schema traced from the Python types to compute a _query_ of a
    schema entry and path.

    Scipion Bridge uses two implementations of this class:
    1. RootEngine: The NumPy/Awkward Array backend that owns the data
    2. StorageView: Not another backend, but a reference into the RootEngine
    for nested fields and subslices
    """

    def __init__(
        self,
        root: KeyPath = KeyPath(),
        parent: Optional["_BaseStorage"] = None,
    ) -> None:

        self.parent = parent
        self.root: KeyPath = root

    def view(self, path: KeyPath) -> StorageView:
        assert path.path[: len(self.root)] == self.root.path

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

        return self.view(self.root.append(name))

    def narrow_index(self, index: int, length: Optional[int] = None) -> StorageView:
        """Create a child storage view focused on a single scalar item along the active dimension.

        Args:
            index: Relative integer index along the terminal dimension.
            length: Optional sequence length. If omitted, defaults to self.get_length().

        Returns:
            StorageView: A view focused on the indexed element.
        """
        effective_len = length if length is not None else self.get_length()
        return self.view(self.root.narrow_index(index, length=effective_len))

    def narrow_slice(self, index: slice, length: Optional[int] = None) -> StorageView:
        """Create a child storage view restricted to a sub-slice along the active dimension.

        Args:
            index: Relative slice to compose with the current terminal slice.
            length: Optional sequence length. If omitted, defaults to self.get_length().

        Returns:
            StorageView: A view restricted to the sub-sliced window.
        """
        effective_len = length if length is not None else self.get_length()
        return self.view(self.root.narrow_slice(index, length=effective_len))

    def narrow_indices(
        self,
        indices: Union[Sequence[int], NDArray[np.integer]],
        length: Optional[int] = None,
    ) -> StorageView:
        """Create a child storage view restricted to specified integer indices along the active dimension."""
        effective_len = length if length is not None else self.get_length()
        return self.view(self.root.narrow_indices(indices, length=effective_len))

    def narrow_mask(
        self,
        mask: Union[Sequence[bool], NDArray[np.bool_]],
        length: Optional[int] = None,
    ) -> StorageView:
        """Create a child storage view filtered by a boolean mask along the active dimension."""
        effective_len = length if length is not None else self.get_length()
        return self.view(self.root.narrow_mask(mask, length=effective_len))

    @property
    def is_view(self) -> bool:
        return self.parent is not None

    @property
    def root_storage(self) -> "_BaseStorage":
        if self.parent is None:
            return self

        return self.parent.root_storage

    @abc.abstractmethod
    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        """Return active sequence length under key/root, or None if uninitialized."""

    @abc.abstractmethod
    def read(
        self,
        key: KeyPath,
        entry: Entry,
    ) -> Any:
        """Read data for the specified schema entry at this storage's index."""

    @abc.abstractmethod
    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
    ) -> None:
        """Write data for the specified schema entry at this storage's index."""

    @abc.abstractmethod
    def clear(self, key: Optional[KeyPath] = None) -> None:
        """Clear all stored data under key or root."""

    @abc.abstractmethod
    def is_initialized(self, key: KeyPath) -> bool:
        """Return True if the field identified by key has been initialized in storage."""

    def __contains__(self, key: KeyPath) -> bool:
        """Enable 'key in storage' membership testing."""
        return self.is_initialized(key)

    def concat(
        self,
        others: Sequence["_BaseStorage"],
        entry: Entry,
    ) -> "_BaseStorage":
        """Concatenate this storage with other compatible storages along axis 0."""
        assert isinstance(entry, (SchemaEntry, SchemaSetEntry))

        target_cls = type(self.root_storage)
        result_storage = target_cls()
        all_storages = [self, *others]

        for path, target_entry in entry.schema.tree_iter():
            if not isinstance(target_entry, ArrayEntryBase):
                continue

            st_paths = [st.root.extend(path) for st in all_storages]
            init_mask = [p in st for p, st in zip(st_paths, all_storages)]

            if not any(init_mask):
                continue

            if not all(init_mask):
                raise UninitializedFieldError(
                    f"Cannot concatenate sets: field '{path}' is initialized in some sets but not others.",
                )

            buffers = [
                st.read(p, target_entry) for p, st in zip(st_paths, all_storages)
            ]

            match target_entry.is_static:
                case True:
                    concatenated = np.concatenate(buffers, axis=0)
                case False:
                    concatenated = concat_columns(buffers)

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

    def is_initialized(self, key: KeyPath) -> bool:
        return self.root_storage.is_initialized(key)

    def clear(self, key: Optional[KeyPath] = None) -> None:
        target_key = key if key is not None else self.root
        self.root_storage.clear(target_key)

    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        target_key = key if key is not None else self.root
        return self.root_storage.get_length(target_key)

    def read(self, key: KeyPath, entry: Entry) -> Any:
        return self.root_storage.read(key, entry)

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        return self.root_storage.write(key, entry, data)


class RootEngine(_BaseStorage):
    """Mutable in-memory storage engine backed by NumPy and Awkward Arrays.

    The root of a storage tree: it owns the data of a Struct or Set, and the
    StorageViews of nested fields and selections resolve to it.

    Every leaf field of the schema is stored as one column, keyed by its field
    path (without indices). A column lives either in ``_data`` as a single
    buffer, or, while a dynamic field is written element by element, in
    ``_chunks`` as a list of per-row buffers that is stacked into a buffer on
    the first read of the whole column.

    The type of a buffer follows from the data, not from the schema: it is a
    NumPy array whenever all rows are present and have the same shape, and an
    Awkward Array only when rows differ in shape (possible for dynamic fields,
    which have ``None`` dimensions) or some rows are missing. A buffer becomes
    Awkward only when the data requires it, e.g. a row of a different shape is
    written or a ragged column is concatenated; reads never convert it back.
    Where a column is built from parts (stacking rows, ``concat``, restoring
    from Arrow), regular data is built as NumPy directly.

    Concurrent element-wise access is supported as long as every thread reads
    and writes its own elements (e.g. ``Op.map_element``): writes into
    disjoint rows of an existing column touch disjoint memory and are
    lock-free. Transitions that replace a whole column object (allocation,
    copy-on-write of a read-only buffer, splitting a ragged column into rows,
    materializing pending rows) run once under a lock with double-checked
    locking. Whole-column writes and buffer expansion are not thread-safe.
    """

    def __init__(
        self,
        root: KeyPath = KeyPath(),
        parent: Optional[_BaseStorage] = None,
    ) -> None:
        super().__init__(root=root, parent=parent)
        self._data: Dict[FieldKey, Column] = {}
        self._chunks: Dict[FieldKey, List[Any]] = {}
        self._lock = threading.RLock()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self._lock = threading.RLock()

    # --------------------------------------------------------------------------
    # Keys and columns
    # --------------------------------------------------------------------------

    @staticmethod
    def _decompose(key: KeyPath) -> Tuple[FieldKey, Tuple[IndexType, ...]]:
        """Extracts field tuple (excluding 'root') and corresponding index tuple."""
        path = key.path
        if not path or path[0] != "root":
            raise ValueError("KeyPath must start with a 'root' component")

        return path[1:], key.indices

    @staticmethod
    def _compute_index(index_tuple: Tuple[IndexType, ...]) -> Tuple[IndexType, ...]:
        """Strip slice(None) no-ops to form a clean index tuple for array storage."""
        return tuple(
            idx
            for idx in index_tuple
            if not (isinstance(idx, slice) and idx == slice(None))
        )

    @staticmethod
    def _is_under(field_key: FieldKey, prefix: FieldKey) -> bool:
        """True if ``field_key`` is ``prefix`` or a field nested below it."""
        return field_key[: len(prefix)] == prefix

    def _columns(self) -> Tuple[Tuple[FieldKey, Union[Column, List[Any]]], ...]:
        """Snapshot of all columns: buffers first, then pending row lists."""
        return (*tuple(self._data.items()), *tuple(self._chunks.items()))

    def _siblings(self, field_key: FieldKey) -> List[Union[Column, List[Any]]]:
        """Other columns of the container that holds ``field_key``.

        Columns of one container share their outer length.
        """
        return [
            column
            for key, column in self._columns()
            if key != field_key
            and len(key) == len(field_key)
            and key[:-1] == field_key[:-1]
        ]

    def _sibling_length(self, field_key: FieldKey) -> int:
        """Length of an existing column in the same container, or 0.

        A new column is allocated at full length at once instead of being
        expanded by later (possibly concurrent) writes.
        """
        return max((len(column) for column in self._siblings(field_key)), default=0)

    @staticmethod
    def _get_item_from_index(
        container: Any,
        effective_idx: Tuple[IndexType, ...],
    ) -> Any:
        """Index nested row lists level by level; None if a level is missing."""
        curr = container
        for idx in effective_idx:
            if curr is None:
                return None
            curr = curr[idx]
        return curr

    @classmethod
    def _element_at(
        cls,
        column: Union[Column, List[Any]],
        effective_idx: Tuple[IndexType, ...],
    ) -> Any:
        match column:
            case list():
                return cls._get_item_from_index(column, effective_idx)
            case _:
                return column[effective_idx]

    # --------------------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------------------

    def is_initialized(self, key: KeyPath) -> bool:
        """Return True if the field identified by key has been initialized in staging."""
        field_key, _ = self._decompose(key)
        return any(self._is_under(k, field_key) for k, _ in self._columns())

    def clear(self, key: Optional[KeyPath] = None) -> None:
        """Clear all stored data under key or root."""
        target_key = key if key is not None else self.root
        prefix, _ = self._decompose(target_key)

        self._data = {
            k: v for k, v in tuple(self._data.items()) if not self._is_under(k, prefix)
        }
        self._chunks = {
            k: v
            for k, v in tuple(self._chunks.items())
            if not self._is_under(k, prefix)
        }

    def get_length(self, key: Optional[KeyPath] = None) -> Optional[int]:
        """Return the active sequence length of data stored under key or root.

        All columns below a container share their outer length, so the length
        is taken from the first column found. A missing element has length 0.
        """
        target_key = key if key is not None else self.root
        prefix, index_tuple = self._decompose(target_key)
        effective_idx = self._compute_index(index_tuple)

        column = next(
            (c for k, c in self._columns() if self._is_under(k, prefix)),
            None,
        )
        match (column, effective_idx):
            case (None, _):
                return None
            case (_, ()):
                return len(column)
            case _:
                element = self._element_at(column, effective_idx)
                return 0 if element is None else len(element)

    # --------------------------------------------------------------------------
    # Reading
    # --------------------------------------------------------------------------

    def read(self, key: KeyPath, entry: Entry) -> Any:
        field_key, index_tuple = self._decompose(key)
        effective_idx = self._compute_index(index_tuple)

        chunks = self._chunks.get(field_key)
        match chunks:
            case list() if effective_idx and all(
                isinstance(idx, int) for idx in effective_idx
            ):
                # A single element of a pending column is read without
                # stacking the rows into a buffer.
                return self._initialized(
                    self._get_item_from_index(chunks, effective_idx),
                    key,
                )
            case list():
                self._materialize_pending(field_key, entry)
            case None:
                pass

        if field_key not in self._data:
            raise UninitializedFieldError(
                f"Field '{key}' has not been initialized.",
            )

        buffer = self._data[field_key]
        if not effective_idx:
            return buffer

        return self._initialized(buffer[effective_idx], key)

    @staticmethod
    def _initialized(value: Any, key: KeyPath) -> Any:
        """Return ``value``, raising if it is a missing (never written) element.

        Missing rows read back as None from both pending row lists and masked
        Awkward columns. An empty array, e.g. from an all-False mask, is data.
        """
        match value:
            case None:
                raise UninitializedFieldError(
                    f"Element at '{key}' has not been initialized.",
                )
            case _:
                return value

    def _materialize_pending(self, field_key: FieldKey, entry: Entry) -> None:
        """Merge pending rows of a column into one buffer, once, under the lock."""
        with self._lock:
            if field_key not in self._chunks:
                return

            self._data[field_key] = stack_rows(self._chunks[field_key])
            # Publish the buffer before removing the rows, so that
            # concurrent readers always find the field.
            del self._chunks[field_key]

    # --------------------------------------------------------------------------
    # Writing
    # --------------------------------------------------------------------------

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        assert isinstance(entry, ArrayEntryBase)
        self._validate_dtype(data, entry.dtype, key)

        field_key, index_tuple = self._decompose(key)
        effective_idx = self._compute_index(index_tuple)

        match (effective_idx, entry.is_static):
            case ((), _):
                self._write_full_column(field_key, entry, data, key)
            case (_, True):
                self._write_indexed_static(field_key, entry, effective_idx, data, key)
            case (_, False):
                self._write_indexed_dynamic(field_key, entry, effective_idx, data)

    def _validate_dtype(self, data: Any, target_dtype: Any, key: KeyPath) -> None:
        """Raise if ``data`` cannot be cast to ``target_dtype``.

        A sequence of arrays is represented by its first element; Awkward
        Arrays are not checked.
        """
        match data:
            case ak.Array():
                return
            case list() | tuple() if len(data) == 0:
                return
            case list() | tuple() if isinstance(
                data[0],
                (ak.Array, np.ndarray, list, tuple),
            ):
                self._validate_dtype(data[0], target_dtype, key)
            case np.ndarray():
                check_castable(data.dtype, target_dtype, key)
            case _:
                representative = as_regular_numpy(data, None)
                if representative is not None:
                    check_castable(representative.dtype, target_dtype, key)

    def _write_full_column(
        self,
        field_key: FieldKey,
        entry: ArrayEntryBase,
        data: Any,
        key: KeyPath,
    ) -> None:
        """Replace a whole column, validating its length for Set fields."""
        if isinstance(entry, SetEntryBase):
            self._check_column_length(field_key, entry, len(data), key)

        self._chunks.pop(field_key, None)
        self._data[field_key] = self._allocate_buffer(data, entry, key)

    def _check_column_length(
        self,
        field_key: FieldKey,
        entry: SetEntryBase,
        length: int,
        key: KeyPath,
    ) -> None:
        """Validate a new column against the capacity or its sibling columns.

        The comparison with the siblings is skipped when Python runs with -O.
        """
        match entry.capacity:
            case int(capacity) if length > capacity:
                raise ValueError(
                    f"Shape mismatch for key '{key}': length {length} exceeds capacity {capacity}.",
                )
            case None if __debug__:
                mismatched = [
                    len(column)
                    for column in self._siblings(field_key)
                    if len(column) != length
                ]
                if mismatched:
                    raise ValueError(
                        f"Length mismatch for key '{key}': data length {length} "
                        f"does not match existing column length {mismatched[0]}.",
                    )
            case _:
                pass

    def _allocate_buffer(
        self,
        data: Any,
        entry: ArrayEntryBase,
        key: KeyPath,
    ) -> Column:
        """Build the buffer of a whole column from the written data.

        Sequences of rows are stacked and padded to the capacity of the
        field. Other data is converted as a whole; static fields additionally
        check the element shape. Regular data becomes a NumPy array, also for
        dynamic fields; an Awkward Array that only wraps a NumPy array is
        unwrapped without copying.
        """
        capacity = entry.capacity if isinstance(entry, SetEntryBase) else None

        match data:
            case ak.Array() if entry.is_static:
                raise ValueError(f"Shape mismatch for static field '{key}'.")

            case ak.Array() if has_numpy_layout(data):
                return ak.to_numpy(data)

            case ak.Array():
                return data

            case list() | tuple() if len(data) == 0:
                return pad_to_capacity(self._empty_column(entry), capacity, entry.dtype)

            case list() | tuple():
                return self._allocate_from_rows(data, entry, capacity, key)

            case _:
                return self._allocate_from_array(data, entry, key)

    @staticmethod
    def _empty_column(entry: ArrayEntryBase) -> np.ndarray:
        """A column without rows; dynamic dimensions get length 0."""
        element_shape = tuple(0 if dim is None else dim for dim in entry.shape)
        return np.zeros((0, *element_shape), dtype=entry.dtype)

    def _allocate_from_rows(
        self,
        rows: Sequence[Any],
        entry: ArrayEntryBase,
        capacity: Optional[int],
        key: KeyPath,
    ) -> Column:
        """Allocate a buffer from a list or tuple of rows."""
        stacked = stack_sequence(rows)
        match stacked:
            case np.ndarray():
                return pad_to_capacity(stacked, capacity, entry.dtype)
            case ak.Array():
                return pad_to_capacity(stacked, capacity, entry.dtype)
            case None:
                pass

        arr = as_regular_numpy(rows, entry.dtype)
        match arr:
            case None if entry.is_static:
                raise ValueError(f"Shape mismatch for static field '{key}'.")
            case None:
                return ak.Array(rows)
            case _:
                return pad_to_capacity(arr, capacity, entry.dtype)

    @staticmethod
    def _allocate_from_array(data: Any, entry: ArrayEntryBase, key: KeyPath) -> Column:
        """Allocate a buffer from a scalar or an array."""
        arr = as_regular_numpy(data, entry.dtype)
        match arr:
            case None if entry.is_static:
                raise ValueError(f"Shape mismatch for static field '{key}'.")
            case None:
                return ak.Array(data)
            case _ if entry.is_static:
                return coerce_static_shape(
                    arr,
                    entry.shape,
                    key,
                    is_set=isinstance(entry, SetEntryBase),
                )
            case _:
                return arr

    def _write_indexed_static(
        self,
        field_key: FieldKey,
        entry: ArrayEntryBase,
        effective_idx: Tuple[IndexType, ...],
        data: Any,
        key: KeyPath,
    ) -> None:
        """Indexed write on a static (fixed-shape) field with auto-allocation and expansion."""
        buffer = self._writable_static_buffer(field_key, entry, effective_idx)
        arr = as_regular_numpy(data, buffer.dtype)
        if arr is None or not broadcastable(buffer[effective_idx].shape, arr.shape):
            raise ValueError(
                f"Shape mismatch writing to static field '{key}'.",
            )

        buffer[effective_idx] = arr

    def _writable_static_buffer(
        self,
        field_key: FieldKey,
        entry: ArrayEntryBase,
        effective_idx: Tuple[IndexType, ...],
    ) -> np.ndarray:
        """Return a writable buffer of a static column that covers ``effective_idx``.

        Allocating, expanding or copying a read-only buffer replaces the column
        object; this happens once under the lock. Otherwise the buffer is
        returned lock-free.
        """
        buffer = self._data.get(field_key)
        if self._is_writable_for(buffer, effective_idx):
            assert isinstance(buffer, np.ndarray)
            return buffer

        with self._lock:
            buffer = self._data.get(field_key)
            match buffer:
                case None:
                    outer_dims = infer_outer_dims(
                        entry,
                        effective_idx,
                        min_length=self._sibling_length(field_key),
                    )
                    element_shape = cast(Tuple[int, ...], entry.shape)
                    buffer = np.zeros((*outer_dims, *element_shape), dtype=entry.dtype)
                case np.ndarray() if self._is_writable_for(buffer, effective_idx):
                    return buffer
                case np.ndarray():
                    # Expansion allocates a new buffer. Without expansion, the
                    # buffer was adopted zero-copy from Arrow (e.g. after
                    # unpickling) and is read-only; copy it once.
                    expanded = expand_to_fit(buffer, effective_idx)
                    buffer = expanded if expanded is not buffer else buffer.copy()
                case _:
                    raise TypeError(
                        f"Static field buffer must be a numpy array, got '{type(buffer).__name__}'.",
                    )
            self._data[field_key] = buffer
            return buffer

    @staticmethod
    def _is_writable_for(
        buffer: Optional[Column],
        effective_idx: Tuple[IndexType, ...],
    ) -> bool:
        return (
            isinstance(buffer, np.ndarray)
            and buffer.flags.writeable
            and expanded_shape(buffer.shape, effective_idx) == buffer.shape
        )

    def _write_indexed_dynamic(
        self,
        field_key: FieldKey,
        entry: ArrayEntryBase,
        effective_idx: Tuple[IndexType, ...],
        data: Any,
    ) -> None:
        """Indexed write on a dynamic field, staged as rows for deferred materialization."""
        rows = self._row_chunks(field_key, entry, effective_idx)
        assign_rows(rows, effective_idx, data)

    def _row_chunks(
        self,
        field_key: FieldKey,
        entry: ArrayEntryBase,
        effective_idx: Tuple[IndexType, ...],
    ) -> List[Any]:
        """Return the per-row chunk list of a dynamic column that covers ``effective_idx``.

        Splitting a column into rows, allocating and extending the list happen
        once under the lock. Otherwise the list is returned lock-free.
        """
        chunks = self._chunks.get(field_key)
        if chunks is not None and self._covers(chunks, effective_idx):
            return chunks

        with self._lock:
            chunks = self._chunks.get(field_key)
            match chunks:
                case None if field_key in self._data:
                    chunks = split_rows(self._data[field_key])
                    # Publish the rows before removing the column, so that
                    # concurrent readers always find the field.
                    self._chunks[field_key] = chunks
                    del self._data[field_key]
                case None:
                    outer_dims = infer_outer_dims(
                        entry,
                        effective_idx,
                        min_length=self._sibling_length(field_key),
                    )
                    chunks = nested_rows(outer_dims)
                    self._chunks[field_key] = chunks
                case _:
                    pass

            match effective_idx:
                case (int(idx), *_) if idx >= len(chunks):
                    chunks.extend([None] * (idx + 1 - len(chunks)))
                case _:
                    pass
            return chunks

    @staticmethod
    def _covers(chunks: List[Any], effective_idx: Tuple[IndexType, ...]) -> bool:
        match effective_idx:
            case (int(idx), *_):
                return idx < len(chunks)
            case _:
                return True
