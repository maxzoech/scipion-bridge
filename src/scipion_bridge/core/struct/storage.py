"""Unified Arrow-backed storage engine for Struct and Set data structures."""

from __future__ import annotations

import abc
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike
import pyarrow as pa

from .schema import (
    Schema,
    Entry,
    KeyPath,
    ArrayEntryBase,
    ArrayEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
    _ArrayEntryBase,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
)
from .utils.arrow_utils import (
    RaggedArrayView,
    build_tensor_array,
    build_ragged_array,
)

IndexType: TypeAlias = Union[slice, int]
Offset: TypeAlias = Optional[Tuple[IndexType, ...]]


def _lookup_entry(schema: Schema, path: KeyPath) -> Optional[ArrayEntryBase]:
    """Look up a leaf ArrayEntryBase in a schema by KeyPath tuple."""
    match path:
        case ("root", *tail):
            return _lookup_in_schema(schema, tuple(tail))
        case _:
            return _lookup_in_schema(schema, path)


def _lookup_in_schema(schema: Schema, path: KeyPath) -> Optional[ArrayEntryBase]:
    match path:
        case (head,):
            entry = schema.fields.get(head)
            return entry if isinstance(entry, ArrayEntryBase) else None
        case (head, *tail):
            entry = schema.fields.get(head)
            if entry is not None and entry.children is not None:
                return _lookup_in_schema(entry.children, tuple(tail))
            return None
        case _:
            return None


def _get_nested_col(batch: Union[pa.RecordBatch, pa.StructArray], path: KeyPath) -> pa.Array:
    """Navigate a KeyPath tuple in a nested Arrow RecordBatch or StructArray."""
    curr: Any = batch
    for segment in path:
        if isinstance(curr, pa.RecordBatch):
            curr = curr.column(segment)
        elif isinstance(curr, pa.StructArray):
            curr = curr.field(segment)
        else:
            raise KeyError(f"Cannot resolve path segment '{segment}' in '{type(curr).__name__}'.")
    return curr


class _BaseStorage(abc.ABC):
    """Abstract base storage class managing offset tracking, key qualification, and read/write."""

    def __init__(
        self,
        schema: Schema,
        parent: Optional["_BaseStorage"] = None,
        path: KeyPath = ("root",),
        offset: Offset = None,
    ) -> None:
        self._schema = schema
        self.parent = parent
        self.offset = offset
        self.path: KeyPath = tuple(path)

    @property
    def is_view(self) -> bool:
        return self.parent is not None

    @property
    def root_storage(self) -> "ArrayStorage":
        curr: _BaseStorage = self
        while curr.parent is not None:
            curr = curr.parent
        return cast("ArrayStorage", curr)

    def qualify_path(self, key: Union[str, KeyPath]) -> KeyPath:
        if isinstance(key, str):
            return (*self.path, key)
        return (*self.path, *key)

    def compute_slice_offset(self, start: int, stop: int) -> Tuple[IndexType, ...]:
        """Compute the offset tuple when slicing a range along the active batch dimension."""
        if not self.offset:
            return (slice(start, stop),)

        *prefix, last = self.offset
        match last:
            case slice() as base:
                base_start = base.start or 0
                new_start = base_start + start
                new_stop = base_start + stop
                return (*prefix, slice(new_start, new_stop))
            case _:
                raise ValueError(
                    f"Cannot slice dimension '{last}' that is already an indexed integer."
                )

    def compute_index_offset(self, index: int) -> Tuple[IndexType, ...]:
        """Compute the offset tuple when selecting an element index along the active batch dimension."""
        if not self.offset:
            return (index,)

        *prefix, last = self.offset
        match last:
            case slice() as base:
                base_start = base.start or 0
                return (*prefix, base_start + index)
            case _:
                raise ValueError(
                    f"Cannot index dimension '{last}' that is already an indexed integer."
                )

    def descend_set_offset(self) -> Tuple[IndexType, ...]:
        """Compute the offset tuple when descending into a child Set dimension."""
        return (*(self.offset or ()), slice(None, None, None))

    def schema(self) -> Schema:
        return self._schema

    def read(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        """Read data for the specified schema entry at the given offset."""
        full_path = self.qualify_path(key)
        effective_offset = self.offset if offset is None else offset
        return self.root_storage._engine.read(full_path, entry, offset=effective_offset)

    def write(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        """Write data for the specified schema entry at the given offset."""
        full_path = self.qualify_path(key)
        effective_offset = self.offset if offset is None else offset
        self.root_storage._engine.write(full_path, entry, data, offset=effective_offset)

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch:
        """Freeze and compile the storage into an immutable Arrow RecordBatch."""
        ...

    def __contains__(self, key: Union[str, KeyPath]) -> bool:
        full_path = self.qualify_path(key)
        return full_path in self.root_storage._engine


class _StorageEngine(abc.ABC):
    """Abstract interface for polymorphic storage engines."""

    @abc.abstractmethod
    def read(self, key: KeyPath, entry: Entry, offset: Offset = None) -> Any:
        ...

    @abc.abstractmethod
    def write(self, key: KeyPath, entry: Entry, data: Any, offset: Offset = None) -> None:
        ...

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch:
        ...

    @abc.abstractmethod
    def __contains__(self, key: KeyPath) -> bool:
        ...

    @property
    @abc.abstractmethod
    def capacity(self) -> Optional[int]:
        ...

    @property
    @abc.abstractmethod
    def is_frozen(self) -> bool:
        ...


class _StagingEngine(_StorageEngine):
    """Mutable staging engine backed by in-memory NumPy dictionaries."""

    def __init__(self, schema: Schema, capacity: Optional[int] = None) -> None:
        self._schema = schema
        self._capacity = capacity

        self._static_staging: Dict[KeyPath, np.ndarray] = {}
        self._ragged_staging: Dict[KeyPath, List[Optional[np.ndarray]]] = {}

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

    @property
    def is_frozen(self) -> bool:
        return False

    def _resolve_key(self, key: KeyPath) -> KeyPath:
        if key in self._static_staging or key in self._ragged_staging:
            return key
        if key and key[0] != "root":
            root_key = ("root", *key)
            if root_key in self._static_staging or root_key in self._ragged_staging:
                return root_key
        elif key and key[0] == "root":
            tail_key = key[1:]
            if tail_key in self._static_staging or tail_key in self._ragged_staging:
                return tail_key
        return key

    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        key = self._resolve_key(key)
        schema_entry = _lookup_entry(self._schema, key) or entry

        match schema_entry:
            case RaggedArraySetEntry():
                self._write_ragged(key, schema_entry, data, offset=offset)
            case ArrayEntry() | ArraySetEntry():
                self._write_static(key, schema_entry, data, offset=offset)
            case _:
                raise TypeError(
                    f"Unsupported schema entry '{type(entry).__name__}' for write to key '{key}'."
                )

    def _write_static(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        data: ArrayLike,
        offset: Offset = None,
    ) -> None:
        key = self._resolve_key(key)
        arr = np.asanyarray(data)
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )

        arr = arr.astype(entry.dtype, copy=False)

        is_sub_offset = (
            offset is not None
            and offset != ()
            and any(
                isinstance(idx, int)
                or (
                    isinstance(idx, slice)
                    and (idx.start is not None or idx.stop is not None or idx.step is not None)
                )
                for idx in offset
            )
        )

        if is_sub_offset:
            assert offset is not None
            int_dims = tuple(d for d in entry.shape if d is not None)
            if key not in self._static_staging:
                if isinstance(entry, ArraySetEntry):
                    idx_req = 1
                    if offset and isinstance(offset[0], int):
                        idx_req = offset[0] + 1
                    elif offset and isinstance(offset[0], slice) and offset[0].stop is not None:
                        idx_req = offset[0].stop

                    cap = entry.capacity or self.capacity or idx_req
                    cap = max(cap, idx_req)
                    self._static_staging[key] = np.zeros((cap, *int_dims), dtype=entry.dtype)
                else:
                    self._static_staging[key] = np.zeros(int_dims, dtype=entry.dtype)
            else:
                if isinstance(entry, ArraySetEntry):
                    current_buf = self._static_staging[key]
                    idx_req = 1
                    if offset and isinstance(offset[0], int):
                        idx_req = offset[0] + 1
                    elif offset and isinstance(offset[0], slice) and offset[0].stop is not None:
                        idx_req = offset[0].stop
                    if idx_req > len(current_buf):
                        new_cap = max(idx_req, len(current_buf) * 2)
                        new_buf = np.zeros((new_cap, *int_dims), dtype=entry.dtype)
                        new_buf[:len(current_buf)] = current_buf
                        self._static_staging[key] = new_buf

            self._static_staging[key][offset] = arr
            return

        # Full column / root write validation
        if isinstance(entry, ArraySetEntry):
            entry_ndim = len(entry.shape)
            if arr.ndim < entry_ndim:
                raise ValueError(
                    f"Shape mismatch for key '{key}': expected at least {entry_ndim} dimensions, "
                    f"got data shape {arr.shape}."
                )
            for dim_idx, (expected_dim, actual_dim) in enumerate(zip(entry.shape, arr.shape[-entry_ndim:])):
                if expected_dim is not None and expected_dim != actual_dim:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': expected dimension {dim_idx} to be {expected_dim}, "
                        f"got {actual_dim}."
                    )
            
        elif isinstance(entry, ArrayEntry):
            if arr.ndim != len(entry.shape):
                raise ValueError(
                    f"Dimension count mismatch for key '{key}': expected {len(entry.shape)} dimensions, "
                    f"got {arr.ndim} (shape {arr.shape})."
                )
            for dim_idx, (expected_dim, actual_dim) in enumerate(zip(entry.shape, arr.shape)):
                if expected_dim is not None and expected_dim != actual_dim:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': expected dimension {dim_idx} to be {expected_dim}, "
                        f"got {actual_dim}."
                    )

        self._static_staging[key] = arr

    def _write_ragged(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        key = self._resolve_key(key)
        capacity = self.capacity or 0
        if not self.capacity and isinstance(data, (list, tuple, RaggedArrayView)):
            capacity = max(capacity, len(data))

        if key not in self._ragged_staging:
            self._ragged_staging[key] = [None] * capacity

        match offset:
            case (int(idx),):
                slots = self._ragged_staging[key]
                if len(slots) <= idx:
                    slots.extend([None] * (idx + 1 - len(slots)))

                slots[idx] = np.ascontiguousarray(data, dtype=entry.dtype)

            case (slice() as sl,):
                for i, item in zip(range(*sl.indices(capacity)), data):
                    self._write_ragged(key, entry, item, offset=(i,))

            # Recursive case 2: decompose bulk column write into per-element writes
            case None | ():
                if not isinstance(data, (list, tuple, RaggedArrayView)):
                    raise ValueError

                for i, item in enumerate(data):
                    self._write_ragged(key, entry, item, offset=(i,))

            case _:
                raise TypeError(f"Unsupported offset pattern '{offset}' for ragged write.")

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        key = self._resolve_key(key)
        schema_entry = _lookup_entry(self._schema, key) or entry

        if isinstance(schema_entry, RaggedArraySetEntry):
            return self._read_ragged(key, schema_entry, offset=offset)
        elif isinstance(schema_entry, (ArrayEntry, ArraySetEntry)):
            return self._read_static(key, schema_entry, offset=offset)
        else:
            raise TypeError(
                f"Unsupported schema entry '{type(entry).__name__}' for read of key '{key}'."
            )

    def _prepare_entry(self, key: KeyPath) -> None:
        key = self._resolve_key(key)
        if key in self._static_staging:
            return

        schema_entry = _lookup_entry(self._schema, key)
        assert isinstance(schema_entry, (ArrayEntry, ArraySetEntry))

        if not schema_entry.is_static and isinstance(schema_entry, ArrayEntry):
            raise AttributeError(
                f"Field '{key}' is dynamic and has not been initialized."
            )

    def _read_static(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        offset: Offset = None,
    ) -> ArrayLike:
        key = self._resolve_key(key)
        if key not in self._static_staging:
            schema_entry = _lookup_entry(self._schema, key)
            if schema_entry is not None and not schema_entry.is_static and isinstance(schema_entry, ArrayEntry):
                raise AttributeError(
                    f"Field '{key}' is dynamic and has not been initialized."
                )
            raise AttributeError(f"Field '{key}' has not been initialized.")

        arr = self._static_staging[key]
        if offset is None or offset == ():
            return arr

        return arr[offset]

    def _read_ragged(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        offset: Offset = None,
    ) -> Any:
        key = self._resolve_key(key)
        match offset:
            case (int(index),):
                return self._ragged_staging[key][index]

            case (slice() as sl,):
                if key not in self._ragged_staging:
                    raise AttributeError(f"Field '{key}' has not been initialized.")
                chunks = self._ragged_staging[key][sl]
                return RaggedArrayView(build_ragged_array(chunks, entry.dtype), entry.dtype)

            case None | ():
                if key not in self._ragged_staging:
                    raise AttributeError(f"Field '{key}' has not been initialized.")

                chunks = self._ragged_staging[key]
                return RaggedArrayView(build_ragged_array(chunks, entry.dtype), entry.dtype)

            case _:
                raise NotImplementedError

    def to_record_batch(self) -> pa.RecordBatch:
        columns, names = self._build_columns(self._schema, prefix=())
        return pa.RecordBatch.from_arrays(columns, names)

    def _build_columns(self, schema: Schema, prefix: KeyPath = ()) -> Tuple[List[pa.Array], List[str]]:
        columns: List[pa.Array] = []
        names: List[str] = []

        raise NotImplementedError

    def __contains__(self, key: KeyPath) -> bool:
        resolved = self._resolve_key(key)
        return resolved in self._static_staging or resolved in self._ragged_staging


class _ArrowEngine(_StorageEngine):
    """Immutable zero-copy storage engine wrapping an Apache Arrow RecordBatch."""

    def __init__(self, schema: Schema, batch: pa.RecordBatch) -> None:
        self._schema = schema
        self._batch = batch

    @property
    def capacity(self) -> Optional[int]:
        return len(self._batch)

    @property
    def is_frozen(self) -> bool:
        return True

    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        raise RuntimeError("Cannot mutate a frozen Set.")

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        raise NotImplementedError

    def to_record_batch(self) -> pa.RecordBatch:
        return self._batch

    def __contains__(self, key: KeyPath) -> bool:
        try:
            _get_nested_col(self._batch, key)
            return True
        except (KeyError, IndexError):
            return False


class ArrayStorage(_BaseStorage):
    """Unified Arrow-backed storage engine delegating to polymorphic _StorageEngine implementations."""

    def __init__(
        self,
        schema: Schema,
        capacity: Optional[int] = None,
        record_batch: Optional[pa.RecordBatch] = None,
        path: KeyPath = ("root",),
        offset: Offset = None,
    ) -> None:
        super().__init__(schema=schema, parent=None, path=path, offset=offset)
        self._schema = schema

        if record_batch is not None:
            self._engine: _StorageEngine = _ArrowEngine(schema, record_batch)
        else:
            self._engine = _StagingEngine(schema, capacity=capacity)

    @property
    def capacity(self) -> Optional[int]:
        return self._engine.capacity

    @property
    def is_frozen(self) -> bool:
        return self._engine.is_frozen

    def to_record_batch(self) -> pa.RecordBatch:
        """Freeze and compile the storage into an immutable Arrow RecordBatch."""
        batch = self._engine.to_record_batch()
        if not self._engine.is_frozen:
            # Transition to immutable Arrow engine and drop staging buffers
            self._engine = _ArrowEngine(self._schema, batch)
        return batch

    @classmethod
    def from_record_batch(cls, batch: pa.RecordBatch, schema: Schema) -> "ArrayStorage":
        """Construct an ArrayStorage directly wrapping a frozen Arrow RecordBatch."""
        return cls(schema=schema, capacity=len(batch), record_batch=batch)


class ArrayStorageView(_BaseStorage):
    """Sub-view into a parent storage with qualified path prefix and offset propagation."""

    def qualify_key(self, key: Any) -> Any:
        return self.qualify_path(key)

    def to_record_batch(self) -> pa.RecordBatch:
        assert self.parent is not None
        parent_batch = self.parent.to_record_batch()
        raise NotImplementedError


# Aliases for backward compatibility
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView