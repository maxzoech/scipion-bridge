"""Unified Arrow-backed storage engine for Struct and Set data structures."""

from __future__ import annotations

import abc
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeAlias

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
        case (head,):
            entry = schema.fields.get(head)
            return entry if isinstance(entry, ArrayEntryBase) else None
        case (head, *tail):
            entry = schema.fields.get(head)
            if entry is not None and entry.children is not None:
                return _lookup_entry(entry.children, tuple(tail))
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
        path: KeyPath = (),
        offset: Offset = None,
    ) -> None:
        self._schema = schema
        self.parent = parent
        self.offset = offset
        self.path: KeyPath = tuple(path)

    @property
    def is_view(self) -> bool:
        return self.parent is not None

    def compute_offset(self, item: IndexType) -> Tuple[IndexType, ...]:
        """Compute the offset tuple when indexing or slicing along the active batch dimension."""

        match self.offset, item:
            case (*prefix, slice() as base), slice() as current:
                base_start = base.start or 0

                start = base_start + (current.start or 0)
                stop = base_start + current.stop if current.stop is not None else None

                return (*prefix, slice(start, stop))

            case (*prefix, slice() as base), int(idx):
                return (*prefix, (base.start or 0) + idx)

            case _:
                return (item,)

    def compute_slice_offset(self, start: int, stop: int) -> Tuple[IndexType, ...]:
        return self.compute_offset(slice(start, stop))

    def compute_index_offset(self, index: int) -> Tuple[IndexType, ...]:
        return self.compute_offset(index)

    def schema(self) -> Schema:
        return self._schema

    @abc.abstractmethod
    def read(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        """Read data for the specified schema entry at the given offset."""
        ...

    @abc.abstractmethod
    def write(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        """Write data for the specified schema entry at the given offset."""
        ...

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch:
        """Freeze and compile the storage into an immutable Arrow RecordBatch."""
        ...

    @abc.abstractmethod
    def __contains__(self, key: Union[str, KeyPath]) -> bool:
        ...


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

    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
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
        arr = np.asanyarray(data)
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )

        arr = arr.astype(entry.dtype, copy=False)

        if offset:
            raise NotImplementedError
            # if key not in self._static_staging:
            #     _ = self._read_static(key, entry)

            # self._static_staging[key][offset] = arr
            # return

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
            if entry.capacity is not None and arr.shape[0] > entry.capacity:
                raise ValueError(
                    f"Capacity mismatch for key '{key}': data batch size {arr.shape[0]} exceeds capacity {entry.capacity}."
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
        if key not in self._ragged_staging:
            cap = self._capacity or 0
            self._ragged_staging[key] = [None] * cap

        if offset is not None:
            idx = offset[0]
            if isinstance(idx, int):
                while len(self._ragged_staging[key]) <= idx:
                    self._ragged_staging[key].append(None)
                self._ragged_staging[key][idx] = np.ascontiguousarray(data, dtype=entry.dtype)
            elif isinstance(idx, slice):
                base_start = idx.start or 0
                for i, d in enumerate(data):
                    target_i = base_start + i
                    while len(self._ragged_staging[key]) <= target_i:
                        self._ragged_staging[key].append(None)
                    self._ragged_staging[key][target_i] = np.ascontiguousarray(d, dtype=entry.dtype)
        else:
            if isinstance(data, (list, tuple, np.ndarray, RaggedArrayView)):
                self._ragged_staging[key] = [
                    np.ascontiguousarray(x, dtype=entry.dtype) for x in data
                ]
            else:
                self._ragged_staging[key] = [np.ascontiguousarray(data, dtype=entry.dtype)]

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
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
        

        match entry:
            case ArrayEntry():
                if key not in self._static_staging:
                    if not entry.is_static:
                        raise AttributeError(
                            f"Field '{key}' is dynamic and has not been initialized."
                        )

                    entry_shape = tuple(dim for dim in entry.shape if dim is not None)
                    assert len(entry_shape) == len(entry.shape), "Some dimensions in entry.shape were None"
                    assert offset is None

                    self._static_staging[key] = np.zeros(entry_shape, dtype=entry.dtype)

                return self._static_staging[key]

            case ArraySetEntry():
                assert key in self._static_staging
                if key not in self._static_staging:
                    raise AttributeError(
                        f"Field '{key}' not been initialized."
                    )

                return self._static_staging[key][offset]
                
            case _:
                raise NotImplementedError


    def _read_ragged(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        offset: Offset = None,
    ) -> Any:
        raise NotImplementedError

    def to_record_batch(self) -> pa.RecordBatch:
        columns, names = self._build_columns(self._schema, prefix=())
        return pa.RecordBatch.from_arrays(columns, names)

    def _build_columns(self, schema: Schema, prefix: KeyPath = ()) -> Tuple[List[pa.Array], List[str]]:
        columns: List[pa.Array] = []
        names: List[str] = []

        raise NotImplementedError

    def __contains__(self, key: KeyPath) -> bool:
        return key in self._static_staging or key in self._ragged_staging


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
    ) -> None:
        super().__init__(schema=schema, parent=None, path=(), offset=None)
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

    def write(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        """Write data via the active storage engine."""
        path = (key,) if isinstance(key, str) else key
        self._engine.write(path, entry, data, offset=offset)

    def read(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        """Read data via the active storage engine."""
        path = (key,) if isinstance(key, str) else key
        return self._engine.read(path, entry, offset=offset)

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

    def __contains__(self, key: Union[str, KeyPath]) -> bool:
        path = (key,) if isinstance(key, str) else key
        return path in self._engine


class ArrayStorageView(_BaseStorage):
    """Sub-view into a parent storage with qualified path prefix and offset propagation."""

    def qualify_path(self, key: Union[str, KeyPath]) -> KeyPath:
        if isinstance(key, str):
            return (*self.path, key)
        return (*self.path, *key)

    def qualify_key(self, key: Any) -> Any:
        return self.qualify_path(key)

    def _resolve(self, key: Union[str, KeyPath], offset: Offset) -> Tuple[KeyPath, Offset]:
        full_path = self.qualify_path(key)
        merged = self.offset if offset is None else (*(self.offset or ()), *offset)
        return full_path, merged

    def read(self, key: Union[str, KeyPath], entry: Entry, offset: Offset = None) -> Any:
        assert self.parent is not None
        full_path, effective_offset = self._resolve(key, offset)
        return self.parent.read(full_path, entry, offset=effective_offset)

    def write(self, key: Union[str, KeyPath], entry: Entry, data: Any, offset: Offset = None) -> None:
        assert self.parent is not None
        full_path, effective_offset = self._resolve(key, offset)
        self.parent.write(full_path, entry, data, offset=effective_offset)

    def to_record_batch(self) -> pa.RecordBatch:
        assert self.parent is not None
        parent_batch = self.parent.to_record_batch()
        raise NotImplementedError

    def __contains__(self, key: Union[str, KeyPath]) -> bool:
        if self.parent is None:
            return False
        return self.qualify_path(key) in self.parent


# Aliases for backward compatibility
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView