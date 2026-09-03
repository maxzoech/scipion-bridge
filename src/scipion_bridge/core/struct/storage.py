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


def _lookup_entry(schema: Schema, key: str) -> Optional[ArrayEntryBase]:
    """Look up a leaf ArrayEntryBase in a schema by dotted key path."""
    head, _, tail = key.partition(".")

    match schema.fields.get(head), tail:
        case ArrayEntryBase() as leaf, "":
            return leaf
        
        case Entry(children=Schema() as child), rest:
            return _lookup_entry(child, rest)
        
        case _:
            return None


def _get_nested_col(batch: Union[pa.RecordBatch, pa.StructArray], key: str) -> pa.Array:
    """Navigate dotted path in a nested Arrow RecordBatch or StructArray."""
    head, _, tail = key.partition(".")

    match batch:
        case pa.RecordBatch():
            col = batch.column(head)
        case pa.StructArray():
            col = batch.field(head)
        case _:
            raise KeyError(f"Cannot resolve key segment '{head}' in '{type(batch).__name__}'.")

    return _get_nested_col(col, tail) if tail else col


class _BaseStorage(abc.ABC):
    """Abstract base storage class managing offset tracking, key qualification, and read/write."""

    def __init__(
        self,
        schema: Schema,
        parent: Optional["_BaseStorage"] = None,
        path: Tuple[str, ...] = (),
        offset: Offset = None,
    ) -> None:
        self._schema = schema
        self.parent = parent
        self.offset = offset
        self.path = tuple(path)

    @property
    def root(self) -> str:
        return ".".join(self.path)

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
        key: str,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        """Read data for the specified schema entry at the given offset."""
        ...

    @abc.abstractmethod
    def write(
        self,
        key: str,
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
    def __contains__(self, key: str) -> bool:
        ...


class _StorageEngine(abc.ABC):
    """Abstract interface for polymorphic storage engines."""

    @abc.abstractmethod
    def read(self, key: str, entry: Entry, offset: Offset = None) -> Any:
        ...

    @abc.abstractmethod
    def write(self, key: str, entry: Entry, data: Any, offset: Offset = None) -> None:
        ...

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch:
        ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
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

        self._static_staging: Dict[str, np.ndarray] = {}
        self._ragged_staging: Dict[str, List[Optional[np.ndarray]]] = {}

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

    @property
    def is_frozen(self) -> bool:
        return False

    def write(
        self,
        key: str,
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
        key: str,
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
        key: str,
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
        key: str,
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

    def _prepare_entry(self, key: str):
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
        key: str,
        entry: Union[ArrayEntry, ArraySetEntry],
        offset: Offset = None,
    ) -> ArrayLike:

        if isinstance(entry, ArraySetEntry):
            raise NotImplementedError
        
        if key not in self._static_staging:
            if not entry.is_static and isinstance(entry, ArrayEntry):
                raise AttributeError(
                    f"Field '{key}' is dynamic and has not been initialized."
                )

            entry_shape = tuple(dim for dim in entry.shape if dim is not None)
            assert len(entry_shape) == len(entry.shape), "Some dimensions in entry.shape were None"

            self._static_staging[key] = np.zeros(entry_shape, dtype=entry.dtype)
        
        array = self._static_staging[key]

        offset = offset or tuple()
        return array[offset]
        
        # if key not in self._static_staging:
        #     schema_entry = _lookup_entry(self._schema, key) or entry
        
        #     entry_shape = tuple(dim for dim in schema_entry.shape if dim is not None)
        #     if self._capacity is not None and isinstance(schema_entry, ArraySetEntry):
        #         shape = (self._capacity, *entry_shape)
        #     else:
        #         shape = entry_shape
        #     self._static_staging[key] = np.zeros(shape, dtype=schema_entry.dtype)
        # arr = self._static_staging[key]

        # if offset:
        #     return arr[offset]
        # return arr

    def _read_ragged(
        self,
        key: str,
        entry: RaggedArraySetEntry,
        offset: Offset = None,
    ) -> Any:
        raise NotImplementedError
        
        # if offset is not None:
        #     idx = offset[0]
        #     if isinstance(idx, int):
        #         chunks = self._ragged_staging.get(key, [])
        #         if idx < len(chunks) and chunks[idx] is not None:
        #             return chunks[idx]
        #         return np.empty(0, dtype=entry.dtype)

        # chunks = self._ragged_staging.get(key, [])
        # if self._capacity is not None and len(chunks) < self._capacity:
        #     chunks = list(chunks) + [None] * (self._capacity - len(chunks))
        # col = build_ragged_array(chunks, dtype=entry.dtype)

        # if not isinstance(col, pa.ListArray):
        #     raise TypeError(f"Expected ListArray for ragged field '{key}', got {type(col).__name__}")
        # return RaggedArrayView(col, dtype=entry.dtype)

    def to_record_batch(self) -> pa.RecordBatch:
        columns, names = self._build_columns(self._schema, prefix="")
        return pa.RecordBatch.from_arrays(columns, names)

    def _build_columns(self, schema: Schema, prefix: str) -> Tuple[List[pa.Array], List[str]]:
        columns: List[pa.Array] = []
        names: List[str] = []

        raise NotImplementedError

        for field_name, entry in schema.fields.items():
            full_key = f"{prefix}.{field_name}" if prefix else field_name
            names.append(field_name)

            if entry.children is not None:
                child_cols, child_names = self._build_columns(entry.children, prefix=full_key)
                struct_col = pa.StructArray.from_arrays(child_cols, child_names)
                columns.append(struct_col)
            elif isinstance(entry, ArrayEntryBase):
                if entry.is_static:
                    arr = self._static_staging.get(full_key)
                    entry_shape = tuple(dim for dim in entry.shape if dim is not None)
                    if arr is None:
                        cap = self._capacity or 1
                        arr = np.zeros((cap, *entry_shape), dtype=entry.dtype)

                    if self._capacity is not None and arr.shape[0] != self._capacity:
                        full_shape = (self._capacity, *entry_shape)
                        padded = np.zeros(full_shape, dtype=entry.dtype)
                        copy_len = min(arr.shape[0], self._capacity)
                        padded[:copy_len] = arr[:copy_len]
                        arr = padded

                    col = build_tensor_array(arr, shape=entry_shape, dtype=entry.dtype)
                    columns.append(col)
                else:
                    chunks = self._ragged_staging.get(full_key, [])
                    if self._capacity is not None and len(chunks) < self._capacity:
                        chunks = list(chunks) + [None] * (self._capacity - len(chunks))
                    col = build_ragged_array(chunks, dtype=entry.dtype)
                    columns.append(col)
            else:
                raise TypeError(
                    f"Unsupported schema entry '{type(entry).__name__}' for field '{field_name}'."
                )

        return columns, names

    def __contains__(self, key: str) -> bool:
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
        key: str,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        raise RuntimeError("Cannot mutate a frozen Set.")

    def read(
        self,
        key: str,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        raise NotImplementedError
        
        schema_entry = _lookup_entry(self._schema, key) or entry

        if isinstance(schema_entry, RaggedArraySetEntry):
            col = _get_nested_col(self._batch, key)
            if not isinstance(col, pa.ListArray):
                raise TypeError(f"Expected ListArray for ragged column '{key}', got {type(col).__name__}")
            if offset is not None:
                idx = offset[0]
                if isinstance(idx, int):
                    scalar = col[idx]
                    return scalar.values.to_numpy() if scalar.is_valid else np.empty(0, dtype=schema_entry.dtype)
            return RaggedArrayView(col, dtype=schema_entry.dtype)

        elif isinstance(schema_entry, (ArrayEntry, ArraySetEntry)):        
            col = _get_nested_col(self._batch, key)
            if hasattr(col, "to_numpy_ndarray"):
                arr = col.to_numpy_ndarray()
            else:
                arr = col.to_numpy()
            if offset:
                return arr[offset]
            return arr

        else:
            raise TypeError(
                f"Unsupported schema entry '{type(entry).__name__}' for read of key '{key}'."
            )

    def to_record_batch(self) -> pa.RecordBatch:
        return self._batch

    def __contains__(self, key: str) -> bool:
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
        key: str,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        """Write data via the active storage engine."""
        self._engine.write(key, entry, data, offset=offset)

    def read(
        self,
        key: str,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        """Read data via the active storage engine."""
        return self._engine.read(key, entry, offset=offset)

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

    def __contains__(self, key: str) -> bool:
        return key in self._engine


class ArrayStorageView(_BaseStorage):
    """Sub-view into a parent storage with qualified path prefix and offset propagation."""

    def qualify_key(self, key: str) -> str:
        return ".".join((*self.path, key)) if self.path else key

    def _resolve(self, key: str, offset: Offset) -> Tuple[str, Offset]:
        full_key = self.qualify_key(key)
        merged = self.offset if offset is None else (*(self.offset or ()), *offset)
        return full_key, merged

    def read(self, key: str, entry: Entry, offset: Offset = None) -> Any:
        assert self.parent is not None
        
        full_key, effective_offset = self._resolve(key, offset)
        return self.parent.read(full_key, entry, offset=effective_offset)

    def write(self, key: str, entry: Entry, data: Any, offset: Offset = None) -> None:
        assert self.parent is not None

        full_key, effective_offset = self._resolve(key, offset)
        self.parent.write(full_key, entry, data, offset=effective_offset)

    def to_record_batch(self) -> pa.RecordBatch:
        assert self.parent is not None

        parent_batch = self.parent.to_record_batch()

        raise NotImplementedError
        # if self.path:
        #     col = _get_nested_col(parent_batch, ".".join(self.path))
        #     if isinstance(col, pa.StructArray):
        #         batch = pa.RecordBatch.from_arrays(
        #             [col.field(i) for i in range(col.type.num_fields)],
        #             [col.type.field(i).name for i in range(col.type.num_fields)],
        #         )
        #     else:
        #         batch = parent_batch
        # else:
        #     batch = parent_batch

        # if self.offset and isinstance(self.offset[0], slice):
        #     sl = self.offset[0]
        #     start = sl.start or 0
        #     stop = sl.stop if sl.stop is not None else len(batch)
        #     return batch.slice(start, max(0, stop - start))

        # return batch

    def __contains__(self, key: str) -> bool:
        if self.parent is None:
            return False
        return self.qualify_key(key) in self.parent


# Aliases for backward compatibility
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView