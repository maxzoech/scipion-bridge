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
    def __contains__(self, key: str) -> bool:
        ...


class ArrayStorage(_BaseStorage):
    """Unified Arrow-backed storage engine for Struct and Set containers."""

    def __init__(
        self,
        schema: Schema,
        capacity: Optional[int] = None,
        record_batch: Optional[pa.RecordBatch] = None,
    ) -> None:
        super().__init__(schema=schema, parent=None, path=(), offset=None)
        self._schema = schema

        self._capacity = capacity
        if record_batch is not None:
            self._capacity = len(record_batch)

        self._frozen_batch: Optional[pa.RecordBatch] = record_batch

        self._static_staging: Dict[str, np.ndarray] = {}
        self._ragged_staging: Dict[str, List[Optional[np.ndarray]]] = {}


    @property
    def capacity(self) -> Optional[int]:
        if self._frozen_batch is not None:
            return len(self._frozen_batch)
        
        return self._capacity

    @property
    def is_frozen(self) -> bool:
        return self._frozen_batch is not None

    def _check_not_frozen(self) -> None:
        if self._frozen_batch is not None:
            raise RuntimeError("Cannot mutate a frozen Set.")

    def write(
        self,
        key: str,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        """Write data to static or ragged staging buffers based on the entry type."""
        self._check_not_frozen()
        schema_entry = _lookup_entry(self._schema, key) or entry

        if isinstance(schema_entry, RaggedArraySetEntry):
            self._write_ragged(key, schema_entry, data, offset=offset)
        elif isinstance(schema_entry, (ArrayEntry, ArraySetEntry)):
            self._write_static(key, schema_entry, data, offset=offset)
        else:
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
            if key not in self._static_staging:
                _ = self._read_static(key, entry)
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
        """Read data from frozen Arrow batch or mutable staging buffers."""
        schema_entry = _lookup_entry(self._schema, key) or entry

        if isinstance(schema_entry, RaggedArraySetEntry):
            return self._read_ragged(key, schema_entry, offset=offset)
        elif isinstance(schema_entry, (ArrayEntry, ArraySetEntry)):
            return self._read_static(key, schema_entry, offset=offset)
        else:
            raise TypeError(
                f"Unsupported schema entry '{type(entry).__name__}' for read of key '{key}'."
            )

    def _read_static(
        self,
        key: str,
        entry: Union[ArrayEntry, ArraySetEntry],
        offset: Offset = None,
    ) -> ArrayLike:
        if self._frozen_batch is not None:
            col = _get_nested_col(self._frozen_batch, key)
            if hasattr(col, "to_numpy_ndarray"):
                arr = col.to_numpy_ndarray()
            else:
                arr = col.to_numpy()
        else:
            if key not in self._static_staging:
                schema_entry = _lookup_entry(self._schema, key) or entry
                if not schema_entry.is_static and isinstance(schema_entry, ArrayEntry):
                    raise AttributeError(
                        f"Field '{key}' is dynamic and has not been initialized."
                    )
                entry_shape = tuple(dim for dim in schema_entry.shape if dim is not None)
                if self._capacity is not None and isinstance(schema_entry, ArraySetEntry):
                    shape = (self._capacity, *entry_shape)
                else:
                    shape = entry_shape
                self._static_staging[key] = np.zeros(shape, dtype=schema_entry.dtype)
            arr = self._static_staging[key]

        if offset:
            return arr[offset]
        return arr

    def _read_ragged(
        self,
        key: str,
        entry: RaggedArraySetEntry,
        offset: Offset = None,
    ) -> Any:
        if offset is not None:
            idx = offset[0]
            if isinstance(idx, int):
                if self._frozen_batch is not None:
                    col = _get_nested_col(self._frozen_batch, key)
                    if not isinstance(col, pa.ListArray):
                        raise TypeError(f"Expected ListArray for ragged column '{key}', got {type(col).__name__}")
                    scalar = col[idx]
                    return scalar.values.to_numpy() if scalar.is_valid else np.empty(0, dtype=entry.dtype)
                else:
                    chunks = self._ragged_staging.get(key, [])
                    if idx < len(chunks) and chunks[idx] is not None:
                        return chunks[idx]
                    return np.empty(0, dtype=entry.dtype)

        # Whole column read: return RaggedArrayView
        if self._frozen_batch is not None:
            col = _get_nested_col(self._frozen_batch, key)
        else:
            chunks = self._ragged_staging.get(key, [])
            if self._capacity is not None and len(chunks) < self._capacity:
                chunks = list(chunks) + [None] * (self._capacity - len(chunks))
            col = build_ragged_array(chunks, dtype=entry.dtype)

        if not isinstance(col, pa.ListArray):
            raise TypeError(f"Expected ListArray for ragged field '{key}', got {type(col).__name__}")
        return RaggedArrayView(col, dtype=entry.dtype)

    def to_record_batch(self) -> pa.RecordBatch:
        """Freeze and compile the storage into an immutable Arrow RecordBatch."""
        if self._frozen_batch is not None:
            return self._frozen_batch

        columns, names = self._build_columns(self._schema, prefix="")
        batch = pa.RecordBatch.from_arrays(columns, names)
        self._frozen_batch = batch
        return self._frozen_batch

    def _build_columns(self, schema: Schema, prefix: str) -> Tuple[List[pa.Array], List[str]]:
        columns: List[pa.Array] = []
        names: List[str] = []

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

    @classmethod
    def from_record_batch(cls, batch: pa.RecordBatch, schema: Schema) -> "ArrayStorage":
        """Construct an ArrayStorage directly wrapping a frozen Arrow RecordBatch."""
        return cls(schema=schema, capacity=len(batch), record_batch=batch)

    def __contains__(self, key: str) -> bool:
        if self._frozen_batch is not None:
            try:
                _get_nested_col(self._frozen_batch, key)
                return True
            except (KeyError, IndexError):
                return False
        return key in self._static_staging or key in self._ragged_staging


class ArrayStorageView(_BaseStorage):
    """Sub-view into a parent storage with qualified path prefix and offset propagation."""

    def qualify_key(self, key: str) -> str:
        return ".".join((*self.path, key)) if self.path else key

    def _resolve(self, key: str, offset: Offset) -> Tuple[str, Offset]:
        full_key = self.qualify_key(key)
        merged = self.offset if offset is None else (*(self.offset or ()), *offset)
        return full_key, merged

    def read(self, key: str, entry: Entry, offset: Offset = None) -> Any:
        if self.parent is None:
            raise RuntimeError("ArrayStorageView has no parent storage.")
        full_key, effective_offset = self._resolve(key, offset)
        return self.parent.read(full_key, entry, offset=effective_offset)

    def write(self, key: str, entry: Entry, data: Any, offset: Offset = None) -> None:
        if self.parent is None:
            raise RuntimeError("ArrayStorageView has no parent storage.")
        full_key, effective_offset = self._resolve(key, offset)
        self.parent.write(full_key, entry, data, offset=effective_offset)

    def to_record_batch(self) -> pa.RecordBatch:
        if self.parent is None:
            raise RuntimeError("ArrayStorageView has no parent storage.")
        parent_batch = self.parent.to_record_batch()

        if self.path:
            col = _get_nested_col(parent_batch, ".".join(self.path))
            if isinstance(col, pa.StructArray):
                batch = pa.RecordBatch.from_arrays(
                    [col.field(i) for i in range(col.type.num_fields)],
                    [col.type.field(i).name for i in range(col.type.num_fields)],
                )
            else:
                batch = parent_batch
        else:
            batch = parent_batch

        if self.offset and isinstance(self.offset[0], slice):
            sl = self.offset[0]
            start = sl.start or 0
            stop = sl.stop if sl.stop is not None else len(batch)
            return batch.slice(start, max(0, stop - start))

        return batch

    def __contains__(self, key: str) -> bool:
        if self.parent is None:
            return False
        return self.qualify_key(key) in self.parent


# Aliases for backward compatibility
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView