import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pyarrow as pa
from dependency_injector.wiring import Provide, inject

from scipion_bridge.backend.standalone.container import Container
from scipion_bridge.core.environment.storage import ArrayStorageProvider
from .schema import (
    Schema,
    _ArrayEntryBase,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaEntry,
    _SchemaSetEntry,
)
from .utils.arrow_utils import (
    RaggedArrayView,
    build_tensor_array,
    build_ragged_array,
)

IndexType = Union[slice, int]


class _BaseStorage(metaclass=abc.ABCMeta):

    def __init__(
        self,
        schema: Schema,
        parent: Optional["_BaseStorage"],
        path: Tuple[str, ...] = (),
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        super().__init__()
        self._schema = schema
        self.parent = parent
        self.offset = offset
        self.path = tuple(path)

    @property
    def root(self) -> str:
        return ".".join(self.path)

    def compute_offset(self, item: IndexType) -> Tuple[IndexType, ...]:
        """Compute the offset tuple when indexing or slicing along the active batch dimension."""
        offset = self.offset or ()

        if offset and isinstance(offset[-1], slice):
            base = offset[-1].start or 0
            new_last = (
                slice(base + item.start, base + item.stop, item.step)
                if isinstance(item, slice)
                else base + item
            )
            return (*offset[:-1], new_last)
        else:
            return (item,)

    def compute_slice_offset(self, start: int, stop: int) -> Tuple[IndexType, ...]:
        return self.compute_offset(slice(start, stop, 1))

    def compute_index_offset(self, index: int) -> Tuple[IndexType, ...]:
        return self.compute_offset(index)

    def schema(self) -> Schema:
        return self._schema

    @abc.abstractmethod
    def write_static_array(
        self,
        key: str,
        entry: _ArrayEntryBase,
        data: ArrayLike,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def read_static_array(
        self,
        key: str,
        entry: _ArrayEntryBase,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> ArrayLike:
        ...

    @abc.abstractmethod
    def read_ragged_array(
        self,
        key: str,
        entry: _RaggedArraySetEntry,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> Any:
        ...

    @abc.abstractmethod
    def write_ragged_array(
        self,
        key: str,
        entry: _RaggedArraySetEntry,
        data: Any,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
        ...


def _lookup_entry(schema: Schema, key: str) -> Optional[_ArrayEntryBase]:
    parts = key.split(".")
    curr: Optional[Schema] = schema
    for part in parts[:-1]:
        if curr is not None and part in curr.fields and curr.fields[part].children is not None:
            curr = curr.fields[part].children
        else:
            return None
    if curr is None:
        return None
    leaf = curr.fields.get(parts[-1])
    return leaf if isinstance(leaf, _ArrayEntryBase) else None


def _get_nested_col(batch: Union[pa.RecordBatch, pa.StructArray], key: str) -> pa.Array:
    """Navigate dotted path in a nested Arrow RecordBatch or StructArray."""
    parts = key.split(".")
    curr: Any = batch
    for part in parts:
        if isinstance(curr, pa.RecordBatch):
            curr = curr.column(part)
        elif isinstance(curr, pa.StructArray):
            curr = curr.field(part)
        else:
            raise KeyError(f"Cannot resolve key segment '{part}' in '{type(curr).__name__}'.")
    return curr


class ArrayStorage(_BaseStorage):
    """Unified Arrow-backed storage engine for Struct and Set containers."""

    @inject
    def __init__(
        self,
        schema: Schema,
        capacity: Optional[int] = None,
        record_batch: Optional[pa.RecordBatch] = None,
        storage_provider: Optional[ArrayStorageProvider] = Provide[Container.storage_provider],
    ) -> None:
        super().__init__(schema=schema, parent=None, path=(), offset=None)
        self._schema = schema
        self._storage_provider = storage_provider

        if capacity is not None:
            self._capacity = capacity
        elif record_batch is not None:
            self._capacity = len(record_batch)
        else:
            self._capacity = self._infer_capacity(schema)

        self._frozen_batch: Optional[pa.RecordBatch] = record_batch
        self._static_staging: Dict[str, np.ndarray] = {}
        self._ragged_staging: Dict[str, List[Optional[np.ndarray]]] = {}

    def _infer_capacity(self, schema: Schema) -> Optional[int]:
        for _, entry in schema.tree_iter():
            if hasattr(entry, "capacity") and entry.capacity is not None:
                return entry.capacity
        return None

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

    def write_static_array(
        self,
        key: str,
        entry: _ArrayEntryBase,
        data: ArrayLike,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        self._check_not_frozen()

        schema_entry = _lookup_entry(self._schema, key)
        if schema_entry is not None and isinstance(schema_entry, _RaggedArraySetEntry):
            return self.write_ragged_array(key, schema_entry, data, offset=offset)

        if isinstance(entry, _RaggedArraySetEntry):
            raise NotImplementedError(
                f"Cannot write ragged array field '{key}' via static array storage. "
                "Use write_ragged_array instead."
            )

        if not isinstance(entry, (_ArrayEntry, _ArraySetEntry)):
            raise TypeError(
                f"Expected static array entry for key '{key}', got {type(entry).__name__}."
            )

        arr = np.asanyarray(data)
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )

        arr = arr.astype(entry.dtype, copy=False)
        entry_shape = tuple(dim for dim in entry.shape if dim is not None)

        if offset:
            # Ensure buffer exists in staging
            if key not in self._static_staging:
                _ = self.read_static_array(key, entry)
            self._static_staging[key][offset] = arr
            return

        # Full column / root write validation
        if isinstance(entry, _ArraySetEntry):
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
        elif isinstance(entry, _ArrayEntry):
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

    def read_static_array(
        self,
        key: str,
        entry: _ArrayEntryBase,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> ArrayLike:
        schema_entry = _lookup_entry(self._schema, key)
        if schema_entry is not None and isinstance(schema_entry, _RaggedArraySetEntry):
            return self.read_ragged_array(key, schema_entry, offset=offset)

        if isinstance(entry, _RaggedArraySetEntry):
            raise NotImplementedError(
                f"Cannot read ragged array field '{key}' via static array storage. "
                "Use read_ragged_array instead."
            )

        if not isinstance(entry, (_ArrayEntry, _ArraySetEntry)):
            raise TypeError(
                f"Expected static array entry for key '{key}', got {type(entry).__name__}."
            )

        if self._frozen_batch is not None:
            col = _get_nested_col(self._frozen_batch, key)
            if hasattr(col, "to_numpy_ndarray"):
                arr = col.to_numpy_ndarray()
            else:
                arr = col.to_numpy()
        else:
            if key not in self._static_staging:
                schema_entry = _lookup_entry(self._schema, key) or entry
                if not schema_entry.is_static and isinstance(schema_entry, _ArrayEntry):
                    raise AttributeError(
                        f"Field '{key}' is dynamic and has not been initialized."
                    )
                entry_shape = tuple(dim for dim in schema_entry.shape if dim is not None)
                if self._capacity is not None and isinstance(schema_entry, _ArraySetEntry):
                    shape = (self._capacity, *entry_shape)
                else:
                    shape = entry_shape
                self._static_staging[key] = np.zeros(shape, dtype=schema_entry.dtype)
            arr = self._static_staging[key]

        if offset:
            return arr[offset]  # type: ignore
        return arr

    def write_ragged_array(
        self,
        key: str,
        entry: _RaggedArraySetEntry,
        data: Any,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        self._check_not_frozen()

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

    def read_ragged_array(
        self,
        key: str,
        entry: _RaggedArraySetEntry,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> Any:
        if offset is not None:
            idx = offset[0]
            if isinstance(idx, int):
                if self._frozen_batch is not None:
                    col = _get_nested_col(self._frozen_batch, key)
                    assert isinstance(col, pa.ListArray)
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

        assert isinstance(col, pa.ListArray)
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
            elif isinstance(entry, _ArrayEntryBase):
                if entry.is_static:
                    arr = self._static_staging.get(full_key)
                    entry_shape = tuple(dim for dim in entry.shape if dim is not None)
                    if arr is None:
                        cap = self._capacity or 1
                        arr = np.zeros((cap, *entry_shape), dtype=entry.dtype)

                    if self._capacity is not None and arr.shape[0] != self._capacity:
                        # Reshape or pad if needed
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
        """Construct an ArrowStorage directly wrapping a frozen Arrow RecordBatch."""
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

    def qualify_key(self, key: str) -> str:
        return ".".join((*self.path, key)) if self.path else key

    def read_static_array(
        self,
        key: str,
        entry: _ArrayEntryBase,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> ArrayLike:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        effective_offset = self.offset if offset is None else (*(self.offset or ()), *offset)
        return self.parent.read_static_array(full_key, entry, offset=effective_offset)

    def write_static_array(
        self,
        key: str,
        entry: _ArrayEntryBase,
        data: ArrayLike,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        effective_offset = self.offset if offset is None else (*(self.offset or ()), *offset)
        self.parent.write_static_array(full_key, entry, data, offset=effective_offset)

    def read_ragged_array(
        self,
        key: str,
        entry: _RaggedArraySetEntry,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> Any:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        effective_offset = self.offset if offset is None else (*(self.offset or ()), *offset)
        return self.parent.read_ragged_array(full_key, entry, offset=effective_offset)

    def write_ragged_array(
        self,
        key: str,
        entry: _RaggedArraySetEntry,
        data: Any,
        offset: Optional[Tuple[Union[slice, int], ...]] = None,
    ) -> None:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        effective_offset = self.offset if offset is None else (*(self.offset or ()), *offset)
        self.parent.write_ragged_array(full_key, entry, data, offset=effective_offset)

    def to_record_batch(self) -> pa.RecordBatch:
        assert self.parent is not None
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
            return batch.slice(start, stop - start)

        return batch

    def __contains__(self, key: str) -> bool:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        return full_key in self.parent


# Backward/forward-compatible aliases
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView