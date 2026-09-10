"""Unified Arrow-backed storage engine for Struct and Set data structures."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as _pc

pc: Any = _pc

from .exceptions import UninitializedFieldError
from .schema import (
    Schema,
    Entry,
    KeyPath,
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
from .offset import Offset, IndexType


def _lookup_entry(schema: Schema, path: KeyPath) -> Optional[ArrayEntryBase]:
    """Look up a leaf ArrayEntryBase in a schema by KeyPath tuple."""
    return schema.lookup_array(path)


class _BaseStorage(abc.ABC):
    """Abstract base storage class managing offset tracking, key qualification, and read/write."""

    def __init__(
        self,
        schema: Schema,
        parent: Optional["_BaseStorage"] = None,
        path: KeyPath = ("root",),
        offset: Union[Offset, Sequence[IndexType]] = (),
    ) -> None:
        self._schema = schema
        self.parent = parent
        self.offset: Offset = Offset(offset)
        self.path: KeyPath = tuple(path)
        self._field_capacities: Dict[str, Optional[int]] = {}

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

    def schema(self) -> Schema:
        return self._schema

    def read(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
    ) -> Any:
        """Read data for the specified schema entry at this storage's offset."""
        full_path = self.qualify_path(key)
        return self.root_storage._engine.read(full_path, entry, offset=self.offset)

    def write(
        self,
        key: Union[str, KeyPath],
        entry: Entry,
        data: Any,
    ) -> None:
        """Write data for the specified schema entry at this storage's offset."""
        full_path = self.qualify_path(key)
        self.root_storage._engine.write(full_path, entry, data, offset=self.offset)

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch:
        """Freeze and compile the storage into an immutable Arrow RecordBatch."""
        ...

    def __contains__(self, key: Union[str, KeyPath]) -> bool:
        full_path = self.qualify_path(key)
        return full_path in self.root_storage._engine


class _StorageEngine(abc.ABC):
    """Abstract interface for polymorphic storage engines."""

    def _resolve_key(self, key: KeyPath) -> KeyPath:
        """Canonically normalize keys to be schema-relative by stripping the container 'root' prefix."""
        if key and key[0] == "root":
            return key[1:]
        return key

    @abc.abstractmethod
    def read(
        self, key: KeyPath, entry: Entry, offset: Offset = Offset.empty()
    ) -> Any: ...

    @abc.abstractmethod
    def write(
        self, key: KeyPath, entry: Entry, data: Any, offset: Offset = Offset.empty()
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
    def capacity(self) -> Optional[int]: ...

    @property
    @abc.abstractmethod
    def is_frozen(self) -> bool: ...


class _StagingEngine(_StorageEngine):
    """Unified mutable staging engine for Struct and Set data structures."""

    def __init__(
        self,
        schema: Schema,
        capacity: Optional[int] = None,
        root_entry: Optional[SchemaEntry] = None,
    ) -> None:
        if root_entry is not None:
            self._root_entry = root_entry
        elif capacity is not None:
            self._root_entry = SchemaSetEntry(schema=schema, capacity=capacity)
        else:
            self._root_entry = SchemaEntry(schema=schema)

        self._schema = self._root_entry.schema
        self._capacity = (
            self._root_entry.capacity
            if isinstance(self._root_entry, SchemaSetEntry)
            else None
        )
        # Separate, typed staging mappings:
        self._static_staging: Dict[KeyPath, np.ndarray] = {}
        self._ragged_staging: Dict[KeyPath, List[Any]] = {}

    @property
    def root_entry(self) -> SchemaEntry:
        return self._root_entry

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

    @property
    def is_frozen(self) -> bool:
        return False

    def __contains__(self, key: KeyPath) -> bool:
        resolved = self._resolve_key(key)
        return resolved in self._static_staging or resolved in self._ragged_staging

    def _validate_dtype(self, data: Any, target_dtype: np.dtype, key: KeyPath) -> None:
        """Verify data dtype compatibility with schema entry."""
        if isinstance(data, (np.ndarray, np.generic)):
            if not np.can_cast(data.dtype, target_dtype, casting="same_kind"):
                raise TypeError(
                    f"Cannot cast data of dtype '{data.dtype}' to field '{key}' dtype '{target_dtype}'."
                )

        elif isinstance(data, (int, float, complex, bool, str)):
            scalar_arr = np.asanyarray(data)
            if not np.can_cast(scalar_arr.dtype, target_dtype, casting="same_kind"):
                raise TypeError(
                    f"Cannot cast data of dtype '{scalar_arr.dtype}' to field '{key}' dtype '{target_dtype}'."
                )

        elif isinstance(data, (list, tuple)):
            if len(data) > 0:
                self._validate_dtype(data[0], target_dtype, key)

    def _validate_shape(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        arr: np.ndarray,
    ) -> None:
        """Validate tensor shape compatibility with static schema entry."""
        if isinstance(entry, ArraySetEntry):
            entry_ndim = len(entry.shape)
            if arr.ndim < entry_ndim:
                raise ValueError(
                    f"Shape mismatch for key '{key}': expected at least {entry_ndim} dimensions, "
                    f"got data shape {arr.shape}."
                )

            for dim_idx, (expected_dim, actual_dim) in enumerate(
                zip(entry.shape, arr.shape[-entry_ndim:])
            ):
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

            for dim_idx, (expected_dim, actual_dim) in enumerate(
                zip(entry.shape, arr.shape)
            ):
                if expected_dim is not None and expected_dim != actual_dim:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': expected dimension {dim_idx} to be {expected_dim}, "
                        f"got {actual_dim}."
                    )

    def _prepare_ragged_data(self, data: Any, dtype: np.dtype) -> Any:
        """Recursively prepare ragged input into contiguous arrays of expected dtype."""
        if data is None:
            return None

        if isinstance(data, RaggedArrayView):
            data = list(data)

        if isinstance(data, (list, tuple)):
            return [self._prepare_ragged_data(item, dtype) for item in data]
        return np.ascontiguousarray(data, dtype=dtype)

    def _get_path_batch_capacities(self, key: KeyPath) -> List[Optional[int]]:
        """Resolve sequence/batch capacities of all enclosing Set containers along key."""
        batch_capacities: List[Optional[int]] = []
        if isinstance(self._root_entry, SchemaSetEntry):
            batch_capacities.append(self._root_entry.capacity)

        curr_schema: Optional[Schema] = self._root_entry.children
        for seg in key[:-1]:
            if curr_schema is None:
                break
            entry = curr_schema.fields.get(seg)
            match entry:
                case SchemaSetEntry():
                    batch_capacities.append(entry.capacity)
                    curr_schema = entry.children
                case SchemaEntry():
                    curr_schema = entry.children
                case _:
                    curr_schema = None

        return batch_capacities

    def _validate_offset_bounds(
        self,
        key: KeyPath,
        offset: Offset,
        batch_capacities: List[Optional[int]],
    ) -> None:
        """Validate offset index and slice bounds against dimension capacities."""
        for dim_idx, cap in enumerate(batch_capacities):
            if dim_idx >= len(offset.dims):
                break
            idx_val = offset.dims[dim_idx]
            if cap is None and isinstance(idx_val, int):
                raise IndexError(
                    "Cannot perform indexed write on a Set with dynamic capacity without a prior bound or slice."
                )
            req_len = offset.required_len(dim_idx)
            if cap is not None and req_len is not None and req_len > cap:
                raise ValueError(
                    f"Write target index/stop {req_len} exceeds capacity {cap} for dimension {dim_idx} of field '{key}'."
                )

    def _ensure_static_buffer(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        data: np.ndarray,
        offset: Offset,
    ) -> None:
        """Ensure static buffer in self._static_staging is allocated and sized for the write target."""
        batch_capacities = self._get_path_batch_capacities(key)

        if offset.is_unbounded:
            # Full column or unindexed struct write
            for dim_idx, cap in enumerate(batch_capacities):
                if cap is not None and data.ndim > dim_idx and data.shape[dim_idx] > cap:
                    raise ValueError(
                        f"Provided dimension {dim_idx} length {data.shape[dim_idx]} exceeds capacity {cap} for field '{key}'."
                    )

            if (
                key not in self._static_staging
                or self._static_staging[key].shape != data.shape
            ):
                self._static_staging[key] = np.empty_like(data)

            return

        shape = tuple([e for e in entry.shape if e is not None])
        assert len(shape) == len(entry.shape)

        self._validate_offset_bounds(key, offset, batch_capacities)

        if key in self._static_staging:
            buf = self._static_staging[key]
            for dim_idx in range(min(len(offset.dims), buf.ndim)):
                req_len = offset.required_len(dim_idx)
                if req_len is not None and req_len > buf.shape[dim_idx]:
                    raise ValueError(
                        f"Write target index/stop {req_len} exceeds buffer length {buf.shape[dim_idx]} for field '{key}'."
                    )
            return

        if any(cap is None for cap in batch_capacities):
            raise IndexError(
                "Cannot perform indexed write on a Set with dynamic capacity without a prior bound or slice."
            )

        full_shape = (*[c for c in batch_capacities if c is not None], *shape)
        self._static_staging[key] = np.zeros(full_shape, dtype=entry.dtype)

    def _write_static(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        data: Any,
        offset: Offset,
    ) -> None:
        """Unified write for static tensors using in-place slice/index assignment."""
        arr = np.asanyarray(data).astype(entry.dtype, copy=False)

        self._validate_shape(key, entry, arr)
        self._ensure_static_buffer(key, entry, arr, offset)

        self._static_staging[key][offset.to_tuple()] = arr

    def _ensure_ragged_slots(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        prepared: Any,
        offset: Offset,
    ) -> None:
        """Ensure ragged slot list in self._ragged_staging exists and is sized for write."""
        batch_capacities = self._get_path_batch_capacities(key)
        cap = batch_capacities[0] if batch_capacities else None

        if offset.is_unbounded:
            needed = len(prepared) if isinstance(prepared, (list, tuple)) else 0
            if cap is not None and needed > cap:
                raise ValueError(
                    f"Ragged data length {needed} exceeds capacity {cap} for field '{key}'."
                )
            self._ragged_staging[key] = [None] * needed
            return

        self._validate_offset_bounds(key, offset, batch_capacities)

        if key in self._ragged_staging:
            buf = self._ragged_staging[key]
            req_0 = offset.required_len(0)
            if req_0 is not None and req_0 > len(buf):
                raise ValueError(
                    f"Ragged write target index/stop {req_0} exceeds slot length {len(buf)} for field '{key}'."
                )
            return

        match cap:
            case None:
                raise IndexError(
                    "Cannot perform indexed write on a ragged Set with dynamic capacity without a prior bound or slice."
                )
            case int(limit):
                self._ragged_staging[key] = [None] * limit

    def _write_ragged(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        data: Any,
        offset: Offset,
    ) -> None:
        """Unified write for ragged entries delegating to _assign_ragged."""
        prepared = self._prepare_ragged_data(data, entry.dtype)
        self._ensure_ragged_slots(key, entry, prepared, offset)
        batch_capacities = self._get_path_batch_capacities(key)
        self._assign_ragged(
            self._ragged_staging[key],
            offset,
            prepared,
            child_capacities=batch_capacities[1:] if len(batch_capacities) > 1 else (),
        )

    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
        offset: Union[Offset, Sequence[IndexType]] = Offset.empty(),
    ) -> None:
        offset_val = Offset(offset)
        key = self._resolve_key(key)
        schema_entry = _lookup_entry(self._schema, key) or entry

        if not isinstance(schema_entry, ArrayEntryBase):
            raise TypeError(
                f"Unsupported schema entry '{type(entry).__name__}' for write to key '{key}'."
            )

        self._validate_dtype(data, schema_entry.dtype, key)

        match schema_entry:
            case ArraySetEntry() | ArrayEntry():
                self._write_static(key, schema_entry, data, offset_val)
            case RaggedArraySetEntry():
                self._write_ragged(key, schema_entry, data, offset_val)
            case _:
                raise TypeError(
                    f"Unsupported schema entry type '{type(schema_entry).__name__}' for write."
                )

    def _assign_ragged(
        self,
        slots: list,
        offset: Offset,
        data: Any,
        child_capacities: Sequence[Optional[int]] = (),
    ) -> None:
        """Recursively assign prepared ragged data into target list slots."""
        match offset.dims:
            case (head,):
                match head:
                    case int(idx):
                        if idx >= len(slots):
                            slots.extend([None] * (idx + 1 - len(slots)))
                        slots[idx] = data
                    case slice() as sl:
                        if not isinstance(data, (list, tuple)):
                            raise TypeError(
                                f"Expected Sequence for ragged slice assignment, got {type(data).__name__}."
                            )
                        if sl.start is None and sl.stop is None and (sl.step is None or sl.step == 1):
                            if len(slots) == 0:
                                slots[:] = list(data)
                            else:
                                if len(data) != len(slots):
                                    raise ValueError(
                                        f"Provided slice data length {len(data)} does not match slice size {len(slots)}."
                                    )
                                slots[:] = list(data)
                        else:
                            if isinstance(sl.stop, int) and sl.stop > len(slots):
                                slots.extend([None] * (sl.stop - len(slots)))
                            indices = list(range(*sl.indices(len(slots))))
                            if len(data) != len(indices):
                                raise ValueError(
                                    f"Provided slice data length {len(data)} does not match slice size {len(indices)}."
                                )
                            for i, d in zip(indices, data):
                                slots[i] = d
                    case _:
                        slots[:] = list(data)

            case (int(idx), *tail):
                if idx >= len(slots):
                    slots.extend([None] * (idx + 1 - len(slots)))

                if slots[idx] is None:
                    next_cap = child_capacities[0] if child_capacities else None
                    slots[idx] = [None] * next_cap if next_cap is not None else []

                self._assign_ragged(
                    slots[idx],
                    Offset(tail),
                    data,
                    child_capacities[1:] if child_capacities else (),
                )

            case (slice() as sl, *tail):
                if not isinstance(data, (list, tuple)):
                    raise TypeError(
                        f"Expected Sequence for ragged slice assignment, got {type(data).__name__}."
                    )
                if isinstance(sl.stop, int) and sl.stop > len(slots):
                    slots.extend([None] * (sl.stop - len(slots)))
                indices = list(range(*sl.indices(len(slots))))
                if len(data) != len(indices):
                    raise ValueError(
                        f"Provided slice data length {len(data)} does not match slice size {len(indices)}."
                    )
                next_cap = child_capacities[0] if child_capacities else None
                next_child_caps = child_capacities[1:] if child_capacities else ()
                for i, item in zip(indices, data):
                    if slots[i] is None:
                        slots[i] = [None] * next_cap if next_cap is not None else []
                    self._assign_ragged(
                        slots[i],
                        Offset(tail),
                        item,
                        next_child_caps,
                    )

            case _:
                slots[:] = list(data)

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Union[Offset, Sequence[IndexType]] = Offset.empty(),
    ) -> Any:
        offset_val = Offset(offset)
        key = self._resolve_key(key)
        schema_entry = _lookup_entry(self._schema, key)
        assert schema_entry is not None

        match schema_entry:
            case ArrayEntry() | ArraySetEntry():
                if key not in self._static_staging:
                    raise UninitializedFieldError(f"Field '{key}' has not been initialized.")
                raw_static = self._static_staging[key]
                if offset_val.is_empty:
                    return raw_static

                return raw_static[offset_val.to_tuple()]
            case RaggedArraySetEntry():
                if key not in self._ragged_staging:
                    raise UninitializedFieldError(f"Field '{key}' has not been initialized.")

                raw = self._ragged_staging[key]
                # Optimization:
                # Avoid converting the raw Python list of NumPy arrays to an Awkward Array via
                # `ak.Array(raw)` unless arbitrary or complex non-integer slicing is requested.
                # Awkward's `fromiter` inspects every nested scalar in C++ (~6 µs per float),
                # which causes severe performance bottlenecks for large image tensors.

                # Fast-path 1: Full / unbounded read (e.g. during internal data copy or field access)
                if offset_val.is_unbounded:
                    if isinstance(raw, list) and len(raw) > 0:
                        first = next((x for x in raw if x is not None), None)
                        if (
                            first is not None
                            and isinstance(first, np.ndarray)
                            and first.ndim == 1
                        ):
                            # 1D ragged array: construct RaggedArrayView directly via contiguous PyArrow buffer
                            return RaggedArrayView(
                                build_ragged_array(raw, schema_entry.dtype),
                                schema_entry.dtype,
                            )
                    return raw

                match offset_val.dims:
                    # Fast-path 2: Single-element integer indexing (e.g. (idx,))
                    case (int(idx),):
                        norm_idx = idx + len(raw) if idx < 0 else idx
                        if norm_idx < 0 or norm_idx >= len(raw):
                            raise IndexError(
                                f"Index {idx} out of range for field '{key}'."
                            )
                        item = raw[norm_idx]
                        if item is None:
                            raise UninitializedFieldError(
                                f"Field '{key}' at index {offset_val.to_tuple()} has not been initialized."
                            )
                        if isinstance(item, np.ndarray):
                            return item
                        if isinstance(item, list):
                            first = next((x for x in item if x is not None), None)
                            if (
                                first is not None
                                and isinstance(first, np.ndarray)
                                and first.ndim == 1
                            ):
                                return RaggedArrayView(
                                    build_ragged_array(item, schema_entry.dtype),
                                    schema_entry.dtype,
                                )
                            return item

                    # Fast-path 3: All-integer multi-index (e.g. (idx1, idx2, ...))
                    case (int(), *tail) if all(isinstance(i, int) for i in tail):
                        curr: Any = raw
                        int_indices = cast(Tuple[int, ...], offset_val.dims)
                        for int_idx in int_indices:
                            if curr is None or not isinstance(curr, list):
                                raise UninitializedFieldError(
                                    f"Field '{key}' at index {offset_val.to_tuple()} has not been initialized."
                                )
                            norm_idx = int_idx + len(curr) if int_idx < 0 else int_idx
                            if norm_idx < 0 or norm_idx >= len(curr):
                                raise IndexError(f"Index {int_idx} out of range.")
                            curr = curr[norm_idx]
                        if curr is None:
                            raise UninitializedFieldError(
                                f"Field '{key}' at index {offset_val.to_tuple()} has not been initialized."
                            )
                        if isinstance(curr, np.ndarray):
                            return curr
                        if isinstance(curr, (list, tuple)):
                            return np.asanyarray(curr, dtype=schema_entry.dtype)
                        return curr

                    # Fast-path 4: Leading int followed by unbounded slice (e.g. (idx, slice(None)))
                    case (int(idx), *tail) if Offset(tail).is_unbounded:
                        norm_idx = idx + len(raw) if idx < 0 else idx
                        if norm_idx < 0 or norm_idx >= len(raw):
                            raise IndexError(
                                f"Index {idx} out of range for field '{key}'."
                            )
                        item = raw[norm_idx]
                        if item is None:
                            raise UninitializedFieldError(
                                f"Field '{key}' at index {offset_val.to_tuple()} has not been initialized."
                            )
                        if isinstance(item, list):
                            first = next((x for x in item if x is not None), None)
                            if (
                                first is not None
                                and isinstance(first, np.ndarray)
                                and first.ndim == 1
                            ):
                                return RaggedArrayView(
                                    build_ragged_array(item, schema_entry.dtype),
                                    schema_entry.dtype,
                                )
                        return item

                # Fallback to Awkward Array for complex arbitrary slices (strided, step != 1, etc.)
                ragged_array = ak.Array(raw)
                sliced = ragged_array[offset_val.to_tuple()]

                return _read_ragged(sliced, schema_entry, offset_val)
            case _:
                raise TypeError(
                    f"Unsupported schema entry '{type(entry).__name__}' for read of key '{key}'."
                )

    def to_record_batch(self) -> pa.RecordBatch:
        columns, names = self._build_columns(
            self._schema,
            prefix=(),
            parent_is_set=True,
            expected_len=self.capacity,
        )
        return pa.RecordBatch.from_arrays(columns, names)

    def _get_staged(self, key: KeyPath) -> Optional[Union[np.ndarray, List[Any]]]:
        """Retrieve buffer from static or ragged staging map."""
        if key in self._static_staging:
            return self._static_staging[key]
        
        return self._ragged_staging.get(key)

    def _get_staged_field_lengths(self, schema: Schema, prefix: KeyPath) -> List[int]:
        """Return lengths of all staged buffers under immediate fields of schema."""
        length = []
        for name in schema.fields:
            val = self._get_staged((*prefix, name))
            if val is not None:
                length.append(len(val))

        return length

    def _resolve_level_len(
        self,
        schema: Schema,
        prefix: KeyPath,
        parent_is_set: bool,
        expected_len: Optional[int],
    ) -> int:
        """Determine row count: uses explicit expected length for children, or infers for root sets."""
        if prefix == ():
            return self._infer_root_dynamic_len(schema)

        if expected_len is not None:
            return expected_len

        raise ValueError(
            f"Internal error: child schema at '{prefix}' did not receive expected_len from parent."
        )

    def _infer_root_dynamic_len(self, schema: Schema) -> int:
        """Infer row count for a top-level Set from staged buffer lengths or capacity."""
        staged_lens = self._get_staged_field_lengths(schema, prefix=())
        all_fields_staged = len(staged_lens) == len(schema.fields)
        has_uniform_lens = bool(
            staged_lens and all(l == staged_lens[0] for l in staged_lens)
        )

        if self.capacity is not None:
            if (
                all_fields_staged
                and has_uniform_lens
                and staged_lens[0] <= self.capacity
            ):
                return staged_lens[0]
            return self.capacity

        if has_uniform_lens:
            return staged_lens[0]

        raise ValueError(
            "Cannot compile dynamic Set to Arrow RecordBatch: capacity is undefined in schema "
            "and no consistent column lengths were found."
        )

    def _calculate_dynamic_set_offsets(
        self,
        schema: Schema,
        path: KeyPath,
        level_len: int,
    ) -> Tuple[List[int], List[int]]:
        """Calculate per-parent child counts and Arrow ListArray offsets for a dynamic nested Set."""
        child_lengths: Optional[List[int]] = None
        first_field: Optional[str] = None

        for child_name in schema.fields:
            child_path = (*path, child_name)
            child_val = self._get_staged(child_path)

            if child_val is not None:
                if (
                    isinstance(child_val, (list, tuple))
                    and len(child_val) > 0
                    and (
                        isinstance(child_val[0], (list, tuple, np.ndarray))
                        or hasattr(child_val[0], "__len__")
                    )
                ):
                    curr_lengths = [len(m) if m is not None else 0 for m in child_val]
                elif isinstance(child_val, (list, tuple, np.ndarray)):
                    curr_lengths = [len(child_val)]
                else:
                    continue

                if child_lengths is None:
                    child_lengths = curr_lengths
                    first_field = child_name
                else:
                    if curr_lengths != child_lengths:
                        raise ValueError(
                            f"Inconsistent child counts under dynamic Set '{path}': "
                            f"field '{first_field}' has counts {child_lengths}, "
                            f"but field '{child_name}' has counts {curr_lengths}."
                        )

        if child_lengths is not None:
            offsets = [0]
            for child_len in child_lengths:
                offsets.append(offsets[-1] + child_len)

            while len(offsets) < level_len + 1:
                offsets.append(offsets[-1])
        else:
            offsets = [0] * (level_len + 1)
            child_lengths = [0] * level_len

        return child_lengths, offsets

    def _build_nested_column(
        self,
        entry: Union[SchemaSetEntry, SchemaEntry],
        path: KeyPath,
        level_len: int,
    ) -> pa.Array:
        """Build a StructArray or ListArray for nested Struct or Set schemas.

        Length Inference Role:
        Arrow RecordBatches and StructArrays require each column at a given level
        to have an identical, explicit row count (`level_len`). For nested structures,
        the expected child length is computed deterministically by the parent:
        1. Nested Struct (`SchemaEntry`):
           Rows match parent rows 1:1 (`child_expected = level_len`).
        2. Nested Fixed-Size Set (`SchemaSetEntry(capacity=cap)`):
           Each of the `level_len` parent rows contains exactly `cap` child elements,
           requiring a flattened child StructArray of `child_expected = level_len * cap`.
           This is wrapped in an Arrow `FixedSizeListArray(sub_struct, cap)`.
        3. Nested Dynamic Set (`SchemaSetEntry(capacity=None)`):
           Parent rows contain variable numbers of child elements. Before descending
           into the child schema, we inspect the staged buffers under `path` to derive
           each parent's child count and offsets. The sum of child counts (`offsets[-1]`)
           is the exact total flattened child elements (`child_expected = offsets[-1]`).
           Passing `child_expected` down ensures all nested fields compile with the exact
           matching length without guessing or backtracking. The child StructArray is
           then wrapped in an Arrow `ListArray.from_arrays(offsets, sub_struct)`.
        4. Top-Level Root Sets (Context):
           Nested children always receive an explicit expected length from their parent (1–3 above).
           Only the root level (`prefix == ()`) delegates to `_infer_root_dynamic_len` to determine
           the batch length from staged buffer dimensions (or explicit root capacity).
        """
        assert entry.children is not None
        match entry:
            case SchemaSetEntry(capacity=int(cap)):
                child_expected = level_len * cap
                sub_cols, sub_names = self._build_columns(
                    entry.children,
                    prefix=path,
                    parent_is_set=True,
                    expected_len=child_expected,
                )
                sub_struct = pa.StructArray.from_arrays(sub_cols, names=sub_names)
                return pa.FixedSizeListArray.from_arrays(sub_struct, cap)

            case SchemaSetEntry(capacity=None):
                child_lengths, offsets = self._calculate_dynamic_set_offsets(
                    entry.children, path, level_len
                )
                child_total = offsets[-1]
                sub_cols, sub_names = self._build_columns(
                    entry.children,
                    prefix=path,
                    parent_is_set=True,
                    expected_len=child_total,
                )
                sub_struct = pa.StructArray.from_arrays(sub_cols, names=sub_names)
                return pa.ListArray.from_arrays(
                    pa.array(offsets, type=pa.int32()), sub_struct
                )

            case SchemaEntry():
                sub_cols, sub_names = self._build_columns(
                    entry.children,
                    prefix=path,
                    parent_is_set=False,
                    expected_len=level_len,
                )
                return pa.StructArray.from_arrays(sub_cols, names=sub_names)

    def _build_static_column(
        self,
        entry: Union[ArraySetEntry, ArrayEntry],
        path: KeyPath,
        level_len: int,
        parent_is_set: bool,
    ) -> pa.ExtensionArray:
        """Compile static multidimensional tensor into an Arrow FixedShapeTensorArray with null bitmask."""
        raw = self._static_staging.get(path)
        shape = cast(Tuple[int, ...], entry.shape)
        if raw is None:
            np_data = np.zeros((level_len, *shape), dtype=entry.dtype)
            mask = [True] * level_len
        else:
            assert isinstance(
                raw, np.ndarray
            ), f"Expected np.ndarray for static field '{path}', got {type(raw).__name__}."
            np_data = np.ascontiguousarray(raw, dtype=entry.dtype)
            if parent_is_set and np_data.ndim > len(shape):
                np_data = np_data.reshape(-1, *shape)
            actual_len = len(np_data)
            if actual_len < level_len:
                padded = np.zeros((level_len, *shape), dtype=entry.dtype)
                padded[:actual_len] = np_data
                np_data = padded
                mask = [False] * actual_len + [True] * (level_len - actual_len)
            else:
                mask = None
        return build_tensor_array(np_data, shape=shape, dtype=entry.dtype, mask=mask)

    @staticmethod
    def _capacity_stride(caps: Sequence[Optional[int]], depth: int) -> int:
        mult = 1
        for c in caps[:depth]:
            if c is None:
                return 0
            mult *= c
        return mult

    @staticmethod
    def _flatten_ragged_hierarchy(
        raw: Any,
        depth: int,
        child_caps: Sequence[Optional[int]],
    ) -> List[Any]:
        """Recursively flatten intermediate container list nesting to reach leaf ragged items.

        If an intermediate container is None, pad with the expected product of child capacities
        so that Arrow FixedSizeListArray strides remain strictly aligned.
        """
        if depth <= 0:
            if isinstance(raw, list):
                return raw
            elif raw is None:
                return []
            return [raw]

        if raw is None:
            mult = _StagingEngine._capacity_stride(child_caps, depth)
            return [None] * mult

        flat: List[Any] = []
        for item in raw:
            if item is None:
                mult = _StagingEngine._capacity_stride(child_caps, depth)
                flat.extend([None] * mult)
            elif depth == 1:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            else:
                flat.extend(
                    _StagingEngine._flatten_ragged_hierarchy(
                        item,
                        depth - 1,
                        child_caps[1:] if len(child_caps) > 1 else (),
                    )
                )

        return flat

    def _build_ragged_column(
        self,
        entry: RaggedArraySetEntry,
        path: KeyPath,
        level_len: int,
        parent_is_set: bool,
        prefix: KeyPath = (),
    ) -> Union[pa.ListArray, pa.LargeListArray]:
        """Compile variable-length 1D or nD ragged array into an Arrow ListArray with null bitmask."""
        raw = self._ragged_staging.get(path)
        if raw is None:
            return build_ragged_array([None] * level_len, entry.dtype)

        batch_capacities = self._get_path_batch_capacities(path)
        if (
            parent_is_set
            and prefix != ()
            and isinstance(raw, (list, tuple))
            and len(batch_capacities) > 1
        ):
            depth = len(batch_capacities) - 1
            child_caps = batch_capacities[1:]
            raw_to_build = self._flatten_ragged_hierarchy(raw, depth, child_caps)
        else:
            raw_to_build = list(raw)

        if len(raw_to_build) < level_len:
            raw_to_build = raw_to_build + [None] * (level_len - len(raw_to_build))

        # Optimization:
        # Avoid running `ak.Array(raw_to_build)` just to check dimensionality.
        # Awkward's `fromiter` inspects every nested float scalar in C++ (~6 µs per float),
        # causing major compilation bottlenecks for multi-dimensional images.
        first_valid = next((x for x in raw_to_build if x is not None), None)
        if first_valid is None or getattr(first_valid, "ndim", 1) <= 1:
            return build_ragged_array(raw_to_build, entry.dtype)
        elif getattr(first_valid, "ndim", 1) == 2:
            return build_multidim_ragged_array(raw_to_build, entry.dtype)
        else:
            # Fallback for 3D+ structures
            ak_candidate = ak.Array(raw_to_build)
            pa_arr = ak.to_arrow(ak_candidate, extensionarray=False)
            if isinstance(pa_arr, pa.ChunkedArray):
                pa_arr = pa_arr.combine_chunks()
            return pa_arr

    def _build_columns(
        self,
        schema: Schema,
        prefix: KeyPath = (),
        parent_is_set: bool = False,
        expected_len: Optional[int] = None,
    ) -> Tuple[List[pa.Array], List[str]]:
        """Recursively compile staged arrays and nested subschemas into Arrow arrays."""
        level_len = self._resolve_level_len(schema, prefix, parent_is_set, expected_len)
        columns: List[pa.Array] = []
        names: List[str] = []

        for name, entry in schema.fields.items():
            path = (*prefix, name)
            names.append(name)

            match entry:
                case SchemaSetEntry() | SchemaEntry():
                    col = self._build_nested_column(entry, path, level_len)
                case ArraySetEntry() | ArrayEntry():
                    col = self._build_static_column(
                        entry, path, level_len, parent_is_set
                    )
                case RaggedArraySetEntry():
                    col = self._build_ragged_column(
                        entry, path, level_len, parent_is_set, prefix
                    )
                case _:
                    raise TypeError(
                        f"Unsupported schema entry type '{type(entry).__name__}'."
                    )
            columns.append(col)

        return columns, names


class _ArrowEngine(_StorageEngine):
    """Immutable zero-copy storage engine backed 100% by Apache Arrow."""

    def __init__(
        self,
        schema: Schema,
        batch: pa.RecordBatch,
        root_entry: Optional[SchemaEntry] = None,
    ) -> None:
        if root_entry is not None:
            self._root_entry = root_entry
        else:
            self._root_entry = SchemaSetEntry(schema=schema, capacity=len(batch))
        self._schema = self._root_entry.schema
        self._batch = batch

    @property
    def root_entry(self) -> SchemaEntry:
        return self._root_entry

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
        offset: Union[Offset, Sequence[IndexType]] = Offset.empty(),
    ) -> None:
        raise RuntimeError("Cannot mutate a frozen Set.")

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Union[Offset, Sequence[IndexType]] = Offset.empty(),
    ) -> Any:
        offset_val = Offset(offset)
        resolved = self._resolve_key(key)
        root_field, *sub_path = resolved

        col = self._batch.column(root_field)
        if sub_path:
            col = _get_nested_arrow_field(col, tuple(sub_path))

        sliced_col = _slice_arrow_array(col, offset_val)

        res: Any
        match entry:
            case ArrayEntryBase() if not entry.is_static:
                res = RaggedArrayView(sliced_col, entry.dtype)

            case ArraySetEntry() | ArrayEntry():
                shape = cast(Tuple[int, ...], entry.shape)
                res = _arrow_to_numpy(sliced_col, shape)

            case _:
                res = sliced_col

        is_single_item = bool(
            (len(offset_val) > 0 and isinstance(offset_val[0], int))
            or (
                isinstance(self._root_entry, SchemaEntry)
                and not isinstance(self._root_entry, SchemaSetEntry)
                and len(self._batch) == 1
            )
        )
        return res[0] if is_single_item else res

    def to_record_batch(self) -> pa.RecordBatch:
        return self._batch

    def __contains__(self, key: KeyPath) -> bool:
        resolved = self._resolve_key(key)
        try:
            root_field, *sub_path = resolved
            col = self._batch.column(root_field)
            if sub_path:
                _get_nested_arrow_field(col, tuple(sub_path))
            return True
        except (KeyError, IndexError, TypeError):
            return False


class ArrayStorage(_BaseStorage):
    """Unified Arrow-backed storage engine delegating to polymorphic _StorageEngine implementations."""

    def __init__(
        self,
        schema: Schema,
        capacity: Optional[int] = None,
        record_batch: Optional[pa.RecordBatch] = None,
        path: KeyPath = ("root",),
        offset: Union[Offset, Sequence[IndexType]] = (),
        root_entry: Optional[SchemaEntry] = None,
    ) -> None:
        super().__init__(schema=schema, parent=None, path=path, offset=Offset(offset))
        self._schema = schema

        if root_entry is not None:
            resolved_root_entry = root_entry
        elif capacity is not None:
            resolved_root_entry = SchemaSetEntry(schema=schema, capacity=capacity)
        else:
            resolved_root_entry = SchemaEntry(schema=schema)

        if record_batch is not None:
            self._engine: _StorageEngine = _ArrowEngine(
                schema, record_batch, root_entry=resolved_root_entry
            )
        else:
            self._engine = _StagingEngine(
                schema, capacity=capacity, root_entry=resolved_root_entry
            )

    @property
    def root_entry(self) -> SchemaEntry:
        return self._engine.root_entry

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
            self._engine = _ArrowEngine(
                self._schema, batch, root_entry=self._engine.root_entry
            )
        return batch

    @classmethod
    def from_record_batch(
        cls,
        batch: pa.RecordBatch,
        schema: Schema,
        root_entry: Optional[SchemaEntry] = None,
    ) -> "ArrayStorage":
        """Construct an ArrayStorage directly wrapping a frozen Arrow RecordBatch."""
        return cls(
            schema=schema,
            capacity=len(batch),
            record_batch=batch,
            root_entry=root_entry,
        )


class ArrayStorageView(_BaseStorage):
    """Sub-view into a parent storage with qualified path prefix and offset propagation."""

    def to_record_batch(self) -> pa.RecordBatch:
        raise NotImplementedError
