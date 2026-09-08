"""Unified Arrow-backed storage engine for Struct and Set data structures."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple, Union, TypeAlias, cast

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as _pc
pc: Any = _pc

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
)

IndexType: TypeAlias = Union[slice, int]
Offset: TypeAlias = Optional[Tuple[IndexType, ...]]


def _is_unbounded_slice_or_empty(offset: Offset) -> bool:
    """Return True if offset represents a full unindexed or unbounded slice write."""
    if not offset:
        return True
    return all(
        isinstance(idx, slice)
        and idx.start is None
        and idx.stop is None
        and idx.step is None
        for idx in offset
    )


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


def _get_nested_col(
    batch: Union[pa.RecordBatch, pa.StructArray], path: KeyPath
) -> pa.Array:
    """Navigate a KeyPath tuple in a nested Arrow RecordBatch or StructArray."""
    curr: Any = batch
    for segment in path:
        if isinstance(curr, pa.RecordBatch):
            curr = curr.column(segment)
        elif isinstance(curr, pa.StructArray):
            curr = curr.field(segment)
        else:
            raise KeyError(
                f"Cannot resolve path segment '{segment}' in '{type(curr).__name__}'."
            )
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


def _read_ragged(sliced: Any, entry: RaggedArraySetEntry, offset: Offset) -> Any:
    """Resolve a read on a RaggedArraySetEntry from an Awkward array slice."""
    is_element_index = (
        offset is not None
        and len(offset) > 0
        and all(isinstance(idx, int) for idx in offset)
    )
    if is_element_index:
        try:
            return ak.to_numpy(sliced)
        except (ValueError, TypeError, pa.ArrowInvalid):
            return sliced

    if sliced.ndim == 2 and (
        offset is None or (len(offset) == 1 and isinstance(offset[0], slice))
    ):
        try:
            pa_arr = ak.to_arrow(sliced, extensionarray=False)
            if isinstance(pa_arr, pa.ChunkedArray):
                pa_arr = pa_arr.combine_chunks()
            if isinstance(pa_arr, (pa.ListArray, pa.LargeListArray)):
                return RaggedArrayView(pa_arr, entry.dtype)
        except (ValueError, TypeError, pa.ArrowInvalid):
            pass

    return sliced


class _StorageEngine(abc.ABC):
    """Abstract interface for polymorphic storage engines."""

    def _resolve_key(self, key: KeyPath) -> KeyPath:
        """Canonically normalize keys to be schema-relative by stripping the container 'root' prefix."""
        if key and key[0] == "root":
            return key[1:]
        return key

    @abc.abstractmethod
    def read(self, key: KeyPath, entry: Entry, offset: Offset = None) -> Any: ...

    @abc.abstractmethod
    def write(
        self, key: KeyPath, entry: Entry, data: Any, offset: Offset = None
    ) -> None: ...

    @abc.abstractmethod
    def to_record_batch(self) -> pa.RecordBatch: ...

    @abc.abstractmethod
    def __contains__(self, key: KeyPath) -> bool: ...

    @property
    @abc.abstractmethod
    def capacity(self) -> Optional[int]: ...

    @property
    @abc.abstractmethod
    def is_frozen(self) -> bool: ...


class _StagingEngine(_StorageEngine):
    """Unified Awkward-backed mutable staging engine for Struct and Set data structures."""

    def __init__(self, schema: Schema, capacity: Optional[int] = None) -> None:
        self._schema = schema
        self._capacity = capacity
        # Single unified mapping: KeyPath -> staged buffer (np.ndarray, nested list, or ak.Array)
        self._staging: Dict[KeyPath, Any] = {}

    @property
    def capacity(self) -> Optional[int]:
        if self._capacity is not None:
            return self._capacity

        return None

    @property
    def is_frozen(self) -> bool:
        return False

    def __contains__(self, key: KeyPath) -> bool:
        resolved = self._resolve_key(key)
        return resolved in self._staging

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


    def _get_target_capacity(self, key: KeyPath, entry: Entry) -> Optional[int]:
        """Resolve the effective container capacity along the indexing dimension for key."""
        if len(key) > 1:
            return self.capacity

        return entry.capacity or self.capacity

    def _get_required_length(self, offset: Offset) -> int:
        """Extract the required 1-based index/stop bound along the primary indexing axis."""
        match offset:
            case (int(idx), *_):
                return idx + 1
            case (slice(stop=int(stop)), *_):
                return stop
            case _:
                return 1

    def _ensure_static_buffer(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        data: np.ndarray,
        offset: Offset,
    ) -> None:
        """Ensure static buffer in self._staging is allocated and sized for the write target."""
        if _is_unbounded_slice_or_empty(offset):
            # Full column or unindexed struct write
            if self.capacity is not None and len(data) > self.capacity:
                raise ValueError(
                    f"Provided data of length {len(data)} exceeds capacity {self.capacity} for field '{key}'."
                )

            if len(key) > 1:
                target_nested_len = (
                    data.shape[1] if self.capacity is not None else len(data)
                )
                min_dims = 2 if self.capacity is not None else 1
                if (
                    entry.capacity is not None
                    and data.ndim >= min_dims
                    and target_nested_len > entry.capacity
                ):
                    raise ValueError(
                        f"Provided nested dimension {target_nested_len} exceeds capacity {entry.capacity} for field '{key}'."
                    )
            elif entry.capacity is not None and len(data) > entry.capacity:
                raise ValueError(
                    f"Provided data of length {len(data)} exceeds capacity {entry.capacity} for field '{key}'."
                )

            if key not in self._staging or self._staging[key].shape != data.shape:
                self._staging[key] = np.empty_like(data)

            return

        shape = tuple([e for e in entry.shape if e is not None])
        assert len(shape) == len(entry.shape)

        idx_req = self._get_required_length(offset)
        cap = self._get_target_capacity(key, entry)

        if cap is not None and idx_req > cap:
            raise ValueError(
                f"Write target index/stop {idx_req} exceeds capacity {cap} for field '{key}'."
            )

        if key in self._staging:
            if idx_req > len(self._staging[key]):
                raise ValueError(
                    f"Write target index/stop {idx_req} exceeds buffer length {len(self._staging[key])} for field '{key}'."
                )
            return

        match (cap, entry):
            case (None, ArraySetEntry()):
                raise IndexError(
                    "Cannot perform indexed write on a Set with dynamic capacity without a prior bound or slice."
                )
            case (int(limit), ArraySetEntry()):
                self._staging[key] = np.zeros((limit, *shape), dtype=entry.dtype)
            case (_, ArrayEntry()):
                self._staging[key] = np.zeros(shape, dtype=entry.dtype)

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

        target_offset = offset if offset is not None else ()
        self._staging[key][target_offset] = arr

    def _ensure_ragged_slots(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        prepared: Any,
        offset: Offset,
    ) -> None:
        """Ensure ragged slot list in self._staging exists and is sized for write."""
        cap = self._get_target_capacity(key, entry)

        if _is_unbounded_slice_or_empty(offset):
            needed = len(prepared) if isinstance(prepared, (list, tuple)) else 0
            if cap is not None and needed > cap:
                raise ValueError(
                    f"Ragged data length {needed} exceeds capacity {cap} for field '{key}'."
                )
            self._staging[key] = [None] * needed
            return

        idx_req = self._get_required_length(offset)

        if cap is not None and idx_req > cap:
            raise ValueError(
                f"Ragged write target index/stop {idx_req} exceeds capacity {cap} for field '{key}'."
            )

        if key in self._staging:
            if idx_req > len(self._staging[key]):
                raise ValueError(
                    f"Ragged write target index/stop {idx_req} exceeds slot length {len(self._staging[key])} for field '{key}'."
                )
            return

        match cap:
            case None:
                raise IndexError(
                    "Cannot perform indexed write on a ragged Set with dynamic capacity without a prior bound or slice."
                )
            case int(limit):
                self._staging[key] = [None] * limit

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

        self._assign_ragged(self._staging[key], offset or (), prepared)

    def write(
        self,
        key: KeyPath,
        entry: Entry,
        data: Any,
        offset: Offset = None,
    ) -> None:
        key = self._resolve_key(key)
        schema_entry = _lookup_entry(self._schema, key) or entry

        if not isinstance(schema_entry, ArrayEntryBase):
            raise TypeError(
                f"Unsupported schema entry '{type(entry).__name__}' for write to key '{key}'."
            )

        self._validate_dtype(data, schema_entry.dtype, key)

        match schema_entry:
            case ArraySetEntry() | ArrayEntry():
                self._write_static(key, schema_entry, data, offset)
            case RaggedArraySetEntry():
                self._write_ragged(key, schema_entry, data, offset)
            case _:
                raise TypeError(
                    f"Unsupported schema entry type '{type(schema_entry).__name__}' for write."
                )

    def _assign_ragged(
        self,
        slots: list,
        offset: Tuple[IndexType, ...],
        data: Any,
    ) -> None:
        """Recursively assign prepared ragged data into target list slots."""
        match offset:
            case (head,):
                slots[head] = data
            case (int(idx), *tail):
                if slots[idx] is None:
                    slots[idx] = []

                self._assign_ragged(slots[idx], tuple(tail), data)
            case (slice() as sl, *tail):
                indices = range(*sl.indices(len(slots)))
                for idx, item in zip(indices, data):
                    if slots[idx] is None:
                        slots[idx] = []
                    self._assign_ragged(slots[idx], tuple(tail), item)
            case _:
                slots[:] = list(data)

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        key = self._resolve_key(key)
        if key not in self._staging:
            raise AttributeError(f"Field '{key}' has not been initialized.")

        schema_entry = _lookup_entry(self._schema, key)
        assert schema_entry is not None

        raw = self._staging[key]

        match schema_entry:
            case ArrayEntry() | ArraySetEntry():
                if not offset:
                    return raw

                return raw[offset]
            case RaggedArraySetEntry():
                ragged_array = ak.Array(raw)
                sliced = ragged_array[offset] if offset else ragged_array

                return _read_ragged(sliced, schema_entry, offset)
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

    def _get_staged_field_lengths(self, schema: Schema, prefix: KeyPath) -> List[int]:
        """Return lengths of all staged buffers under immediate fields of schema."""
        length = []
        for name in schema.fields:
            val = self._staging.get((*prefix, name))
            if isinstance(val, (np.ndarray, list, tuple)):
                length.append(len(val))

        return length

    def _resolve_dynamic_child_len(self, schema: Schema, prefix: KeyPath) -> int:
        """Resolve total child count across dynamic nested set instances from staged buffers."""
        for child_path, _ in schema.tree_iter(root=prefix):
            if child_path in self._staging:
                val = self._staging[child_path]
                if (
                    isinstance(val, (list, tuple))
                    and len(val) > 0
                    and (isinstance(val[0], (list, tuple, np.ndarray)) or hasattr(val[0], "__len__"))
                ):
                    return sum(len(m) if m is not None else 0 for m in val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    return len(val)
                
        return 0

    def _resolve_level_len(
        self,
        schema: Schema,
        prefix: KeyPath,
        parent_is_set: bool,
        expected_len: Optional[int],
    ) -> int:
        """Determine row count: uses schema/parent expected length first, falls back to staging for dynamic sets."""
        if expected_len is not None and prefix != ():
            return expected_len

        if parent_is_set and prefix != ():
            return self._resolve_dynamic_child_len(schema, prefix)

        staged_lens = self._get_staged_field_lengths(schema, prefix)
        all_fields_staged = len(staged_lens) == len(schema.fields)
        has_uniform_lens = staged_lens and all(l == staged_lens[0] for l in staged_lens)

        if self.capacity is not None:
            if all_fields_staged and has_uniform_lens and staged_lens[0] <= self.capacity:
                return staged_lens[0]
            return self.capacity

        if has_uniform_lens:
            return staged_lens[0]

        raise ValueError(
            "Cannot compile dynamic Set to Arrow RecordBatch: capacity is undefined in schema "
            "and no consistent column lengths were found."
        )

    def _build_dynamic_list_array(
        self,
        entry: SchemaSetEntry,
        path: KeyPath,
        level_len: int,
        sub_struct: pa.StructArray,
    ) -> pa.ListArray:
        """Construct a variable-length ListArray for a dynamic nested Set."""
        assert entry.children is not None
        child_lengths: Optional[List[int]] = None

        for child_name in entry.children.fields:
            child_path = (*path, child_name)

            if child_path in self._staging:
                child_val = self._staging[child_path]

                if (
                    isinstance(child_val, (list, tuple))
                    and len(child_val) > 0
                    and (isinstance(child_val[0], (list, tuple, np.ndarray)) or hasattr(child_val[0], "__len__"))
                ):
                    child_lengths = [len(m) if m is not None else 0 for m in child_val]
                    break

        if child_lengths is not None:
            offsets = [0]
            for child_len in child_lengths:
                offsets.append(offsets[-1] + child_len)

            while len(offsets) < level_len + 1:
                offsets.append(offsets[-1])
        else:
            item_count = len(sub_struct)
            parent_count = level_len if level_len > 0 else 1
            items_per_parent = item_count // parent_count if parent_count > 0 else 0
            offsets = [i * items_per_parent for i in range(parent_count + 1)]

        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), sub_struct)

    def _build_nested_column(
        self,
        entry: Union[SchemaSetEntry, SchemaEntry],
        path: KeyPath,
        level_len: int,
    ) -> pa.Array:
        """Build a StructArray or ListArray for nested Struct or Set schemas."""
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
                sub_cols, sub_names = self._build_columns(
                    entry.children,
                    prefix=path,
                    parent_is_set=True,
                    expected_len=None,
                )
                sub_struct = pa.StructArray.from_arrays(sub_cols, names=sub_names)
                return self._build_dynamic_list_array(entry, path, level_len, sub_struct)

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
        raw = self._staging.get(path)
        shape = cast(Tuple[int, ...], entry.shape)
        if raw is None:
            np_data = np.zeros((level_len, *shape), dtype=entry.dtype)
            mask = [True] * level_len
        else:
            np_data = ak.to_numpy(ak.Array(raw))
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

    def _build_ragged_column(
        self,
        entry: RaggedArraySetEntry,
        path: KeyPath,
        level_len: int,
        parent_is_set: bool,
        prefix: KeyPath = (),
    ) -> Union[pa.ListArray, pa.LargeListArray]:
        """Compile variable-length 1D or nD ragged array into an Arrow ListArray with null bitmask."""
        raw = self._staging.get(path)
        if raw is None:
            return build_ragged_array([None] * level_len, entry.dtype)

        if (
            parent_is_set
            and prefix != ()
            and isinstance(raw, (list, tuple))
            and len(raw) > 0
            and (isinstance(raw[0], (list, tuple, np.ndarray)) or hasattr(raw[0], "__len__"))
        ):
            raw_to_build = [item for sub in raw if sub is not None for item in sub]
        else:
            raw_to_build = list(raw)

        if len(raw_to_build) < level_len:
            raw_to_build = raw_to_build + [None] * (level_len - len(raw_to_build))

        ak_candidate = ak.Array(raw_to_build)
        if ak_candidate.ndim > 2:
            pa_arr = ak.to_arrow(ak_candidate, extensionarray=False)
            if isinstance(pa_arr, pa.ChunkedArray):
                pa_arr = pa_arr.combine_chunks()
            return pa_arr
        else:
            return build_ragged_array(raw_to_build, entry.dtype)

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
                    col = self._build_static_column(entry, path, level_len, parent_is_set)
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



def _extract_field_from_arrow(col: pa.Array, field_name: str) -> pa.Array:
    """Recursively traverse StructArray or ListArray layers to extract a named child field."""
    mask = col.is_null() if col.null_count > 0 else None
    match col:
        case pa.StructArray():
            return col.field(field_name)
        case pa.ListArray() | pa.LargeListArray():
            inner = _extract_field_from_arrow(col.values, field_name)
            return type(col).from_arrays(col.offsets, inner, mask=mask)
        case pa.FixedSizeListArray():
            inner = _extract_field_from_arrow(col.values, field_name)
            base = pa.FixedSizeListArray.from_arrays(inner, col.type.list_size, mask=mask)
            return base.slice(col.offset, len(col))
        case _:
            raise TypeError(f"Cannot extract field '{field_name}' from {type(col).__name__}.")


def _get_nested_arrow_field(col: pa.Array, path: KeyPath) -> pa.Array:
    """Traverse nested StructArray / ListArray layers along a KeyPath to resolve the leaf field array."""
    for seg in path:
        col = _extract_field_from_arrow(col, seg)
    return col


def _unwrap_extension_for_compute(arr: pa.Array) -> Tuple[pa.Array, Optional[pa.DataType]]:
    """Unwrap leaf FixedShapeTensorArray to storage array for PyArrow compute kernels."""
    match arr:
        case pa.ListArray() | pa.LargeListArray() | pa.FixedSizeListArray():
            inner, ext_type = _unwrap_extension_for_compute(arr.values)
            if ext_type is not None:
                mask = arr.is_null() if arr.null_count > 0 else None
                if isinstance(arr, pa.FixedSizeListArray):
                    base = pa.FixedSizeListArray.from_arrays(inner, arr.type.list_size, mask=mask)
                    return base.slice(arr.offset, len(arr)), ext_type
                else:
                    return type(arr).from_arrays(arr.offsets, inner, mask=mask), ext_type
            return arr, None
        case pa.ExtensionArray():
            return arr.storage, arr.type
        case _:
            return arr, None


def _rewrap_extension_after_compute(arr: pa.Array, ext_type: Optional[pa.DataType]) -> pa.Array:
    """Re-wrap storage array back into FixedShapeTensorArray after compute operations."""
    if ext_type is None:
        return arr
    if isinstance(arr, pa.FixedSizeListArray) and arr.type.list_size == ext_type.storage_type.list_size:
        return pa.ExtensionArray.from_storage(ext_type, arr)
    if isinstance(arr, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        mask = arr.is_null() if arr.null_count > 0 else None
        wrapped_values = _rewrap_extension_after_compute(arr.values, ext_type)
        if isinstance(arr, pa.FixedSizeListArray):
            base = pa.FixedSizeListArray.from_arrays(wrapped_values, arr.type.list_size, mask=mask)
            return base.slice(arr.offset, len(arr))
        else:
            return type(arr).from_arrays(arr.offsets, wrapped_values, mask=mask)
    return arr


def _slice_arrow_array(arr: pa.Array, offset: Offset) -> pa.Array:
    """Apply multi-dimensional offset tuple using PyArrow slicing and pc.list_slice."""
    if not offset:
        return arr

    first, *inner_offsets = offset

    # 1. Dimension 0 (Batch / Row level)
    match first:
        case int(idx):
            norm_idx = idx + len(arr) if idx < 0 else idx
            if norm_idx < 0 or norm_idx >= len(arr):
                raise IndexError(f"Index {idx} out of range for array of length {len(arr)}.")
            
            arr = arr.slice(norm_idx, 1)
        case slice() as sl:
            start, stop, _ = sl.indices(len(arr))
            arr = arr.slice(start, max(0, stop - start))

    # 2. Dimensions 1+ (Nested List / Set levels)
    if inner_offsets:
        unwrapped, ext_type = _unwrap_extension_for_compute(arr)
        for inner in inner_offsets:
            match inner:
                case int(idx):
                    unwrapped = pc.list_element(unwrapped, idx)
                case slice() as sl:
                    start = sl.start or 0
                    stop = sl.stop
                    unwrapped = pc.list_slice(unwrapped, start=start, stop=stop)
        arr = _rewrap_extension_after_compute(unwrapped, ext_type)

    return arr


def _arrow_to_numpy(col: pa.Array, shape: Tuple[Optional[int], ...]) -> np.ndarray:
    """Convert an Arrow array (ExtensionArray, ListArray, or primitive) to an N-D NumPy array."""
    if isinstance(col, pa.ExtensionArray):
        return col.to_numpy_ndarray()

    if isinstance(col, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        outer_dims = [len(col)]
        curr = col

        while isinstance(curr, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
            total_items = len(curr.values)
            parent_len = outer_dims[-1] if outer_dims[-1] > 0 else 1
            outer_dims.append(total_items // parent_len)
            curr = curr.values

        if isinstance(curr, pa.ExtensionArray):
            leaf_np = curr.to_numpy_ndarray()
        else:
            leaf_np = curr.to_numpy(zero_copy_only=False)

        int_shape = tuple(d for d in shape if d is not None)
        full_shape = (*outer_dims, *int_shape)
        return leaf_np.reshape(full_shape)

    return col.to_numpy(zero_copy_only=False)


class _ArrowEngine(_StorageEngine):
    """Immutable zero-copy storage engine backed 100% by Apache Arrow."""

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
        resolved = self._resolve_key(key)
        root_field, *sub_path = resolved

        col = self._batch.column(root_field)
        if sub_path:
            col = _get_nested_arrow_field(col, tuple(sub_path))

        sliced_col = _slice_arrow_array(col, offset)

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
            (offset and isinstance(offset[0], int))
            or (isinstance(entry, ArrayEntry) and not isinstance(entry, ArraySetEntry) and len(self._batch) == 1)
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

    def to_record_batch(self) -> pa.RecordBatch:
        assert self.parent is not None
        parent_batch = self.parent.to_record_batch()
        raise NotImplementedError


# Aliases for backward compatibility
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView
