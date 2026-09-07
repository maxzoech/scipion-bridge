"""Unified Arrow-backed storage engine for Struct and Set data structures."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple, Union, TypeAlias, cast

import awkward as ak
import numpy as np
import pyarrow as pa

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
        except (ValueError, TypeError, pa.lib.ArrowInvalid):
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
        except (ValueError, TypeError, pa.lib.ArrowInvalid):
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
        for val in self._staging.values():
            try:
                return len(val)
            except TypeError:
                continue
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
        if isinstance(data, RaggedArrayView):
            data = list(data)
        if isinstance(data, (list, tuple)):
            return [self._prepare_ragged_data(item, dtype) for item in data]
        return np.ascontiguousarray(data, dtype=dtype)

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

        is_sub_offset = (
            offset is not None
            and offset != ()
            and any(
                isinstance(idx, int)
                or (
                    isinstance(idx, slice)
                    and (
                        idx.start is not None
                        or idx.stop is not None
                        or idx.step is not None
                    )
                )
                for idx in offset
            )
        )

        if not is_sub_offset:
            self._write_full(key, schema_entry, data)
        else:
            assert offset is not None
            self._write_sub_offset(key, schema_entry, data, offset)

    def _write_full(
        self,
        key: KeyPath,
        entry: ArrayEntryBase,
        data: Any,
    ) -> None:
        if isinstance(entry, (ArrayEntry, ArraySetEntry)):
            arr = np.asanyarray(data)
            arr = arr.astype(entry.dtype, copy=False)
            self._validate_shape(key, entry, arr)
            self._staging[key] = arr
        elif isinstance(entry, RaggedArraySetEntry):
            prepared = self._prepare_ragged_data(data, entry.dtype)
            self._staging[key] = prepared

    def _write_sub_offset(
        self,
        key: KeyPath,
        entry: ArrayEntryBase,
        data: Any,
        offset: Tuple[IndexType, ...],
    ) -> None:
        if isinstance(entry, (ArrayEntry, ArraySetEntry)):
            self._write_sub_offset_static(key, entry, data, offset)
        elif isinstance(entry, RaggedArraySetEntry):
            self._write_sub_offset_ragged(key, entry, data, offset)

    def _write_sub_offset_static(
        self,
        key: KeyPath,
        entry: Union[ArrayEntry, ArraySetEntry],
        data: Any,
        offset: Tuple[IndexType, ...],
    ) -> None:
        arr = np.asanyarray(data).astype(entry.dtype, copy=False)
        int_dims = tuple(d for d in entry.shape if d is not None)

        idx_req = 1
        if offset and isinstance(offset[0], int):
            idx_req = offset[0] + 1
        elif offset and isinstance(offset[0], slice) and offset[0].stop is not None:
            idx_req = offset[0].stop

        if key not in self._staging:
            if isinstance(entry, ArraySetEntry):
                cap = entry.capacity or self.capacity or idx_req
                cap = max(cap, idx_req)
                self._staging[key] = np.zeros((cap, *int_dims), dtype=entry.dtype)
            else:
                self._staging[key] = np.zeros(int_dims, dtype=entry.dtype)
        else:
            if isinstance(entry, ArraySetEntry):
                current_buf = self._staging[key]
                if idx_req > len(current_buf):
                    new_cap = max(idx_req, len(current_buf) * 2)
                    new_buf = np.zeros((new_cap, *int_dims), dtype=entry.dtype)
                    new_buf[: len(current_buf)] = current_buf
                    self._staging[key] = new_buf

        self._staging[key][offset] = arr

    def _write_sub_offset_ragged(
        self,
        key: KeyPath,
        entry: RaggedArraySetEntry,
        data: Any,
        offset: Tuple[IndexType, ...],
    ) -> None:
        cap = entry.capacity or self.capacity or 0
        if offset and isinstance(offset[0], int):
            cap = max(cap, offset[0] + 1)
        elif offset and isinstance(offset[0], slice) and offset[0].stop is not None:
            cap = max(cap, offset[0].stop)
        elif not cap and isinstance(data, (list, tuple, RaggedArrayView)):
            cap = max(cap, len(data))

        if key not in self._staging:
            self._staging[key] = [None] * cap

        prepared = self._prepare_ragged_data(data, entry.dtype)
        self._assign_ragged(self._staging[key], offset, prepared, entry.dtype)

    def _assign_ragged(
        self,
        slots: list,
        offset: Tuple[IndexType, ...],
        data: Any,
        dtype: np.dtype,
    ) -> None:
        match offset:
            case (int(idx),):
                if len(slots) <= idx:
                    slots.extend([None] * (idx + 1 - len(slots)))
                slots[idx] = data

            case (int(idx), *tail):
                if len(slots) <= idx:
                    slots.extend([None] * (idx + 1 - len(slots)))
                if slots[idx] is None:
                    slots[idx] = []
                self._assign_ragged(slots[idx], tuple(tail), data, dtype)

            case (slice() as sl,):
                needed = len(data) if hasattr(data, "__len__") else 0
                stop = sl.stop if sl.stop is not None else max(len(slots), needed)
                if len(slots) < stop:
                    slots.extend([None] * (stop - len(slots)))
                indices = range(*sl.indices(len(slots)))
                for i, item in zip(indices, data):
                    slots[i] = item

            case (slice() as sl, *tail):
                needed = len(data) if hasattr(data, "__len__") else 0
                stop = sl.stop if sl.stop is not None else max(len(slots), needed)
                if len(slots) < stop:
                    slots.extend([None] * (stop - len(slots)))
                indices = range(*sl.indices(len(slots)))
                for i, item in zip(indices, data):
                    if slots[i] is None:
                        slots[i] = []
                    self._assign_ragged(slots[i], tuple(tail), item, dtype)

            case None | ():
                if isinstance(data, (list, tuple)):
                    for i, item in enumerate(data):
                        self._assign_ragged(slots, (i,), item, dtype)
                else:
                    slots.append(data)

    def read(
        self,
        key: KeyPath,
        entry: Entry,
        offset: Offset = None,
    ) -> Any:
        key = self._resolve_key(key)
        if key not in self._staging:
            schema_entry = _lookup_entry(self._schema, key)
            if (
                schema_entry is not None
                and not schema_entry.is_static
                and isinstance(schema_entry, ArrayEntry)
            ):
                raise AttributeError(
                    f"Field '{key}' is dynamic and has not been initialized."
                )
            raise AttributeError(f"Field '{key}' has not been initialized.")

        schema_entry = _lookup_entry(self._schema, key) or entry
        raw = self._staging[key]

        if isinstance(schema_entry, (ArrayEntry, ArraySetEntry)):
            if offset is None or offset == ():
                return raw
            return raw[offset]

        elif isinstance(schema_entry, RaggedArraySetEntry):
            ak_arr = ak.Array(raw)
            sliced = ak_arr[offset] if (offset is not None and offset != ()) else ak_arr
            return _read_ragged(sliced, schema_entry, offset)
        else:
            raise TypeError(
                f"Unsupported schema entry '{type(entry).__name__}' for read of key '{key}'."
            )

    def to_record_batch(self) -> pa.RecordBatch:
        columns, names = self._build_columns(self._schema, prefix=())
        return pa.RecordBatch.from_arrays(columns, names)

    def _infer_len_from_val(
        self,
        val: Any,
        entry: Entry,
        parent_is_set: bool,
    ) -> Optional[int]:
        """Derive batch length from a staged array or sequence."""
        if isinstance(val, np.ndarray):
            int_dims = tuple(d for d in getattr(entry, "shape", ()) if d is not None)
            if len(val.shape) > len(int_dims):
                return int(np.prod(val.shape[: len(val.shape) - len(int_dims)]))
            return len(val)
        elif isinstance(val, (list, tuple)):
            if parent_is_set and len(val) > 0 and isinstance(val[0], (list, tuple)):
                return sum(len(m) for m in val)
            return len(val)
        return None

    def _infer_level_length(
        self,
        schema: Schema,
        prefix: KeyPath,
        parent_is_set: bool,
        expected_len: Optional[int],
    ) -> int:
        """Infer active batch length from explicit expected length, direct fields, or leaf fields."""
        if expected_len is not None:
            return expected_len

        # 1. Check direct fields at this schema level
        for name, entry in schema.fields.items():
            p = (*prefix, name)
            if p in self._staging:
                length = self._infer_len_from_val(self._staging[p], entry, parent_is_set)
                if length is not None and length > 0:
                    return length

        # 2. Fall back to leaf entries in nested subschemas
        for leaf_path, leaf_entry in schema.tree_iter(root=prefix):
            if leaf_path in self._staging:
                val = self._staging[leaf_path]
                if isinstance(val, np.ndarray):
                    return val.shape[0] if len(val.shape) > 0 else 0
                elif isinstance(val, (list, tuple)):
                    return len(val)

        return self.capacity or 0

    def _build_nested_column(
        self,
        entry: Union[SchemaSetEntry, SchemaEntry],
        path: KeyPath,
        level_len: int,
    ) -> pa.Array:
        """Build a StructArray or ListArray for nested Struct or Set schemas."""
        assert entry.children is not None
        is_nested_set = isinstance(entry, SchemaSetEntry)
        if is_nested_set:
            child_expected = (level_len * entry.capacity) if entry.capacity is not None else None
            sub_cols, sub_names = self._build_columns(
                entry.children, prefix=path, parent_is_set=True, expected_len=child_expected
            )
        else:
            sub_cols, sub_names = self._build_columns(
                entry.children, prefix=path, parent_is_set=False, expected_len=level_len
            )

        sub_struct = pa.StructArray.from_arrays(sub_cols, names=sub_names)

        if not is_nested_set:
            return sub_struct

        if entry.capacity is not None:
            return pa.FixedSizeListArray.from_arrays(sub_struct, entry.capacity)

        # Dynamic nested set with variable child lengths
        child_lengths: Optional[List[int]] = None
        for child_name in entry.children.fields:
            child_path = (*path, child_name)
            if child_path in self._staging:
                child_val = self._staging[child_path]
                if (
                    isinstance(child_val, (list, tuple))
                    and len(child_val) > 0
                    and isinstance(child_val[0], (list, tuple))
                ):
                    child_lengths = [len(m) for m in child_val]
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

    def _build_static_column(
        self,
        entry: Union[ArraySetEntry, ArrayEntry],
        path: KeyPath,
        level_len: int,
        parent_is_set: bool,
    ) -> pa.ExtensionArray:
        """Compile static multidimensional tensor into an Arrow FixedShapeTensorArray."""
        raw = self._staging.get(path)
        int_dims = tuple(d for d in entry.shape if d is not None)
        if raw is None:
            np_data = np.zeros((level_len, *int_dims), dtype=entry.dtype)
        else:
            np_data = ak.to_numpy(ak.Array(raw))
            if parent_is_set and np_data.ndim > len(int_dims):
                np_data = np_data.reshape(-1, *int_dims)
        return build_tensor_array(np_data, shape=int_dims, dtype=entry.dtype)

    def _build_ragged_column(
        self,
        entry: RaggedArraySetEntry,
        path: KeyPath,
        level_len: int,
        parent_is_set: bool,
    ) -> Union[pa.ListArray, pa.LargeListArray]:
        """Compile variable-length 1D or nD ragged array into an Arrow ListArray."""
        raw = self._staging.get(path)
        if raw is None:
            return build_ragged_array([None] * level_len, entry.dtype)

        if (
            parent_is_set
            and isinstance(raw, (list, tuple))
            and len(raw) > 0
            and isinstance(raw[0], (list, tuple))
        ):
            raw_to_build = [item for sub in raw for item in sub]
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
        columns: List[pa.Array] = []
        names: List[str] = []
        level_len = self._infer_level_length(schema, prefix, parent_is_set, expected_len)

        for name, entry in schema.fields.items():
            path = (*prefix, name)
            names.append(name)

            match entry:
                case SchemaSetEntry() | SchemaEntry():
                    columns.append(self._build_nested_column(entry, path, level_len))
                case ArraySetEntry() | ArrayEntry():
                    columns.append(self._build_static_column(entry, path, level_len, parent_is_set))
                case RaggedArraySetEntry():
                    columns.append(self._build_ragged_column(entry, path, level_len, parent_is_set))
                case _:
                    raise TypeError(f"Unsupported schema entry type '{type(entry).__name__}'.")

        return columns, names


def _unwrap_extension_array(a: pa.Array) -> pa.Array:
    """Recursively strip Arrow extension types (e.g. FixedShapeTensor) to underlying storage arrays for Awkward."""
    if isinstance(a, pa.ExtensionArray):
        return _unwrap_extension_array(a.storage)
    if isinstance(a, pa.StructArray):
        fields = [_unwrap_extension_array(a.field(i)) for i in range(a.type.num_fields)]
        names = [a.type.field(i).name for i in range(a.type.num_fields)]
        return pa.StructArray.from_arrays(fields, names=names)
    if isinstance(a, pa.FixedSizeListArray):
        unwrapped_values = _unwrap_extension_array(a.values)
        return pa.FixedSizeListArray.from_arrays(unwrapped_values, a.type.list_size)
    if isinstance(a, (pa.ListArray, pa.LargeListArray)):
        unwrapped_values = _unwrap_extension_array(a.values)
        return type(a).from_arrays(a.offsets, unwrapped_values)
    return a


class _ArrowEngine(_StorageEngine):
    """Immutable zero-copy storage engine wrapping an Apache Arrow RecordBatch."""

    def __init__(self, schema: Schema, batch: pa.RecordBatch) -> None:
        self._schema = schema
        self._batch = batch
        unwrapped_columns = [_unwrap_extension_array(c) for c in batch.columns]
        unwrapped_batch = pa.RecordBatch.from_arrays(unwrapped_columns, names=batch.schema.names)
        self._ak_batch = ak.from_arrow(unwrapped_batch)

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
        col = self._ak_batch
        for seg in resolved:
            col = col[seg]

        if offset is not None and offset != ():
            sliced = col[offset]
        else:
            sliced = col

        if isinstance(entry, (ArrayEntry, ArraySetEntry)):
            res = ak.to_numpy(sliced)
            int_dims = tuple(d for d in entry.shape if d is not None)
            if int_dims and res.shape[-len(int_dims):] != int_dims:
                res = res.reshape((*res.shape[:-1], *int_dims))
            return res
        elif isinstance(entry, RaggedArraySetEntry):
            return _read_ragged(sliced, entry, offset)
        return sliced

    def to_record_batch(self) -> pa.RecordBatch:
        return self._batch

    def __contains__(self, key: KeyPath) -> bool:
        resolved = self._resolve_key(key)
        try:
            _get_nested_col(self._batch, resolved)
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

    def to_record_batch(self) -> pa.RecordBatch:
        assert self.parent is not None
        parent_batch = self.parent.to_record_batch()
        raise NotImplementedError


# Aliases for backward compatibility
ArrowStorage = ArrayStorage
ArrowStorageView = ArrayStorageView
