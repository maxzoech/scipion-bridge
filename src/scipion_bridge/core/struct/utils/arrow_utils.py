"""Utilities for bridging scipion-bridge schemas/structures with Apache Arrow."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union, overload
import awkward as ak
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.compute as pc

from ..schema import Schema, ArrayEntryBase, RaggedArraySetEntry
from ..key_path import IndexType, KeyPath
from ..exceptions import UninitializedFieldError


class RaggedArrayView(Sequence[Any]):
    """A zero-copy multi-axis sequence view over single- or multi-level Apache Arrow ListArrays."""

    def __init__(
        self,
        list_array: Union[pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray],
        dtype: np.dtype,
    ) -> None:
        self._list_array = list_array
        self.dtype = np.dtype(dtype)

    def __len__(self) -> int:
        return len(self._list_array)

    def __getitem__(
        self, item: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> Any:
        match item:
            case ():
                return self

            case (first, *rest):
                sub = self[first]
                return sub[tuple(rest)] if rest else sub

            case slice():
                return RaggedArrayView(self._list_array[item], self.dtype)

            case int(idx):
                norm_idx = idx + len(self) if idx < 0 else idx
                if norm_idx < 0 or norm_idx >= len(self):
                    raise IndexError(
                        f"Index {idx} out of range for RaggedArrayView of length {len(self)}."
                    )

                scalar = self._list_array[norm_idx]
                if not scalar.is_valid:
                    raise UninitializedFieldError(
                        f"Cannot read unpopulated or null value at index {norm_idx}."
                    )

                match scalar.values:
                    case pa.ListArray() | pa.LargeListArray() | pa.FixedSizeListArray():
                        try:
                            return ak.to_numpy(ak.from_arrow(scalar.values))
                        except (ValueError, TypeError):
                            return RaggedArrayView(scalar.values, self.dtype)
                    case _:
                        return scalar.values.to_numpy(zero_copy_only=False)

            case _:
                raise TypeError(
                    f"Invalid RaggedArrayView index type '{type(item).__name__}'."
                )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_list(self) -> List[Any]:
        return list(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (RaggedArrayView, list, tuple)):
            if len(self) != len(other):
                return False
            return all(
                (
                    np.array_equal(a, b)
                    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                    else a == b
                )
                for a, b in zip(self, other)
            )
        return False

    def to_numpy(self) -> np.ndarray:
        """Convert the ragged array view to a regular NumPy ndarray.

        Raises:
            ValueError: If subarray lengths are irregular or contain null values.
        """
        try:
            arr = ak.to_numpy(ak.from_arrow(self._list_array))
        except ValueError as e:
            raise ValueError(
                f"Cannot resolve {type(self).__name__} to regular NumPy ndarray: "
                "subarray lengths are not uniform."
            ) from e

        if isinstance(arr, np.ma.MaskedArray):
            if np.ma.is_masked(arr):
                raise ValueError(
                    f"Cannot resolve {type(self).__name__} to NumPy ndarray: "
                    "data contains null or uninitialized values."
                )
            return arr.data
        return arr

    def __array__(
        self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None
    ) -> np.ndarray:
        arr = self.to_numpy()
        if dtype is not None and arr.dtype != dtype:
            return arr.astype(dtype, copy=copy if copy is not None else True)
        elif copy:
            return arr.copy()
        else:
            return arr

    def __repr__(self) -> str:
        if len(self) <= 3:
            shapes = [
                tuple(arr.shape) if isinstance(arr, np.ndarray) else f"len={len(arr)}"
                for arr in self
            ]
            return (
                f"RaggedArrayView(len={len(self)}, shapes={shapes}, dtype={self.dtype})"
            )
        shapes = [
            (
                tuple(self[i].shape)
                if isinstance(self[i], np.ndarray)
                else f"len={len(self[i])}"
            )
            for i in range(3)
        ]
        return (
            f"RaggedArrayView(len={len(self)}, shapes={shapes}..., dtype={self.dtype})"
        )


def schema_to_arrow_schema(schema: Schema) -> pa.Schema:
    """Recursively convert a scipion-bridge Schema into a pyarrow.Schema."""
    fields: List[pa.Field] = []

    for name, entry in schema.fields.items():
        if entry.children is not None:
            child_pa_schema = schema_to_arrow_schema(entry.children)
            field_type = pa.struct(
                [child_pa_schema.field(i) for i in range(len(child_pa_schema))]
            )
            fields.append(pa.field(name, field_type))
        elif isinstance(entry, ArrayEntryBase):
            pa_dtype = pa.from_numpy_dtype(entry.dtype)
            if entry.is_static:
                shape = tuple(dim for dim in entry.shape if dim is not None)
                field_type = pa.fixed_shape_tensor(pa_dtype, shape)
            else:
                field_type = pa.list_(pa_dtype)
            fields.append(pa.field(name, field_type))
        else:
            raise TypeError(
                f"Unsupported schema entry type '{type(entry).__name__}' for field '{name}'."
            )

    return pa.schema(fields)


def build_tensor_array(
    data: np.ndarray,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    mask: Optional[Sequence[bool] | NDArray] = None,
) -> pa.ExtensionArray:
    """Compile a contiguous multidimensional NumPy array into an Arrow FixedShapeTensorArray with optional validity bitmask."""
    cell_size = int(np.prod(shape))
    pa_dtype = pa.from_numpy_dtype(dtype)
    flat_data = np.ascontiguousarray(data, dtype=dtype).ravel()
    flat_pa = pa.array(flat_data, type=pa_dtype)
    pa_mask = pa.array(mask, type=pa.bool_()) if mask is not None else None
    storage = pa.FixedSizeListArray.from_arrays(flat_pa, cell_size, mask=pa_mask)
    tensor_type = pa.fixed_shape_tensor(pa_dtype, shape)
    return pa.ExtensionArray.from_storage(tensor_type, storage)


def build_ragged_array(
    chunks: Sequence[Optional[np.ndarray]], dtype: np.dtype
) -> pa.ListArray:
    """Compile a list of variable-length NumPy arrays into an Arrow ListArray using offsets and validity bitmask."""
    offsets = [0]
    valid_chunks: List[np.ndarray] = []
    mask: List[bool] = []

    for c in chunks:
        is_missing = c is None
        mask.append(is_missing)
        if not is_missing:
            arr = np.ascontiguousarray(c, dtype=dtype)
        else:
            arr = np.empty(0, dtype=dtype)
        offsets.append(offsets[-1] + len(arr))
        valid_chunks.append(arr)

    flat = np.concatenate(valid_chunks) if valid_chunks else np.empty(0, dtype=dtype)
    pa_offsets = pa.array(offsets, type=pa.int32())
    pa_values = pa.array(flat, type=pa.from_numpy_dtype(dtype))
    pa_mask = pa.array(mask, type=pa.bool_()) if any(mask) else None
    return pa.ListArray.from_arrays(pa_offsets, pa_values, mask=pa_mask)


def build_multidim_ragged_array(
    chunks: Sequence[Optional[np.ndarray]], dtype: np.dtype
) -> Union[pa.ListArray, pa.LargeListArray]:
    """Compile a sequence of 2D NumPy arrays into a nested Arrow ListArray[ListArray].

    Optimization:
    Constructing the Arrow ListArray hierarchy directly via contiguous NumPy buffers
    and offsets avoids Awkward Array's C++ `fromiter` traversal, which inspects
    every float scalar individually (~6 µs per scalar).
    """
    offsets_0 = [0]
    offsets_1 = [0]
    valid_flats = []
    mask_0 = []

    for c in chunks:
        if c is None:
            mask_0.append(True)
            offsets_0.append(offsets_0[-1])
        else:
            mask_0.append(False)
            arr = np.ascontiguousarray(c, dtype=dtype)
            if arr.ndim == 2:
                H, W = arr.shape
                offsets_0.append(offsets_0[-1] + H)
                row_offsets = np.arange(1, H + 1, dtype=np.int32) * W + offsets_1[-1]
                offsets_1.extend(row_offsets)
                valid_flats.append(arr.ravel())
            elif arr.ndim == 1:
                offsets_0.append(offsets_0[-1] + len(arr))
                valid_flats.append(arr)
            else:
                # Fallback to Awkward Array for 3D+ structures
                ak_candidate = ak.Array(chunks)
                pa_arr = ak.to_arrow(ak_candidate, extensionarray=False)
                if isinstance(pa_arr, pa.ChunkedArray):
                    pa_arr = pa_arr.combine_chunks()
                return pa_arr

    flat = np.concatenate(valid_flats) if valid_flats else np.empty(0, dtype=dtype)
    pa_flat = pa.array(flat, type=pa.from_numpy_dtype(dtype))
    if len(offsets_1) > 1:
        pa_inner = pa.ListArray.from_arrays(
            pa.array(offsets_1, type=pa.int32()), pa_flat
        )
        pa_mask = pa.array(mask_0, type=pa.bool_()) if any(mask_0) else None
        return pa.ListArray.from_arrays(
            pa.array(offsets_0, type=pa.int32()), pa_inner, mask=pa_mask
        )
    else:
        pa_mask = pa.array(mask_0, type=pa.bool_()) if any(mask_0) else None
        return pa.ListArray.from_arrays(
            pa.array(offsets_0, type=pa.int32()), pa_flat, mask=pa_mask
        )


def is_regular_awkward(arr: ak.Array) -> bool:
    """Check if an Awkward array has regular (non-jagged) dimensions at all depths."""
    if arr.ndim <= 1:
        return True
    for axis in range(1, arr.ndim):
        lengths = ak.num(arr, axis=axis)
        while lengths.ndim > 1:
            lengths = ak.flatten(lengths, axis=1)
        if len(lengths) > 0 and not ak.all(lengths == lengths[0]):
            return False
    return True


def _normalize_offset_dims(offset: Any) -> Tuple[IndexType, ...]:
    match offset:
        case KeyPath() as kp:
            return kp.indices
        case tuple() as t:
            return t
        case list() as l:
            return tuple(l)
        case None:
            return ()
        case int() | slice() | np.ndarray():
            return (offset,)
        case _ if isinstance(offset, Sequence):
            return tuple(offset)
        case _:
            return ()


def read_ragged(sliced: Any, entry: RaggedArraySetEntry, offset: Any) -> Any:
    """Resolve a read on a RaggedArraySetEntry from an Awkward array slice."""
    dims = _normalize_offset_dims(offset)
    is_element_index = len(dims) > 0 and all(isinstance(d, int) for d in dims)
    is_empty = len(dims) == 0

    if is_element_index:
        if isinstance(sliced, ak.Array) and is_regular_awkward(sliced):
            return ak.to_numpy(sliced)
        return sliced

    if (
        isinstance(sliced, ak.Array)
        and sliced.ndim == 2
        and not np.issubdtype(entry.dtype, np.complexfloating)
        and (is_empty or (len(dims) == 1 and isinstance(dims[0], slice)))
    ):
        pa_arr = ak.to_arrow(sliced, extensionarray=False)
        if isinstance(pa_arr, pa.ChunkedArray):
            pa_arr = pa_arr.combine_chunks()
        if isinstance(pa_arr, (pa.ListArray, pa.LargeListArray)):
            return RaggedArrayView(pa_arr, entry.dtype)

    return sliced


def extract_field_from_arrow(col: pa.Array, field_name: str) -> pa.Array:
    """Recursively traverse StructArray or ListArray layers to extract a named child field."""
    mask = col.is_null() if col.null_count > 0 else None
    match col:
        case pa.StructArray():
            return col.field(field_name)
        case pa.ListArray() | pa.LargeListArray():
            inner = extract_field_from_arrow(col.values, field_name)
            return type(col).from_arrays(col.offsets, inner, mask=mask)
        case pa.FixedSizeListArray():
            inner = extract_field_from_arrow(col.values, field_name)
            base = pa.FixedSizeListArray.from_arrays(
                inner, col.type.list_size, mask=mask
            )
            return base.slice(col.offset, len(col))
        case _:
            raise TypeError(
                f"Cannot extract field '{field_name}' from {type(col).__name__}."
            )


def get_nested_arrow_field(col: pa.Array, path: KeyPath) -> pa.Array:
    """Traverse nested StructArray / ListArray layers along a KeyPath to resolve the leaf field array."""
    for seg in path:
        col = extract_field_from_arrow(col, seg)
    return col


def unwrap_extension_for_compute(
    arr: pa.Array,
) -> Tuple[pa.Array, Optional[pa.DataType]]:
    """Unwrap leaf FixedShapeTensorArray to storage array for PyArrow compute kernels."""
    match arr:
        case pa.ListArray() | pa.LargeListArray() | pa.FixedSizeListArray():
            inner, ext_type = unwrap_extension_for_compute(arr.values)
            if ext_type is not None:
                mask = arr.is_null() if arr.null_count > 0 else None
                if isinstance(arr, pa.FixedSizeListArray):
                    base = pa.FixedSizeListArray.from_arrays(
                        inner, arr.type.list_size, mask=mask
                    )
                    return base.slice(arr.offset, len(arr)), ext_type
                else:
                    return (
                        type(arr).from_arrays(arr.offsets, inner, mask=mask),
                        ext_type,
                    )
            return arr, None
        case pa.ExtensionArray():
            return arr.storage, arr.type
        case _:
            return arr, None


def rewrap_extension_after_compute(
    arr: pa.Array, ext_type: Optional[pa.DataType]
) -> pa.Array:
    """Re-wrap storage array back into FixedShapeTensorArray after compute operations."""
    if ext_type is None:
        return arr

    if (
        isinstance(arr, pa.FixedSizeListArray)
        and arr.type.list_size == ext_type.storage_type.list_size
    ):
        return pa.ExtensionArray.from_storage(ext_type, arr)
    if isinstance(arr, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        mask = arr.is_null() if arr.null_count > 0 else None
        wrapped_values = rewrap_extension_after_compute(arr.values, ext_type)

        if isinstance(arr, pa.FixedSizeListArray):
            base = pa.FixedSizeListArray.from_arrays(
                wrapped_values, arr.type.list_size, mask=mask
            )
            return base.slice(arr.offset, len(arr))
        else:
            return type(arr).from_arrays(arr.offsets, wrapped_values, mask=mask)

    return arr


def slice_arrow_array(arr: pa.Array, offset: Any) -> pa.Array:
    """Apply multi-dimensional offset tuple using PyArrow slicing and pc.list_slice."""
    dims = _normalize_offset_dims(offset)
    if not dims:
        return arr

    first, *inner_offsets = dims

    # 1. Dimension 0 (Batch / Row level)
    match first:
        case int(idx):
            norm_idx = idx + len(arr) if idx < 0 else idx
            if norm_idx < 0 or norm_idx >= len(arr):
                raise IndexError(
                    f"Index {idx} out of range for array of length {len(arr)}."
                )

            arr = arr.slice(norm_idx, 1)
        case slice() as sl:
            start, stop, _ = sl.indices(len(arr))
            arr = arr.slice(start, max(0, stop - start))

    # 2. Dimensions 1+ (Nested List / Set levels)
    if inner_offsets:
        unwrapped, ext_type = unwrap_extension_for_compute(arr)
        for inner in inner_offsets:
            match inner:
                case int(idx):
                    unwrapped = pc.list_element(unwrapped, idx)
                case slice() as sl:
                    start = sl.start or 0
                    stop = sl.stop
                    unwrapped = pc.list_slice(unwrapped, start=start, stop=stop)
        arr = rewrap_extension_after_compute(unwrapped, ext_type)

    return arr


def arrow_to_numpy(col: pa.Array, shape: Tuple[Optional[int], ...]) -> np.ndarray:
    """Convert an Arrow array (ExtensionArray, ListArray, or primitive) to an N-D NumPy array."""
    if isinstance(col, pa.ExtensionArray):
        return col.to_numpy_ndarray()

    if isinstance(col, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        outer_dims = [len(col)]
        curr = col

        while isinstance(
            curr, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)
        ):
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


# Legacy aliases for backward compatibility
_is_regular_awkward = is_regular_awkward
_read_ragged = read_ragged
_extract_field_from_arrow = extract_field_from_arrow
_get_nested_arrow_field = get_nested_arrow_field
_unwrap_extension_for_compute = unwrap_extension_for_compute
_rewrap_extension_after_compute = rewrap_extension_after_compute
_slice_arrow_array = slice_arrow_array
_arrow_to_numpy = arrow_to_numpy
