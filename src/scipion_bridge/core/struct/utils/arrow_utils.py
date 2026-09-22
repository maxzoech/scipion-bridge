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

    if is_element_index:
        if isinstance(sliced, ak.Array) and is_regular_awkward(sliced):
            return ak.to_numpy(sliced)
        return sliced

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
