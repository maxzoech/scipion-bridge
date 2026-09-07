"""Utilities for bridging scipion-bridge schemas/structures with Apache Arrow."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union, overload
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

from ..schema import Schema, ArrayEntryBase


class RaggedArrayView(Sequence[NDArray]):
    """A zero-copy sequence view over an Apache Arrow ListArray returning 1D NumPy slices."""

    def __init__(self, list_array: pa.ListArray, dtype: np.dtype) -> None:
        self._list_array = list_array
        self.dtype = np.dtype(dtype)

    def __len__(self) -> int:
        return len(self._list_array)

    @overload
    def __getitem__(self, item: int) -> NDArray: ...

    @overload
    def __getitem__(self, item: slice) -> "RaggedArrayView": ...

    def __getitem__(self, item: Union[int, slice]) -> Union[NDArray, "RaggedArrayView"]:
        if isinstance(item, slice):
            sliced = self._list_array[item]
            return RaggedArrayView(sliced, self.dtype)
        if isinstance(item, int):
            if item < 0:
                item += len(self)
            if item < 0 or item >= len(self):
                raise IndexError(
                    f"Index {item} out of range for RaggedArrayView of length {len(self)}."
                )
            scalar = self._list_array[item]
            if not scalar.is_valid:
                return np.empty(0, dtype=self.dtype)
            return scalar.values.to_numpy(zero_copy_only=False)
        raise TypeError(f"Invalid RaggedArrayView index type '{type(item).__name__}'.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_list(self) -> List[NDArray]:
        return list(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (RaggedArrayView, list, tuple)):
            if len(self) != len(other):
                return False
            return all(np.array_equal(a, b) for a, b in zip(self, other))
        return False

    def __repr__(self) -> str:
        if len(self) <= 3:
            shapes = [tuple(arr.shape) for arr in self]
            return f"RaggedArrayView(len={len(self)}, shapes={shapes}, dtype={self.dtype})"
        shapes = [tuple(self[i].shape) for i in range(3)]
        return f"RaggedArrayView(len={len(self)}, shapes={shapes}..., dtype={self.dtype})"


def schema_to_arrow_schema(schema: Schema) -> pa.Schema:
    """Recursively convert a scipion-bridge Schema into a pyarrow.Schema."""
    fields: List[pa.Field] = []

    for name, entry in schema.fields.items():
        if entry.children is not None:
            child_pa_schema = schema_to_arrow_schema(entry.children)
            field_type = pa.struct([child_pa_schema.field(i) for i in range(len(child_pa_schema))])
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
            raise TypeError(f"Unsupported schema entry type '{type(entry).__name__}' for field '{name}'.")

    return pa.schema(fields)


def build_tensor_array(data: np.ndarray, shape: Tuple[int, ...], dtype: np.dtype) -> pa.ExtensionArray:
    """Compile a contiguous multidimensional NumPy array into an Arrow FixedShapeTensorArray."""
    cell_size = int(np.prod(shape))
    pa_dtype = pa.from_numpy_dtype(dtype)
    flat_data = np.ascontiguousarray(data, dtype=dtype).ravel()
    flat_pa = pa.array(flat_data, type=pa_dtype)
    storage = pa.FixedSizeListArray.from_arrays(flat_pa, cell_size)
    tensor_type = pa.fixed_shape_tensor(pa_dtype, shape)
    return pa.ExtensionArray.from_storage(tensor_type, storage)


def build_ragged_array(chunks: Sequence[Optional[np.ndarray]], dtype: np.dtype) -> pa.ListArray:
    """Compile a list of variable-length NumPy arrays into an Arrow ListArray using offsets."""
    offsets = [0]
    valid_chunks: List[np.ndarray] = []

    for c in chunks:
        if c is not None:
            arr = np.ascontiguousarray(c, dtype=dtype)
        else:
            arr = np.empty(0, dtype=dtype)
        offsets.append(offsets[-1] + len(arr))
        valid_chunks.append(arr)

    flat = np.concatenate(valid_chunks) if valid_chunks else np.empty(0, dtype=dtype)
    pa_offsets = pa.array(offsets, type=pa.int32())
    pa_values = pa.array(flat, type=pa.from_numpy_dtype(dtype))
    return pa.ListArray.from_arrays(pa_offsets, pa_values)
