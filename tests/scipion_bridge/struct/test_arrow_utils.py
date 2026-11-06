"""Converting Arrow arrays to NumPy, independently of the pyarrow version."""

import numpy as np
import pyarrow as pa
import pytest

from scipion_bridge.core.struct.utils.arrow_utils import (
    arrow_to_numpy,
    build_tensor_array,
    tensor_to_numpy,
)


def _cells(n: int) -> np.ndarray:
    return np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)


def test_null_tensor_cells_keep_their_stored_values():
    # pyarrow >= 26 raises on null cells in to_numpy_ndarray().
    cells = _cells(3)
    column = build_tensor_array(
        cells, (2, 2), np.dtype(np.float32), mask=[False, True, False]
    )

    assert column.null_count == 1
    np.testing.assert_array_equal(tensor_to_numpy(column), cells)


def test_a_sliced_tensor_array_starts_at_its_offset():
    cells = _cells(4)
    column = build_tensor_array(
        cells, (2, 2), np.dtype(np.float32), mask=[False, True, False, False]
    )

    np.testing.assert_array_equal(tensor_to_numpy(column.slice(1, 2)), cells[1:3])


def test_tensor_cells_nested_in_lists_are_converted():
    cells = _cells(4)
    leaves = build_tensor_array(
        cells, (2, 2), np.dtype(np.float32), mask=[True, False, False, False]
    )
    column = pa.FixedSizeListArray.from_arrays(leaves, 2)

    np.testing.assert_array_equal(
        arrow_to_numpy(column, (2, 2)),
        cells.reshape(2, 2, 2, 2),
    )


def test_permuted_tensor_layouts_are_rejected():
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (2, 2), permutation=[1, 0])
    storage = pa.FixedSizeListArray.from_arrays(pa.array(_cells(1).ravel()), 4)
    column = pa.ExtensionArray.from_storage(tensor_type, storage)

    with pytest.raises(AssertionError, match="Permuted"):
        tensor_to_numpy(column)
