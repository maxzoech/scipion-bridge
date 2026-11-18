"""Converting Arrow arrays to NumPy, independently of the pyarrow version."""

import subprocess
import sys

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


# Converts arrays with axes of size 0 in a fresh interpreter: PyArrow kills the
# process (SIGFPE) instead of raising if a conversion divides by a list size
# of 0, which would take pytest down with it.
_EMPTY_AXES = """
import pickle
import numpy as np
import scipion_bridge as B
from scipion_bridge.core.struct.utils.arrow_utils import (
    regular_arrow_to_numpy,
    regular_numpy_to_arrow,
)

for shape in [(3, 0), (3, 0, 4), (3, 4, 0), (0, 0)]:
    data = np.zeros(shape, np.float32)
    restored = regular_arrow_to_numpy(regular_numpy_to_arrow(data))
    assert restored.shape == shape, (restored.shape, shape)

class Members(B.Struct):
    x: B.Array[np.float64] = B.Array(shape=(None,))

members = Members(x=np.zeros(0))
assert pickle.loads(pickle.dumps(members)).x.shape == (0,)
print("ok")
"""


def test_arrays_with_axes_of_size_0_convert_and_pickle():
    result = subprocess.run(
        [sys.executable, "-c", _EMPTY_AXES],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
