import awkward as ak
import numpy as np
import pyarrow as pa
import pytest

from scipion_bridge.core.struct.buffers import (
    as_regular_numpy,
    assign_rows,
    concat_columns,
    has_numpy_layout,
    pad_to_capacity,
    regularized,
    stack_rows,
)
from scipion_bridge.core.struct.utils.arrow_utils import (
    is_regular_arrow,
    regular_arrow_to_numpy,
    regular_numpy_to_arrow,
)


def test_as_regular_numpy_rejects_ragged_and_object_data():
    assert as_regular_numpy([[1, 2], [3]], float) is None
    assert as_regular_numpy([object()], None) is None
    regular = as_regular_numpy([[1, 2], [3, 4]], float)
    assert regular is not None and regular.shape == (2, 2)


def test_pad_to_capacity_pads_numpy_with_zeros_and_awkward_with_missing_rows():
    padded = pad_to_capacity(np.ones((2, 3)), 4, np.float32)
    assert padded.shape == (4, 3)
    assert padded.dtype == np.float32
    assert padded[2:].sum() == 0

    assert ak.to_list(pad_to_capacity(ak.Array([[1.0]]), 2, float)) == [[1.0], None]


def test_pad_to_capacity_keeps_columns_at_or_above_capacity():
    column = np.ones((3, 2))
    assert pad_to_capacity(column, None, float) is column
    assert pad_to_capacity(column, 2, float) is column


def test_stack_rows_masks_missing_rows():
    stacked = stack_rows([np.zeros(2), None, np.ones(1)])
    assert ak.to_list(stacked) == [[0.0, 0.0], None, [1.0]]


def test_stack_rows_of_missing_rows_only():
    assert ak.to_list(stack_rows([None, None])) == [None, None]


def test_assign_rows_writes_one_element_per_selected_row():
    rows: list = [None] * 4
    assign_rows(rows, (slice(1, 3),), [np.zeros(1), np.ones(2)])
    assign_rows(rows, (np.array([3]),), [np.full(3, 2.0)])

    assert rows[0] is None
    assert [len(row) for row in rows[1:]] == [1, 2, 3]


def test_assign_rows_rejects_length_mismatch_instead_of_resizing():
    rows: list = [None] * 4

    with pytest.raises(ValueError, match="Cannot assign 1 elements to 2 rows"):
        assign_rows(rows, (slice(0, 2),), [np.zeros(1)])

    assert len(rows) == 4


def test_assign_rows_rejects_unsupported_index():
    with pytest.raises(TypeError, match="Unsupported index"):
        assign_rows([None], ("a",), np.zeros(1))  # type: ignore[arg-type]


def test_stack_rows_of_one_shape_gives_numpy():
    stacked = stack_rows([np.zeros((2, 3)), np.ones((2, 3))])

    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (2, 2, 3)


def test_stack_rows_of_regular_awkward_rows_gives_numpy():
    stacked = stack_rows([ak.Array([[1.0, 2.0]]), np.array([[3.0, 4.0]])])

    assert isinstance(stacked, np.ndarray)
    assert stacked.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]]]


def test_stack_rows_of_different_shapes_gives_awkward():
    stacked = stack_rows([np.zeros((2, 2)), np.ones((3, 3))])

    assert isinstance(stacked, ak.Array)
    assert [np.asarray(row).shape for row in stacked] == [(2, 2), (3, 3)]


def test_concat_columns_of_one_row_shape_gives_numpy():
    joined = concat_columns([np.zeros((2, 3, 3)), np.ones((1, 3, 3))])

    assert isinstance(joined, np.ndarray)
    assert joined.shape == (3, 3, 3)


def test_concat_columns_of_different_row_shapes_gives_awkward():
    joined = concat_columns([np.zeros((2, 3, 3)), np.ones((1, 2, 2))])

    assert isinstance(joined, ak.Array)
    assert [np.asarray(row).shape for row in joined] == [(3, 3), (3, 3), (2, 2)]


def test_concat_columns_with_awkward_gives_awkward():
    joined = concat_columns([np.zeros((1, 2)), ak.Array([[1.0, 2.0, 3.0]])])

    assert isinstance(joined, ak.Array)
    assert ak.to_list(joined) == [[0.0, 0.0], [1.0, 2.0, 3.0]]


def test_concat_columns_ignores_empty_columns():
    joined = concat_columns([np.zeros((0,)), np.ones((2, 3, 3))])

    assert isinstance(joined, np.ndarray)
    assert joined.shape == (2, 3, 3)


def test_regularized_converts_only_complete_regular_columns():
    assert isinstance(regularized(ak.Array(np.zeros((2, 3)))), np.ndarray)
    assert isinstance(regularized(ak.Array([[1.0, 2.0], [3.0, 4.0]])), np.ndarray)
    assert isinstance(regularized(ak.Array([[1.0], [2.0, 3.0]])), ak.Array)
    assert isinstance(regularized(ak.Array([[1.0], None])), ak.Array)
    assert isinstance(regularized(ak.Array([[1.0, None]])), ak.Array)


def test_has_numpy_layout_only_for_regular_dimensions():
    assert has_numpy_layout(ak.Array(np.zeros((2, 3, 3))))
    assert not has_numpy_layout(ak.Array([[1.0, 2.0], [3.0, 4.0]]))


def test_regular_numpy_column_arrow_round_trip_without_copy():
    column = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    exported = regular_numpy_to_arrow(column)
    restored = regular_arrow_to_numpy(exported)

    assert exported.type == ak.to_arrow(ak.Array(column), extensionarray=False).type
    assert is_regular_arrow(exported)
    assert restored.dtype == np.float32
    assert np.array_equal(restored, column)
    assert np.shares_memory(restored, column)


def test_regular_arrow_to_numpy_respects_slices():
    column = np.arange(24, dtype=np.float32).reshape(4, 3, 2)

    restored = regular_arrow_to_numpy(regular_numpy_to_arrow(column).slice(1, 2))

    assert np.array_equal(restored, column[1:3])


def test_is_regular_arrow_rejects_ragged_and_nullable_columns():
    assert not is_regular_arrow(pa.array([[1.0], [2.0, 3.0]]))
    assert not is_regular_arrow(pa.array([[1.0, 2.0], None], pa.list_(pa.float64(), 2)))
    assert not is_regular_arrow(pa.array([[1.0, None]], pa.list_(pa.float64(), 2)))
