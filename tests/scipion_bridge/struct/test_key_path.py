import numpy as np
import pytest
from scipion_bridge.core.struct import KeyPath


def _bar():
    """root.foo.bar[:]: the nested Set ``bar`` of the struct field ``foo`` of
    a Struct, the only component with an axis."""
    return KeyPath((("root", None),)).append("foo", axis=False).append("bar", axis=True)


def _single(name):
    """root.<name>[:]: a nested Set field of a Struct."""
    return KeyPath((("root", None),)).append(name, axis=True)


def test_basic_key_path():

    key = _bar()
    assert key.path == ("root", "foo", "bar")
    assert key.indices == (None, None, slice(None))


def test_narrow_by_int():

    key = _bar()

    key_narrowed = key.narrow_index(5)
    assert key_narrowed.path == ("root", "foo", "bar")
    assert key_narrowed.indices == (None, None, 5)

    with pytest.raises(ValueError):
        key_narrowed.narrow_index(10)


def test_narrow_slice_by_int_known_span():
    key = _bar()

    # Parent slice is [3:10] -> span of 7 items (relative indices 0..6)
    slice_key = key.narrow_slice(slice(3, 10))

    # Positive index: index 2 within [3:10] -> absolute index 3 + 2 = 5
    narrowed_pos = slice_key.narrow_index(2)
    assert narrowed_pos.indices[-1] == 5

    # Negative index: index -1 within [3:10] -> absolute index 10 - 1 = 9
    narrowed_neg = slice_key.narrow_index(-1)
    assert narrowed_neg.indices[-1] == 9

    # Another negative index: index -3 within [3:10] -> absolute index 10 - 3 = 7
    narrowed_neg_offset = slice_key.narrow_index(-3)
    assert narrowed_neg_offset.indices[-1] == 7


def test_narrow_slice_by_int_out_of_bounds():
    key = _bar()
    slice_key = key.narrow_slice(slice(3, 8))  # span of 5 items (relative 0..4)

    # Positive index out of bounds (relative index 5 on length 5)
    with pytest.raises(IndexError):
        slice_key.narrow_index(5)

    # Negative index out of bounds (relative index -6 on length 5)
    with pytest.raises(IndexError):
        slice_key.narrow_index(-6)


def test_narrow_slice_by_int_open_ended():
    key = _bar()

    # Parent slice is [4:] -> positive relative indices offset from 4
    open_key = key.narrow_slice(slice(4, None))
    narrowed = open_key.narrow_index(3)
    assert narrowed.indices[-1] == 7  # 4 + 3

    # Negative index on an open-ended slice cannot resolve without sequence length
    with pytest.raises(ValueError, match="without knowing sequence length"):
        open_key.narrow_index(-1)


def test_narrow_slice_by_int_right_anchored_stop():
    key = _bar()

    # Parent slice is [:-2] -> negative index offsets further from the end
    # index -1 inside [:-2] means the item right before -2 -> -3
    right_anchored_key = key.narrow_slice(slice(None, -2))
    narrowed = right_anchored_key.narrow_index(-1)
    assert narrowed.indices[-1] == -3

    # Positive index on a right-anchored slice cannot resolve without sequence length
    with pytest.raises(ValueError, match="without knowing sequence length"):
        right_anchored_key.narrow_index(2)


def test_narrow_by_slice():

    key = _bar()

    narrowed_key = key.narrow_slice(slice(3, 10)).narrow_slice(slice(1, 10))
    assert narrowed_key.components[-1] == ("bar", slice(4, 10, 1))


def test_narrow_by_slice_clamping_and_bounds():
    key = _bar()

    # Case 1: Within bounds -> [0:10][1:3] => [1:3]
    k1 = key.narrow_slice(slice(0, 10)).narrow_slice(slice(1, 3))
    assert k1.components[-1][1] == slice(1, 3, 1)

    # Case 2: Open-ended parent -> [5:][:4] => [5:9]
    k2 = key.narrow_slice(slice(5, None)).narrow_slice(slice(None, 4))
    assert k2.components[-1][1] == slice(5, 9, 1)

    # Case 3: Right-anchored offset -> [2:-2][:-1] => [2:-3]
    k3 = key.narrow_slice(slice(2, -2)).narrow_slice(slice(None, -1))
    assert k3.components[-1][1] == slice(2, -3, 1)


def test_narrow_by_slice_invalid_anchors():
    key = _bar()

    # Mixed-sign anchors raise ValueError without concrete sequence length
    with pytest.raises(ValueError):
        key.narrow_slice(slice(4, None)).narrow_slice(slice(-2, None))

    # Unsupported step
    with pytest.raises(ValueError):
        key.narrow_slice(slice(0, 10, 2))


def test_str_and_repr():
    key = _bar()
    assert str(key) == "root.foo.bar[:]"
    assert repr(key) == "KeyPath('root.foo.bar[:]')"

    narrowed = key.narrow_slice(slice(3, 10)).narrow_index(2)
    assert str(narrowed) == "root.foo.bar[5]"
    assert repr(narrowed) == "KeyPath('root.foo.bar[5]')"

    array_key = key.narrow_indices([0, 2, 5])
    assert str(array_key) == "root.foo.bar[[0, 2, 5]]"
    assert repr(array_key) == "KeyPath('root.foo.bar[[0, 2, 5]]')"


def test_narrow_indices():
    key = _single("foo")

    # 1. On unbounded slice with length
    k1 = key.narrow_indices([0, -1, 2], length=5)
    np.testing.assert_array_equal(k1.indices[-1], np.array([0, 4, 2]))

    # Out of bounds with length
    with pytest.raises(IndexError):
        key.narrow_indices([0, 5], length=5)
    with pytest.raises(IndexError):
        key.narrow_indices([-6], length=5)

    # Negative index without length
    with pytest.raises(ValueError):
        key.narrow_indices([0, -1])

    # 2. On known span slice [3:10] (span 7)
    slice_key = key.narrow_slice(slice(3, 10))
    k2 = slice_key.narrow_indices([0, -1, 2])
    np.testing.assert_array_equal(k2.indices[-1], np.array([3, 9, 5]))

    with pytest.raises(IndexError):
        slice_key.narrow_indices([7])
    with pytest.raises(IndexError):
        slice_key.narrow_indices([-8])

    # 3. Chained narrow_indices (taking from array index)
    k3 = k2.narrow_indices([1, 0])
    np.testing.assert_array_equal(k3.indices[-1], np.array([9, 3]))

    # 4. Chained narrow_index (scalar from array index)
    k4 = k2.narrow_index(1)
    assert k4.indices[-1] == 9
    assert isinstance(k4.indices[-1], int)

    k4_neg = k2.narrow_index(-1)
    assert k4_neg.indices[-1] == 5

    with pytest.raises(IndexError):
        k2.narrow_index(3)

    # 5. Chained narrow_slice (slice of array index)
    k5 = k2.narrow_slice(slice(1, 3))
    np.testing.assert_array_equal(k5.indices[-1], np.array([9, 5]))


def test_narrow_mask():
    key = _single("foo")

    # 1. On unbounded slice with length
    k1 = key.narrow_mask([True, False, True, False, True], length=5)
    np.testing.assert_array_equal(k1.indices[-1], np.array([0, 2, 4]))

    # Length mismatch
    with pytest.raises(IndexError):
        key.narrow_mask([True, False], length=5)

    # Unbounded without length succeeds and converts True positions to indices
    k1_no_len = key.narrow_mask([True, False, True])
    np.testing.assert_array_equal(k1_no_len.indices[-1], np.array([0, 2]))

    # 2. On known span slice [2:7] (span 5)
    slice_key = key.narrow_slice(slice(2, 7))
    k2 = slice_key.narrow_mask([True, False, False, True, False])
    np.testing.assert_array_equal(k2.indices[-1], np.array([2, 5]))

    with pytest.raises(IndexError):
        slice_key.narrow_mask([True, False])

    # 3. On open right bound [2:] offsets by start
    open_key = key.narrow_slice(slice(2, None))
    k_open = open_key.narrow_mask([True, False, True])
    np.testing.assert_array_equal(k_open.indices[-1], np.array([2, 4]))

    # Negative bounds slice without length raises ValueError
    neg_key = key.narrow_slice(slice(None, -2))
    with pytest.raises(ValueError):
        neg_key.narrow_mask([True, False])

    # 4. Chained narrow_mask on array index
    k3 = k1.narrow_mask([True, False, True])
    np.testing.assert_array_equal(k3.indices[-1], np.array([0, 4]))

    with pytest.raises(IndexError):
        k1.narrow_mask([True, False])


def test_keypath_eq_and_hash():
    key = _single("foo")

    k1 = key.narrow_indices([0, 2, 4])
    k2 = key.narrow_indices(np.array([0, 2, 4]))
    k3 = key.narrow_indices([0, 2, 5])

    assert k1 == k2
    assert k1 != k3
    assert k1 != key

    # KeyPath with numpy array raises TypeError on hash
    with pytest.raises(TypeError, match="unhashable"):
        hash(k1)

    # Standard KeyPath without array remains hashable
    assert isinstance(hash(key), int)


def test_narrow_slice_by_int_with_length():
    key = _single("items")

    # Parent slice [4:] with view length 6 (e.g. underlying buffer of 10)
    open_key = key.narrow_slice(slice(4, None))
    assert open_key.narrow_index(0, length=6).indices[-1] == 4
    assert open_key.narrow_index(2, length=6).indices[-1] == 6
    assert open_key.narrow_index(-1, length=6).indices[-1] == 9
    assert open_key.narrow_index(-6, length=6).indices[-1] == 4

    # Out of bounds with length
    with pytest.raises(IndexError):
        open_key.narrow_index(6, length=6)
    with pytest.raises(IndexError):
        open_key.narrow_index(-7, length=6)


def test_narrow_by_slice_with_length_mixed_signs():
    key = _single("items")

    # Parent slice [4:] with view length 6
    open_key = key.narrow_slice(slice(4, None))

    # Mixed-sign: open parent with negative child slice [-2:]
    k_mixed = open_key.narrow_slice(slice(-2, None), length=6)
    assert k_mixed.components[-1][1] == slice(8, 10, 1)

    # Sub-slice with positive start and stop
    k_sub = open_key.narrow_slice(slice(1, 4), length=6)
    assert k_sub.components[-1][1] == slice(5, 8, 1)


def test_narrow_indices_with_length_open_slice():
    key = _single("items")

    # Parent slice [4:] with view length 6
    open_key = key.narrow_slice(slice(4, None))
    k = open_key.narrow_indices([0, -1, 2], length=6)
    np.testing.assert_array_equal(k.indices[-1], np.array([4, 9, 6]))

    with pytest.raises(IndexError):
        open_key.narrow_indices([6], length=6)
    with pytest.raises(IndexError):
        open_key.narrow_indices([-7], length=6)


def test_narrow_mask_with_length_open_slice():
    key = _single("items")

    open_key = key.narrow_slice(slice(4, None))
    k = open_key.narrow_mask([True, False, True, False, False, True], length=6)
    np.testing.assert_array_equal(k.indices[-1], np.array([4, 6, 9]))

    with pytest.raises(IndexError):
        open_key.narrow_mask([True, False], length=6)


# -- Axes --------------------------------------------------------------------
def test_append_requires_declaring_the_axis():
    with pytest.raises(TypeError, match="axis"):
        KeyPath().append("foo")  # type: ignore[call-arg]


def test_narrowing_selects_along_the_innermost_axis():
    # The fields of a struct share the axis of the Set holding them.
    key = KeyPath().append("ctf", axis=False).append("defocus", axis=False)

    narrowed = key.narrow_slice(slice(1, 3)).narrow_index(1)

    assert narrowed.indices == (2, None, None)
    assert str(narrowed) == "root[2].ctf.defocus"


def test_a_pinned_outer_axis_lets_a_nested_axis_be_narrowed():
    # The members of group 1 of a Set of groups.
    key = KeyPath().narrow_index(1).append("members", axis=True)

    narrowed = key.narrow_slice(slice(0, 2))

    assert narrowed.indices == (1, slice(0, 2, 1))


def test_a_path_without_an_axis_cannot_be_narrowed():
    struct_field = KeyPath((("root", None),)).append("ctf", axis=False)

    with pytest.raises(ValueError, match="no axis"):
        struct_field.narrow_index(0)


@pytest.mark.parametrize(
    "outer",
    [slice(None), slice(0, 3, 1), np.array([2, 0])],
    ids=["full", "slice", "positions"],
)
def test_a_path_over_several_free_axes_narrows_its_inner_axis(outer):
    # The members of every selected group of a Set of groups.
    key = KeyPath((("root", outer),)).append("members", axis=True)

    narrowed = key.narrow_slice(slice(0, 1))

    assert key.spans_several_axes()
    assert narrowed.components[0] == key.components[0]
    assert narrowed.components[-1] == ("members", slice(0, 1, 1))


def test_a_pinned_outer_axis_spans_one_axis():
    assert (
        not KeyPath().narrow_index(1).append("members", axis=True).spans_several_axes()
    )
    assert not KeyPath().append("ctf", axis=False).spans_several_axes()


def test_axis_indices_keep_the_axes_in_order():
    key = (
        KeyPath()
        .narrow_index(4)
        .append("ctf", axis=False)
        .append("members", axis=True)
        .narrow_slice(slice(1, 3))
        .append("value", axis=False)
    )

    assert key.axis_indices() == (4, slice(1, 3, 1))
