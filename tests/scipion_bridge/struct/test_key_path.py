
import pytest
from scipion_bridge.core.struct.key_path import KeyPath


def test_basic_key_path():

    key = KeyPath()

    key = key.append("foo").append("bar")
    assert key.path == ("root", "foo", "bar")
    assert key.indices == (slice(None), slice(None), slice(None),)

def test_narrow_by_int():

    key = KeyPath()
    key = key.append("foo").append("bar")
     
    key_narrowed = key.narrow_index(5)
    assert key_narrowed.path == ("root", "foo", "bar")
    assert key_narrowed.indices == (slice(None), slice(None), 5,)

    with pytest.raises(ValueError):
        key_narrowed.narrow_index(10)

def test_narrow_slice_by_int_known_span():
    key = KeyPath().append("foo").append("bar")

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
    key = KeyPath().append("foo").append("bar")
    slice_key = key.narrow_slice(slice(3, 8))  # span of 5 items (relative 0..4)

    # Positive index out of bounds (relative index 5 on length 5)
    with pytest.raises(IndexError):
        slice_key.narrow_index(5)

    # Negative index out of bounds (relative index -6 on length 5)
    with pytest.raises(IndexError):
        slice_key.narrow_index(-6)


def test_narrow_slice_by_int_open_ended():
    key = KeyPath().append("foo").append("bar")

    # Parent slice is [4:] -> positive relative indices offset from 4
    open_key = key.narrow_slice(slice(4, None))
    narrowed = open_key.narrow_index(3)
    assert narrowed.indices[-1] == 7  # 4 + 3

    # Negative index on an open-ended slice cannot resolve without sequence length
    with pytest.raises(ValueError, match="without knowing sequence length"):
        open_key.narrow_index(-1)


def test_narrow_slice_by_int_right_anchored_stop():
    key = KeyPath().append("foo").append("bar")

    # Parent slice is [:-2] -> negative index offsets further from the end
    # index -1 inside [:-2] means the item right before -2 -> -3
    right_anchored_key = key.narrow_slice(slice(None, -2))
    narrowed = right_anchored_key.narrow_index(-1)
    assert narrowed.indices[-1] == -3

    # Positive index on a right-anchored slice cannot resolve without sequence length
    with pytest.raises(ValueError, match="without knowing sequence length"):
        right_anchored_key.narrow_index(2)


def test_narrow_by_slice():

    key = KeyPath()
    key = key.append("foo").append("bar")

    narrowed_key = key.narrow_slice(slice(3, 10)).narrow_slice(slice(1, 10))
    assert narrowed_key.components[-1] == ("bar", slice(4, 10, 1))


def test_narrow_by_slice_clamping_and_bounds():
    key = KeyPath().append("foo").append("bar")

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
    key = KeyPath().append("foo").append("bar")

    # Mixed-sign anchors raise ValueError without concrete sequence length
    with pytest.raises(ValueError):
        key.narrow_slice(slice(4, None)).narrow_slice(slice(-2, None))

    # Unsupported step
    with pytest.raises(ValueError):
        key.narrow_slice(slice(0, 10, 2))


if __name__ == "__main__":
    test_narrow_by_slice_invalid_anchors()