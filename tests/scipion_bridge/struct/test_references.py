"""Reference columns: rows of a source held by a Set (internal API).

A Set whose ``pixels`` reference rows of a source must behave exactly like a
Set owning the same values through every public operation; the internal
functions of ``references.py`` see the difference.
"""

import os
import pickle

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.struct import concat
from scipion_bridge.core.struct import references
from scipion_bridge.core.struct.references import PicklingMode
from scipion_bridge.core.struct.source import (
    ArraySource,
    ColumnSource,
    FileId,
    Reference,
    StaleReferenceError,
)


class Image(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(4, 4))
    defocus: float


class CountingSource(ArraySource):
    """An in-memory source that counts its reads."""

    def __init__(self, array, files=()):
        super().__init__(array, files)
        self.reads = 0

    def _read(self, index):
        self.reads += 1
        return super()._read(index)


def _rows(n, offset=0):
    return np.arange(offset, offset + n * 16, dtype=np.float32).reshape(n, 4, 4)


def _referencing(n=6, source=None, offset=0):
    """A Set whose pixels reference ``source``, and its eager twin."""
    source = source or CountingSource(_rows(n, offset))
    lazy = B.Set[Image](capacity=n)
    lazy["defocus"] = np.arange(n, dtype=np.float64) + offset
    references.set_reference(lazy, ("pixels",), Reference(source, range(n)))
    eager = B.Set[Image](capacity=n)
    eager["defocus"] = np.arange(n, dtype=np.float64) + offset
    eager["pixels"] = _rows(n, offset)
    return lazy, eager, source


def _states(s):
    return {path: state.state for path, state in references.column_states(s).items()}


def _same(a, b):
    assert len(a) == len(b)
    np.testing.assert_array_equal(a["pixels"], b["pixels"])
    np.testing.assert_array_equal(a["defocus"], b["defocus"])


# -- Reading and writing -------------------------------------------------------
def test_a_reference_is_read_once_and_cached():
    lazy, eager, source = _referencing()
    assert _states(lazy)[("pixels",)] == "reference"
    assert len(lazy) == 6 and source.reads == 0

    _same(lazy, eager)
    _same(lazy, eager)

    assert source.reads == 1
    assert _states(lazy)[("pixels",)] == "cached"


def test_a_write_makes_the_column_owned_without_reading_the_source():
    lazy, _, source = _referencing()

    lazy["pixels"] = np.zeros((6, 4, 4), dtype=np.float32)

    assert source.reads == 0
    assert _states(lazy)[("pixels",)] == "owned"
    assert not lazy["pixels"].any()


def test_an_indexed_write_keeps_the_other_rows():
    lazy, eager, _ = _referencing()

    lazy[2].pixels = np.ones((4, 4), dtype=np.float32)
    eager[2].pixels = np.ones((4, 4), dtype=np.float32)

    _same(lazy, eager)
    assert _states(lazy)[("pixels",)] == "owned"


def test_columns_stay_writable_in_place_as_before():
    lazy, eager, _ = _referencing()

    lazy["pixels"][0] += 1
    eager["pixels"][0] += 1

    _same(lazy, eager)


def test_verification_catches_writes_in_place():
    lazy, _, _ = _referencing()
    lazy["pixels"][0, 0, 0] = -1.0
    assert _states(lazy)[("pixels",)] == "cached"

    references.verify(lazy)

    assert _states(lazy)[("pixels",)] == "owned"
    assert lazy["pixels"][0, 0, 0] == -1.0


def test_prefetch_reads_the_references():
    lazy, _, source = _referencing()

    references.prefetch(lazy)

    assert source.reads == 1
    assert _states(lazy)[("pixels",)] == "cached"


# -- Views and concatenation ---------------------------------------------------
@pytest.mark.parametrize(
    "select",
    [
        lambda s: s[1:4],
        lambda s: s[[5, 0, 3]],
        lambda s: s[np.array([True, False, True, False, False, True])],
        lambda s: s[1:5][1:3],
    ],
)
def test_views_narrow_the_reference(select):
    lazy, eager, source = _referencing()

    view = select(lazy)
    state = references.column_states(view)[("pixels",)]

    assert (state.state, state.rows) == ("reference", len(select(eager)))
    _same(view, select(eager))


def test_concat_joins_references_without_reading_them():
    a, ea, sa = _referencing(3)
    b, eb, sb = _referencing(4, offset=1000)

    joined = concat([a[1:], b])

    assert sa.reads == sb.reads == 0
    assert _states(joined)[("pixels",)] == "reference"
    _same(joined, concat([ea[1:], eb]))


def test_concat_of_read_references_keeps_rows_and_digests():
    a, ea, _ = _referencing(3)
    b, eb, _ = _referencing(4, offset=1000)
    references.prefetch(a)
    references.prefetch(b)

    joined = concat([a, b])

    assert _states(joined)[("pixels",)] == "cached"
    joined["pixels"][0, 0, 0] = -5.0
    references.verify(joined)
    assert _states(joined)[("pixels",)] == "owned"


def test_concat_of_a_reference_and_an_owned_column_owns_it():
    a, ea, _ = _referencing(3)
    _, eb, _ = _referencing(2)

    joined = concat([a, eb])

    assert _states(joined)[("pixels",)] == "owned"
    _same(joined, concat([ea, eb]))


# -- Pickling ------------------------------------------------------------------
def test_pickling_for_transfer_keeps_rows_and_the_reference():
    lazy, eager, _ = _referencing()
    references.prefetch(lazy)

    restored = pickle.loads(pickle.dumps(lazy))

    assert _states(restored)[("pixels",)] == "cached"
    _same(restored, eager)


def test_pickling_an_unread_reference_sends_only_the_reference():
    lazy, eager, source = _referencing()

    data = pickle.dumps(lazy)
    restored = pickle.loads(data)

    assert source.reads == 0
    assert _states(restored)[("pixels",)] == "reference"
    _same(restored, eager)


def test_spilling_drops_the_rows_of_unchanged_references():
    lazy, eager, _ = _referencing(64)
    references.prefetch(lazy)
    transfer = pickle.dumps(lazy)

    with references.pickling(PicklingMode.SPILL):
        spilled = pickle.dumps(lazy)

    # The source travels with the reference here (an in-memory source), so
    # compare with the rows alone.
    assert len(spilled) < len(transfer)
    restored = pickle.loads(spilled)
    assert _states(restored)[("pixels",)] == "reference"
    _same(restored, eager)


def test_spilling_keeps_the_rows_of_references_written_in_place():
    lazy, _, _ = _referencing()
    references.prefetch(lazy)
    lazy["pixels"][1] = 7.0

    with references.pickling(PicklingMode.SPILL):
        restored = pickle.loads(pickle.dumps(lazy))

    assert _states(restored)[("pixels",)] == "owned"
    assert (restored["pixels"][1] == 7.0).all()


def test_pickling_a_view_pickles_its_rows_of_the_reference():
    lazy, eager, _ = _referencing()

    restored = pickle.loads(pickle.dumps(lazy[2:5]))

    assert references.column_states(restored)[("pixels",)].rows == 3
    _same(restored, eager[2:5])


def test_to_arrow_is_unchanged():
    lazy, eager, _ = _referencing()

    assert lazy.to_arrow().equals(eager.to_arrow())


# -- Changed inputs ------------------------------------------------------------
class FileSource(ColumnSource):
    """Rows stored in a .npy file."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._files = (FileId.of(path),)

    @property
    def rows(self):
        return len(np.load(self.path))

    @property
    def nbytes_per_row(self):
        return 64

    @property
    def files(self):
        return self._files

    def _read(self, index):
        array = np.load(self.path)
        return (
            array[slice(index.start, index.stop)]
            if isinstance(index, range)
            else array[index]
        )


def _file_backed(tmp_path):
    path = tmp_path / "rows.npy"
    np.save(path, _rows(4))
    lazy = B.Set[Image](capacity=4)
    references.set_reference(lazy, ("pixels",), Reference(FileSource(path), range(4)))
    references.prefetch(lazy)
    with references.pickling(PicklingMode.SPILL):
        spilled = pickle.dumps(lazy)
    return path, spilled


def test_a_changed_file_fails_loudly(tmp_path):
    path, spilled = _file_backed(tmp_path)
    np.save(path, _rows(5))

    with pytest.raises(StaleReferenceError, match="changed"):
        pickle.loads(spilled)["pixels"]


def test_a_same_size_rewrite_with_the_old_mtime_is_caught_by_the_digests(tmp_path):
    path, spilled = _file_backed(tmp_path)
    stat = os.stat(path)
    changed = _rows(4)
    changed[2] = -1
    np.save(path, changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(StaleReferenceError, match=r"Rows \[2\]"):
        pickle.loads(spilled)["pixels"]


def test_set_reference_checks_the_rows():
    s = B.Set[Image](capacity=3)
    s["defocus"] = np.zeros(3)

    with pytest.raises(ValueError, match="rows"):
        references.set_reference(
            s, ("pixels",), Reference(ArraySource(_rows(4)), range(4))
        )
