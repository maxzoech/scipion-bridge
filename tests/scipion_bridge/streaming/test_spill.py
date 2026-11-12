"""Tests for the spill stores of the streaming pipeline (in-process)."""

import tempfile
from pathlib import Path

import numpy as np

import scipion_bridge as B
from scipion_bridge.core.streaming import PickleSpillStore, pickle_spill_store


class SpillItem(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[SpillItem](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


def test_pickle_store_round_trips_items(tmp_path):
    store = PickleSpillStore(tmp_path / "stage")

    plain = store.put({"a": [1, 2, 3]})
    items = store.put(_make_set([4, 5]))

    assert store.get(plain) == {"a": [1, 2, 3]}
    assert [int(x) for x in store.get(items)["id"].flatten()] == [4, 5]


def test_pickle_store_discard_removes_the_file(tmp_path):
    store = PickleSpillStore(tmp_path)
    handle = store.put(1)

    store.discard(handle)

    assert list(tmp_path.iterdir()) == []


def test_pickle_store_creates_its_directory_on_the_first_item(tmp_path):
    store = PickleSpillStore(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()

    store.put(1)

    assert (tmp_path / "missing").is_dir()


def test_factory_creates_a_directory_per_stage(tmp_path):
    factory = pickle_spill_store(tmp_path)

    factory("0:source(x)").put(1)
    factory("1:map(<lambda>)").put(2)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "0_source_x_",
        "1_map__lambda__",
    ]


def test_close_removes_the_directory(tmp_path):
    store = PickleSpillStore(tmp_path / "stage")
    store.put(1)

    store.close()

    assert list(tmp_path.iterdir()) == []


def test_closing_a_store_that_never_spilled_creates_nothing(tmp_path):
    PickleSpillStore(tmp_path / "stage").close()

    assert list(tmp_path.iterdir()) == []


def test_a_relative_directory_lies_in_the_temporary_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    store = PickleSpillStore(Path("pipeline") / "stage")

    store.put(1)

    assert store.path == tmp_path / "pipeline" / "stage"
    assert len(list(store.path.iterdir())) == 1


def test_the_last_store_of_a_pipeline_removes_its_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    factory = pickle_spill_store(Path("pipeline"))
    first, second = factory("0:source(x)"), factory("1:sink(y)")
    first.put(1)
    second.put(2)

    first.close()
    assert [path.name for path in tmp_path.iterdir()] == ["pipeline"]

    second.close()
    assert list(tmp_path.iterdir()) == []


def test_an_absolute_directory_keeps_its_parent(tmp_path):
    store = pickle_spill_store(tmp_path)("0:source(x)")
    store.put(1)

    store.close()

    assert tmp_path.is_dir()
