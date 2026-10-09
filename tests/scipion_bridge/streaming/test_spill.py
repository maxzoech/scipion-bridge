"""Tests for the spill stores of the streaming pipeline (in-process)."""

import numpy as np

import scipion_bridge as B
from scipion_bridge.core.streaming.spill import PickleSpillStore, pickle_spill_store


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
