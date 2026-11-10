"""The function tables shipping map functions to Ray workers once."""

import os
import threading
import time
from typing import Any, Dict, List, Tuple

import cloudpickle
import pytest
import ray

from scipion_bridge.backend.ray import RayBackend
from scipion_bridge.backend.ray.function_table import FunctionTable, TableCache
from scipion_bridge.backend.ray.mailbox import Ref
from scipion_bridge.core.streaming import Pipeline, Source


class Loader:
    """Loads tables from a dict, counting the loads of every reference."""

    def __init__(self, tables: Dict[str, FunctionTable], delay_s: float = 0.0):
        self.tables = tables
        self.delay_s = delay_s
        self.loads: List[str] = []

    def __call__(self, ref: str) -> FunctionTable:
        self.loads.append(ref)
        time.sleep(self.delay_s)
        return self.tables[ref]


def _table(name: str) -> FunctionTable:
    return FunctionTable(
        functions={name: str},
        parameters={},
        profile=None,
        preparers={},
    )


def test_a_table_is_loaded_once():
    load = Loader({"a": _table("a")})
    cache = TableCache(load, size=2)

    tables = [cache.table(Ref("a")) for _ in range(5)]

    assert load.loads == ["a"]
    assert all(table is tables[0] for table in tables)


def test_the_least_recently_used_table_is_evicted():
    load = Loader({name: _table(name) for name in "abc"})
    cache = TableCache(load, size=2)

    cache.table(Ref("a"))
    cache.table(Ref("b"))
    cache.table(Ref("a"))  # b is now the least recently used
    cache.table(Ref("c"))  # evicts b
    cache.table(Ref("a"))
    cache.table(Ref("b"))

    assert load.loads == ["a", "b", "c", "b"]


def test_concurrent_first_lookups_load_a_table_once():
    load = Loader({"a": _table("a")}, delay_s=0.05)
    cache = TableCache(load, size=2)

    threads = [threading.Thread(target=cache.table, args=(Ref("a"),)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert load.loads == ["a"]


class Model:
    def __init__(self) -> None:
        self.weights = list(range(1000))

    def first(self, item: Any) -> Any:
        return item

    def second(self, item: Any) -> Any:
        return item


def test_the_methods_of_one_object_share_it_after_unpickling():
    model = Model()
    table = FunctionTable(
        functions={"0:first": model.first, "1:second": model.second},
        parameters={},
        profile=None,
        preparers={},
    )

    restored = cloudpickle.loads(cloudpickle.dumps(table))

    first, second = restored.functions["0:first"], restored.functions["1:second"]
    assert first.__self__ is second.__self__
    assert first.__self__ is not model


@ray.remote
class PidTracker:
    """Records the processes that unpickled a probe and that ran a call."""

    def __init__(self) -> None:
        self.unpickled: List[int] = []
        self.called: List[int] = []

    def record_unpickle(self, pid: int) -> None:
        self.unpickled.append(pid)

    def record_call(self, pid: int) -> None:
        self.called.append(pid)

    def pids(self) -> Tuple[List[int], List[int]]:
        return self.unpickled, self.called


def _restore_probe(tracker: Any) -> "Probe":
    ray.get(tracker.record_unpickle.remote(os.getpid()))
    return Probe(tracker)


class Probe:
    """Reports every unpickling of itself to a tracker."""

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker

    def __reduce__(self):
        return (_restore_probe, (self.tracker,))


class TrackedModel:
    def __init__(self, tracker: Any) -> None:
        self.probe = Probe(tracker)

    def first(self, item: Any) -> Any:
        ray.get(self.probe.tracker.record_call.remote(os.getpid()))
        return item + 1

    def second(self, item: Any) -> Any:
        ray.get(self.probe.tracker.record_call.remote(os.getpid()))
        return item * 2


@pytest.mark.usefixtures("ray_cluster")
def test_map_functions_are_unpickled_once_per_process():
    tracker = PidTracker.remote()
    model = TrackedModel(tracker)
    sink = Source("x").map(model.first).map(model.second).sink(lambda item: None)

    with Pipeline.from_sink(sink, backend=RayBackend(init_ray=False)) as pipeline:
        for item in range(20):
            pipeline.send(x=item)

    unpickled, called = ray.get(tracker.pids.remote())
    assert len(called) == 40
    # Every process that ran calls unpickled the model once, for both maps.
    assert sorted(unpickled) == sorted(set(called))
