"""Checkpoints and taps on Ray: items kept in the run store while they pass.

What the store keeps and when the backend leaves checkpoints out is tested
in-process in test_checkpoints.py.
"""

import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray import RayBackend, RayCompiledPipeline
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    FileRunStore,
    Pipeline,
    Source,
)
from scipion_bridge.core.struct import identity
from scipion_bridge.core.struct.id_column import IdColumn

pytestmark = pytest.mark.usefixtures("ray_cluster")

F = b"F" * 16


class Value(B.Struct):
    value: float


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def get(self):
        return self.items


def _collecting(collector):
    return CallbackSinkWriter(lambda item: ray.get(collector.append.remote(item)))


def _values(start, n):
    s = B.Set[Value](capacity=n)
    s["value"] = np.arange(start, start + n, dtype=np.float64)
    identity.set_ids(s, IdColumn.of(F, range(start, start + n)))
    return s


def _scaled(values):
    out = B.Set[Value](capacity=len(values))
    out["value"] = np.asarray(values["value"]) * 2
    identity.set_ids(out, identity.element_ids(values))
    return out


def _run(sink_node, inputs, **backend_kwargs):
    pipeline = Pipeline.from_sink(
        sink_node,
        backend=RayBackend(init_ray=False, **backend_kwargs),
    )
    try:
        for value in inputs:
            pipeline.send(x=value)
        pipeline.flush()
        compiled = pipeline.compiled
        assert isinstance(compiled, RayCompiledPipeline)
        return compiled.received_ids(), pipeline.stats()
    finally:
        pipeline.close()


def test_checkpoints_and_taps_keep_the_items_passing_through(tmp_path):
    store = FileRunStore(tmp_path / "run")
    collector = Collector.remote()
    inputs = [_values(3 * k, 3) for k in range(4)]

    sink_ids, _ = _run(
        Source("x")
        .map(_scaled)
        .checkpoint("scaled")
        .tap("again")
        .write_to(_collecting(collector)),
        inputs,
        run_store=store,
    )

    (received,) = sink_ids.values()

    checkpoints = store.read_checkpoints()
    assert {
        label: (c.replay, c.items, c.complete) for label, c in checkpoints.items()
    } == {
        "2:checkpoint(scaled)": (True, 4, True),
        "3:tap(again)": (False, 4, True),
    }
    # Kept in the order they passed, with the ids the sink received.
    commits = store.read_commits("2:checkpoint(scaled)")
    assert [c.item_id for c in commits] == received
    kept = [store.read_item("2:checkpoint(scaled)", lid) for lid in received]
    assert [list(k["value"]) for k in kept] == [
        list(np.arange(3 * k, 3 * k + 3) * 2.0) for k in range(4)
    ]
    assert len(ray.get(collector.get.remote())) == 4


def test_without_a_run_store_checkpoints_are_left_out():
    collector = Collector.remote()

    _, stats = _run(
        Source("x").map(_scaled).checkpoint().write_to(_collecting(collector)),
        [_values(0, 2)],
    )

    # The stages keep their labels; the checkpoint did nothing.
    assert list(stats) == [
        "0:source(x)",
        "1:map(_scaled)",
        "2:checkpoint",
        "3:sink(CallbackSinkWriter)",
    ]
    assert stats["2:checkpoint"].items_in == 0
    assert stats["3:sink(CallbackSinkWriter)"].items_in == 1
    assert len(ray.get(collector.get.remote())) == 1
