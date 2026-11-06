import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray import RayBackend
from scipion_bridge.core.streaming import IRAccumulate, Tagged
from scipion_bridge.core.streaming import lower
from scipion_bridge.core.streaming.ops import (
    MergeOp,
    Source,
    _merge,
    _merge_initial_state,
)
from scipion_bridge.core.streaming import Pipeline
from scipion_bridge.core.streaming import CallbackSinkWriter


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


def _ids(s):
    return [int(x) for x in s["id"].flatten()]


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, x):
        self.items.append(x)

    def get(self):
        return self.items


def _collecting_writer(collector):
    return CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))


def test_merge_emits_the_items_of_both_ports():
    state = _merge_initial_state()
    state, first = _merge(state, Tagged(port=0, item="a"))
    state, second = _merge(state, Tagged(port=1, item="b"))
    assert first == ["a"]
    assert second == ["b"]


def test_merge_keeps_none_items():
    _, emitted = _merge(_merge_initial_state(), Tagged(port=1, item=None))
    assert emitted == [None]


def test_merge_unknown_port_raises():
    with pytest.raises(ValueError, match="two inputs"):
        _merge(_merge_initial_state(), Tagged(port=2, item="x"))


def test_merge_with_itself_raises():
    source = Source("input")
    with pytest.raises(ValueError, match="two different streams"):
        source.merge(source)


def test_merge_lowering():
    source = Source("input")
    node = source.map(str).merge(source.map(repr))
    assert isinstance(node, MergeOp)

    (ir,) = lower([node])
    assert isinstance(ir, IRAccumulate)
    assert ir.name == "merge"
    assert ir.tag_inputs
    assert ir.flush_fn is None
    assert len(ir.upstream) == 2


@pytest.mark.usefixtures("ray_cluster")
def test_merge_joins_two_branches_of_one_stream():
    collector = Collector.remote()

    items = Source("items")
    small = items.map(lambda x: [x] if x < 3 else []).flatten()
    large = items.map(lambda x: [10 * x] if x >= 3 else []).flatten()
    sink_node = small.merge(large).write_to(_collecting_writer(collector))

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        for item in range(6):
            pipeline.send(items=item)

    assert sorted(ray.get(collector.get.remote())) == [0, 1, 2, 30, 40, 50]


@pytest.mark.usefixtures("ray_cluster")
def test_merge_forwards_the_remainder_of_a_batching_branch():
    """A chunk on one branch emits its remainder at the stream end, after the
    other branch ended: the merge still forwards it."""
    collector = Collector.remote()

    sets = Source("sets")
    passed = sets.map(lambda s: ("passed", _ids(s)))
    batched = sets.chunk(4).map(lambda s: ("batched", _ids(s)))
    sink_node = passed.merge(batched).write_to(_collecting_writer(collector))

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        for ids in ([0, 1, 2], [3, 4], [5]):
            pipeline.send(sets=_make_set(ids))

    emitted = ray.get(collector.get.remote())
    passed_ids = sorted(i for kind, ids in emitted if kind == "passed" for i in ids)
    batches = sorted(ids for kind, ids in emitted if kind == "batched")
    assert passed_ids == list(range(6))
    assert batches == [[0, 1, 2, 3], [4, 5]]
