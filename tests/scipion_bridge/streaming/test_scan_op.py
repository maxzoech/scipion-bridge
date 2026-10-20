import operator

import numpy as np
import pytest
import ray
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ir import IRAccumulate, Keyed
from scipion_bridge.core.streaming.keyed import KeyedAccumulator
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source, _make_scan_accumulator
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, x):
        self.items.append(x)

    def get(self):
        return self.items


def _run(accumulate, state, items):
    emitted = []
    for item in items:
        state, out = accumulate(state, item)
        emitted.extend(out)
    return state, emitted


def test_scan_op_validation():
    with pytest.raises(TypeError, match="scan\\(\\) expects a function"):
        Source("input").scan(1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="expects initial to be a function"):
        Source("input").scan(operator.add, initial=[])  # type: ignore[arg-type]


def test_scan_lowers_to_accumulator_without_flush():
    (ir,) = lower([Source("input").scan(operator.add)])
    assert isinstance(ir, IRAccumulate)
    assert ir.name == "scan(add)"
    assert ir.flush_fn is None


def test_scan_without_initial_seeds_with_first_item():
    accumulate, initial_state = _make_scan_accumulator(operator.add, None)
    _, emitted = _run(accumulate, initial_state(), [1, 2, 3])
    assert emitted == [1, 3, 6]


def test_scan_with_initial_calls_factory_per_state():
    def append(acc, item):
        acc.append(item)
        return acc

    accumulate, initial_state = _make_scan_accumulator(append, list)
    first, emitted = _run(accumulate, initial_state(), [1, 2])
    second, _ = _run(accumulate, initial_state(), [10])

    assert emitted[-1] == [1, 2]
    assert first.acc == [1, 2]
    assert second.acc == [10]


def test_scan_keeps_none_as_state_and_item():
    accumulate, initial_state = _make_scan_accumulator(
        lambda acc, item: (acc, item),
        None,
    )
    _, emitted = _run(accumulate, initial_state(), [None, None])
    assert emitted == [None, (None, None)]


def test_scan_running_weighted_mean():
    # Every item is the average image of a chunk, with its particle count.
    def merge(acc, item):
        image, count = acc
        new_image, new_count = item
        total = count + new_count
        return ((image * count + new_image * new_count) / total, total)

    accumulate, initial_state = _make_scan_accumulator(merge, None)
    chunks = [(np.full((2, 2), 1.0), 1), (np.full((2, 2), 4.0), 3)]
    _, emitted = _run(accumulate, initial_state(), chunks)

    image, count = emitted[-1]
    assert count == 4
    np.testing.assert_allclose(image, np.full((2, 2), 3.25))


def test_scan_keeps_a_state_per_key():
    accumulate, initial_state = _make_scan_accumulator(operator.add, None)
    keyed = KeyedAccumulator(accumulate, initial_state, None)
    items = [Keyed("a", 1), Keyed("b", 10), Keyed("a", 2), Keyed("b", 20)]
    _, emitted = _run(keyed.accumulate, keyed.initial_state(), items)

    assert emitted == [Keyed("a", 1), Keyed("b", 10), Keyed("a", 3), Keyed("b", 30)]


@pytest.mark.usefixtures("ray_cluster")
def test_scan_in_ray_group_by_emits_running_state_per_key():
    collector = Collector.remote()
    writer = CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))

    sink_node = (
        Source("items")
        .group_by(0, lambda s: s.map(operator.itemgetter(1)).scan(operator.add))
        .unkey()
        .write_to(writer)
    )

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        for item in [("a", 1), ("b", 10), ("a", 2), ("b", 20), ("a", 3)]:
            pipeline.send(items=item)

    by_key: dict = {}
    for keyed in ray.get(collector.get.remote()):
        by_key.setdefault(keyed.key, []).append(keyed.value)
    assert by_key == {"a": [1, 3, 6], "b": [10, 30]}
