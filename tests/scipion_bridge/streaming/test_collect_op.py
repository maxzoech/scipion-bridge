import numpy as np
import pytest
import ray
import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ops import Source, _make_set_collect_accumulator
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter

pytestmark = pytest.mark.usefixtures("ray_cluster")


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


def _ids(s):
    return [int(x) for x in s["id"].flatten()]


@ray.remote
class SetCollector:
    def __init__(self):
        self.sets = []

    def append(self, s):
        self.sets.append(_ids(s))

    def get(self):
        return self.sets


def test_collect_op_validation():
    with pytest.raises(ValueError, match="Collect size n must be positive"):
        Source("input").collect(0)


def test_collect_emits_once_truncated_and_ignores_later_items():
    accumulate, initial_state, flush = _make_set_collect_accumulator(4)
    state = initial_state()

    state, out = accumulate(state, _make_set([1, 2, 3]))
    assert out == []

    state, out = accumulate(state, _make_set([4, 5, 6]))
    assert [_ids(s) for s in out] == [[1, 2, 3, 4]]

    state, out = accumulate(state, _make_set([7, 8, 9, 10]))
    assert out == []

    state, out = flush(state)
    assert out == []


def test_collect_flushes_short_stream():
    accumulate, initial_state, flush = _make_set_collect_accumulator(10)
    state, _ = accumulate(initial_state(), _make_set([1, 2]))
    state, _ = accumulate(state, _make_set([]))

    state, out = flush(state)
    assert [_ids(s) for s in out] == [[1, 2]]

    # Nothing is emitted twice
    state, out = flush(state)
    assert out == []


def test_collect_flush_of_empty_stream_emits_nothing():
    _, initial_state, flush = _make_set_collect_accumulator(3)
    _, out = flush(initial_state())
    assert out == []


def test_collect_rejects_non_sets():
    accumulate, initial_state, _ = _make_set_collect_accumulator(3)
    with pytest.raises(TypeError, match="CollectOp expected an instance of Set"):
        accumulate(initial_state(), [1, 2, 3])


def test_collect_in_ray_pipeline_emits_on_flush():
    # The stream is shorter than n: the result is emitted by the stage's flush,
    # before the sink flushes. Emitting a full collect is tested above.
    collector = SetCollector.remote()
    writer = CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))

    sink_node = Source("items").collect(50).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set([1, 2, 3]))
        pipeline.send(items=_make_set([4, 5, 6]))
        pipeline.send(items=_make_set([7]))

    assert ray.get(collector.get.remote()) == [[1, 2, 3, 4, 5, 6, 7]]
