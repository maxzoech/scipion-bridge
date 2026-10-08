import numpy as np
import pytest
import ray
import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ir import IRAccumulate
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


@ray.remote
class ChunkCollector:
    def __init__(self):
        self.chunks = []

    def append(self, s):
        ids = [int(x) for x in s["id"].flatten()]
        self.chunks.append((len(s), ids))

    def get(self):
        return self.chunks


def test_chunk_op_validation():
    source = Source("input")
    with pytest.raises(ValueError, match="Chunk size n must be positive"):
        source.chunk(0)

    with pytest.raises(ValueError, match="Chunk size n must be positive"):
        source.chunk(-5)


def _chunks(op, batches):
    """Run the accumulator ``op`` lowers to in-process, as its stage does on flush."""
    (sink,) = lower([op.sink(print)])
    (accumulate,) = sink.upstream
    assert isinstance(accumulate, IRAccumulate)
    assert accumulate.flush_fn is not None

    state = accumulate.initial_state_fn()
    emitted = []
    for batch in batches:
        state, emissions = accumulate.accumulate_fn(state, batch)
        emitted.extend(emissions)
    _, emissions = accumulate.flush_fn(state)
    return [(len(s), [int(x) for x in s["id"].flatten()]) for s in emitted + emissions]


def test_chunk_sub_batches_drop_last_false():
    chunks = _chunks(
        Source("items").chunk(5, drop_last=False),
        [
            _make_set([1, 2, 3]),
            _make_set([4, 5, 6]),
            _make_set([7, 8, 9]),
            _make_set([10]),
            _make_set([11, 12]),
        ],
    )

    assert chunks == [
        (5, [1, 2, 3, 4, 5]),
        (5, [6, 7, 8, 9, 10]),
        (2, [11, 12]),
    ]


def test_chunk_sub_batches_drop_last_true():
    chunks = _chunks(
        Source("items").chunk(5, drop_last=True),
        [_make_set([1, 2, 3]), _make_set([4, 5, 6]), _make_set([7, 8])],
    )

    assert chunks == [(5, [1, 2, 3, 4, 5])]


def test_chunk_oversized_batch_splitting():
    chunks = _chunks(
        Source("items").chunk(4, drop_last=False),
        [_make_set(list(range(1, 11)))],
    )

    assert chunks == [(4, [1, 2, 3, 4]), (4, [5, 6, 7, 8]), (2, [9, 10])]


def test_chunk_oversized_batch_splitting_drop_last():
    chunks = _chunks(
        Source("items").chunk(4, drop_last=True),
        [_make_set(list(range(1, 11)))],
    )

    assert chunks == [(4, [1, 2, 3, 4]), (4, [5, 6, 7, 8])]


def test_chunk_non_set_raises():
    with pytest.raises(TypeError, match="ChunkOp expected an instance of Set"):
        _chunks(Source("items").chunk(4), ["not_a_set"])


@pytest.mark.usefixtures("ray_cluster")
def test_chunk_in_multi_stage_pipeline():
    collector = ChunkCollector.remote()
    writer = CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))

    def double_ids(s: B.Set[Item]) -> B.Set[Item]:
        doubled = _make_set([int(x) * 2 for x in s["id"].flatten()])
        return doubled

    source = Source("items")
    sink_node = source.map_batch(double_ids).chunk(3, drop_last=False).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set([1, 2]))
        pipeline.send(items=_make_set([3, 4, 5]))

    chunks = ray.get(collector.get.remote())
    # Incoming doubled: [2, 4], then [6, 8, 10]
    # chunk(3) yields: [2, 4, 6] (len 3) and [8, 10] (len 2)
    assert len(chunks) == 2
    assert chunks[0] == (3, [2, 4, 6])
    assert chunks[1] == (2, [8, 10])
