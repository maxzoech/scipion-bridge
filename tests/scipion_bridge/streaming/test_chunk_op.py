import numpy as np
import pytest
import ray
import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
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


def test_chunk_sub_batches_drop_last_false():
    collector = ChunkCollector.remote()
    writer = CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))

    source = Source("items")
    sink_node = source.chunk(5, drop_last=False).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set([1, 2, 3]))
        pipeline.send(items=_make_set([4, 5, 6]))
        pipeline.send(items=_make_set([7, 8, 9]))
        pipeline.send(items=_make_set([10]))
        pipeline.send(items=_make_set([11, 12]))

    chunks = ray.get(collector.get.remote())
    assert len(chunks) == 3
    assert chunks[0] == (5, [1, 2, 3, 4, 5])
    assert chunks[1] == (5, [6, 7, 8, 9, 10])
    assert chunks[2] == (2, [11, 12])


def test_chunk_sub_batches_drop_last_true():
    collector = ChunkCollector.remote()
    writer = CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))

    source = Source("items")
    sink_node = source.chunk(5, drop_last=True).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set([1, 2, 3]))
        pipeline.send(items=_make_set([4, 5, 6]))
        pipeline.send(items=_make_set([7, 8]))

    chunks = ray.get(collector.get.remote())
    assert len(chunks) == 1
    assert chunks[0] == (5, [1, 2, 3, 4, 5])


def test_chunk_oversized_batch_splitting():
    collector = ChunkCollector.remote()
    writer = CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))

    source = Source("items")
    sink_node = source.chunk(4, drop_last=False).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set(list(range(1, 11))))

    chunks = ray.get(collector.get.remote())
    assert len(chunks) == 3
    assert chunks[0] == (4, [1, 2, 3, 4])
    assert chunks[1] == (4, [5, 6, 7, 8])
    assert chunks[2] == (2, [9, 10])


def test_chunk_oversized_batch_splitting_drop_last():
    collector = ChunkCollector.remote()
    writer = CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))

    source = Source("items")
    sink_node = source.chunk(4, drop_last=True).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set(list(range(1, 11))))

    chunks = ray.get(collector.get.remote())
    assert len(chunks) == 2
    assert chunks[0] == (4, [1, 2, 3, 4])
    assert chunks[1] == (4, [5, 6, 7, 8])


def test_chunk_non_set_raises():
    writer = CallbackSinkWriter(lambda s: None)
    source = Source("items")
    sink_node = source.chunk(4).write_to(writer)

    backend = RayBackend(init_ray=False)
    with pytest.raises(Exception) as exc_info:
        with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
            pipeline.send(items="not_a_set")

    assert "ChunkOp expected an instance of Set" in str(exc_info.value)


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
