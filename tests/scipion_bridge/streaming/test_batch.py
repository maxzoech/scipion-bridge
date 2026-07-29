import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.streaming.ops import Source, BatchOp
from scipion_bridge.core.streaming.pipeline import Pipeline


class Metadata(B.Struct):
    foo: int


class Particle(B.Struct):
    pixels: B.Array[np.float32, 256, 256]
    metadata: Metadata


def test_batch_single_elements_into_set():
    received = []

    # Pipeline: Source -> batch(3) -> sink
    source = Source("items")
    sink_node = source.batch(3).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1))
    p2 = Particle(pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2))

    stream.send(items=p1)
    stream.send(items=p2)
    assert len(received) == 0  # Not enough items yet

    p3 = Particle(pixels=np.zeros([256, 256], dtype=np.float32) + 3.0, metadata=Metadata(foo=3))
    stream.send(items=p3)

    assert len(received) == 1
    batch_result = received[0]
    assert isinstance(batch_result, B.Set)
    assert len(batch_result) == 3
    assert batch_result[0].metadata.foo == 1
    assert batch_result[1].metadata.foo == 2
    assert batch_result[2].metadata.foo == 3


def test_batch_rechunk_large_set_into_small_sets():
    received = []

    # Pipeline: Source -> batch(6) -> batch(2) -> sink
    source = Source("items")
    sink_node = source.batch(6).batch(2).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    # Emit 6 single elements
    for i in range(1, 7):
        p = Particle(pixels=np.zeros([256, 256], dtype=np.float32) + i, metadata=Metadata(foo=i))
        stream.send(items=p)

    # Should have split the 6-element set into 3 smaller sets of capacity 2
    assert len(received) == 3

    for batch_idx, batch in enumerate(received):
        assert isinstance(batch, B.Set)
        assert len(batch) == 2
        assert batch[0].metadata.foo == batch_idx * 2 + 1
        assert batch[1].metadata.foo == batch_idx * 2 + 2
