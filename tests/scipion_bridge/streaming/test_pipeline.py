import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.streaming.ops import Source, Sink
from scipion_bridge.core.streaming.pipeline import Pipeline


class Metadata(B.Struct):
    foo: int


class Particle(B.Struct):
    pixels: B.Array[np.float32, 256, 256]
    metadata: Metadata


class ParticleEmbeddings(B.Struct):
    particles: B.Set[Particle, 10]
    latent_code: B.Array[np.float32, 128]


def test_basic_stream():
    batch_size = 5
    received = []

    source = Source("particles")
    sink_node = source.sink(lambda x: received.append(x))

    stream = Pipeline.from_sink(sink_node)

    for batch_idx in range(2):
        particle_set = B.Set[Particle](capacity=batch_size)
        for element_idx in range(batch_size):
            pixels = np.zeros([256, 256], dtype=np.float32) + element_idx
            particle_set[element_idx] = Particle(
                pixels=pixels, metadata=Metadata(foo=batch_idx)
            )

        stream.send(particles=particle_set)

    assert len(received) == 2


def test_pipeline_context_manager_autoflush():
    received = []

    source = Source("items")
    sink_node = source.chunk(10).sink(lambda x: received.append(x))

    p = Particle(pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=42))

    with Pipeline.from_sink(sink_node) as pipe:
        pipe.send(items=B.Set[Particle]([p]))
        assert len(received) == 0  # 1 element buffered for chunk_size=10

    # Upon exiting context manager, auto-flush happens
    assert len(received) == 1
    assert len(received[0]) == 1
    assert received[0][0].metadata.foo == 42


if __name__ == "__main__":
    test_basic_stream()
    test_pipeline_context_manager_autoflush()
