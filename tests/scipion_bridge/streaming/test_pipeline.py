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


def test_disallow_lists():
    source = Source("data")
    stream = Pipeline.from_sink(source)

    with pytest.raises(TypeError, match="Python lists are not supported"):
        stream.send(data=[1, 2, 3])


if __name__ == "__main__":
    test_basic_stream()
    test_disallow_lists()
