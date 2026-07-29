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

    source = Source("particles", dtype=B.Set[Particle])
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


def test_fluent_pipeline():
    received = []

    source = Source("numbers")
    sink_node = (
        source
        .map(lambda x: x + 10)
        .map(lambda x: x * 2)
        .sink(lambda x: received.append(x))
    )

    stream = Pipeline.from_sink(sink_node)
    stream.send(numbers=5)   # 5 -> 15 -> 30
    stream.send(numbers=1)   # 1 -> 11 -> 22

    assert received == [30, 22]


def test_disallow_lists():
    source = Source("data")
    stream = Pipeline.from_sink(source)

    with pytest.raises(TypeError, match="Python lists are not supported"):
        stream.send(data=[1, 2, 3])


def test_modify_set_to_struct_with_latent():
    received = []

    def _encode_particles(particle_set: B.Set[Particle]) -> ParticleEmbeddings:
        latent = np.full(128, 0.42, dtype=np.float32)
        return ParticleEmbeddings(particles=particle_set, latent_code=latent)

    source = Source("particles", dtype=B.Set[Particle])
    sink_node = source.map(_encode_particles).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    batch_size = 3
    particle_set = B.Set[Particle](capacity=batch_size)
    for idx in range(batch_size):
        pixels = np.ones([256, 256], dtype=np.float32) * idx
        particle_set[idx] = Particle(pixels=pixels, metadata=Metadata(foo=idx))

    stream.send(particles=particle_set)

    assert len(received) == 1
    result = received[0]
    assert isinstance(result, ParticleEmbeddings)
    assert isinstance(result.particles, B.Set)
    assert len(result.particles) == batch_size
    assert result.latent_code.shape == (128,)
    assert np.allclose(result.latent_code, 0.42)


if __name__ == "__main__":
    test_basic_stream()
    test_fluent_pipeline()
    test_disallow_lists()
    test_modify_set_to_struct_with_latent()
