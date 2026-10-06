import numpy as np
import ray
import scipion_bridge as B
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import SinkWriter


class Metadata(B.Struct):
    foo: int


class Particle(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(256, 256))
    metadata: Metadata


class ParticleEmbeddings(B.Struct):
    particles: B.Set[Particle] = B.Set[Particle](capacity=10)
    latent_code: B.Array[np.float32] = B.Array(shape=(128,))


@ray.remote
class PipelineCollector:
    def __init__(self):
        self.items = []
        self.finalized = 0

    def append(self, x):
        self.items.append(x)

    def mark_finalized(self):
        self.finalized += 1

    def get_items(self):
        return self.items

    def get_finalized(self):
        return self.finalized


def test_basic_stream():
    collector = PipelineCollector.remote()
    batch_size = 5

    source = Source("particles")
    sink_node = source.sink(lambda x: ray.get(collector.append.remote(x)))

    stream = Pipeline.from_sink(sink_node)

    for batch_idx in range(2):
        particle_set = B.Set[Particle](capacity=batch_size)
        for element_idx in range(batch_size):
            pixels = np.zeros([256, 256], dtype=np.float32) + element_idx
            particle_set[element_idx] = Particle(
                pixels=pixels, metadata=Metadata(foo=batch_idx)
            )

        stream.send(particles=particle_set)

    stream.flush()
    received = ray.get(collector.get_items.remote())
    assert len(received) == 2


def test_pipeline_context_manager_autoflush():
    collector = PipelineCollector.remote()

    class FlushingSinkWriter(SinkWriter):
        async def write(self, item):
            pass

        async def finalize(self):
            ray.get(collector.mark_finalized.remote())

    source = Source("items")
    sink_node = source.write_to(FlushingSinkWriter())

    p = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=42)
    )

    with Pipeline.from_sink(sink_node) as pipe:
        pipe.send(items=B.Set[Particle]([p]))
        assert ray.get(collector.get_finalized.remote()) == 0

    # Upon exiting context manager, auto-flush happens
    assert ray.get(collector.get_finalized.remote()) == 1


if __name__ == "__main__":
    test_basic_stream()
    test_pipeline_context_manager_autoflush()
