import numpy as np
import scipion_bridge as B
from scipion_bridge.core.streaming.stream import Stream


class Metadata(B.Struct):
    foo: int


class Particle(B.Struct):

    pixels: B.Struct[np.float32, [256, 256]]
    metadata: Metadata


def test_basic_stream():

    batch_size = 5

    stream = Stream()

    for batch_idx in range(10):

        particle_set = B.Set[Particle](capacity=batch_size)

        for element_idx in range(batch_size):
            pixels = np.zeros([256, 256], dtype=np.float32) + element_idx
            particle_set[element_idx] = Particle(
                pixels=pixels, metadata=Metadata(foo=batch_idx)
            )

        stream.send(particle_set)


if __name__ == "__main__":
    test_basic_stream()
