import numpy as np

from ..core.struct import Struct, Array, Set

class Particle(Struct):
    pass
    # pixels: Array[np.float32] = Array(shape=(128, 128)) # Hard code this for now, implement a better type system where this is dynamic.


class FlexParticle(Struct):
    pass
    # pixels: Array[float] = Array(shape=(128, 128))
    # embeddings: Array[float] = Array(shape=(None,))


class Class2D(Struct):
    pass

    # particles: Set[Particle]
    # label: int