import numpy as np

from ..core.struct import Struct, Array


class Particle(Struct):
    pixels: Array[np.float32, [128, 128]] # Hard code this for now, implement a better type system where this is dynamic.


class FlexParticle(Struct):
    pixels: Array[float, [128, 128]]
    embeddings: Array[float, [768]]