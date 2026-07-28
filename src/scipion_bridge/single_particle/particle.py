import numpy as np

from ..core.struct import Struct, Array


class Particle(Struct):
    pixels: Array[np.float32, [256, 256]]
