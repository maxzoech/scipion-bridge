import numpy as np

from ..core.struct import Struct, Array, Set


class CTF(Struct):
    defocus_u: float
    defocus_v: float
    defocus_angle: float
    phase_shift: float
    resolution: float
    fit_quality: float


class Coordinate(Struct):
    x: float
    y: float


class Particle(Struct):
    pixels: Array[np.float32] = Array(shape=(None, None))
    ctf: CTF
    coordinate: Coordinate
    sampling_rate: float


class FlexParticle(Particle):
    embeddings: Array[float] = Array(shape=(None,))


class Class2D(Struct):
    particles: Set[Particle]
    class_id: int
    representative: Particle
