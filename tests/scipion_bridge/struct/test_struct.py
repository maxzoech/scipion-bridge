import numpy as np
import pytest
import scipion_bridge as B

from typing import Tuple

class SimpleStruct(B.Struct):
    val_int: int
    val_float: float
    val_bool: bool


class Foo(B.Struct):
    a: np.complex64
    b: float


class CTF(B.Struct):
    voltage_kv: float
    amplitude_contrast: float
    foo: Foo


class Particle(B.Struct):
    pixels: B.Array[float] = B.Array(shape=(None, None))
    ctf: CTF
    foo_2: Foo

class Class2D(B.Struct):

    particle: Particle
    average: B.Array[float] = B.Array(shape=(None, None))

    # def __init__(self, img_size: Tuple[int, int], **fields) -> None:
    #     super().__init__(**fields)

    #     self.particle.pixels = B.Array(shape=img_size)
    #     self.average = B.Array(shape=img_size)

def test_simple_struct():

    struct = SimpleStruct()
    struct.schema.print_tree()

def test_nested_struct():

    struct = CTF()
    struct.schema.print_tree()

def test_dynamic_init():

    class2D = Class2D(
        particle = Particle(
            pixels=B.Array(shape=(128, 128))
        ),
        average=B.Array(shape=(128, 128))
    )

    class2D.schema.print_tree()

if __name__ == "__main__":
    test_dynamic_init()