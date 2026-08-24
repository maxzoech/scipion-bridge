import numpy as np
import pytest
import scipion_bridge as B


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


# class Particle(B.Struct):
#     pixels: B.Array[float] = B.Array(shape=(None, None))
#     ctf: CTF
#     foo_2: Foo


def test_simple_struct():

    struct = SimpleStruct()
    struct.schema.print_tree()

def test_nested_struct():

    struct = CTF()
    struct.schema.print_tree()

if __name__ == "__main__":
    test_nested_struct()