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


class Particle(B.Struct):
    pixels: B.Array[float, 224, 224]
    ctf: CTF
    foo_2: Foo


def test_struct_schema_classmethod():
    schema = SimpleStruct.schema()
    assert schema.entries() == {"val_int", "val_float", "val_bool"}


def test_struct_instance_scalars():
    s = SimpleStruct()
    s.val_int = 42
    s.val_float = 3.14
    s.val_bool = True

    assert s.val_int == 42
    assert isinstance(s.val_int, (int, np.integer))
    assert pytest.approx(s.val_float) == 3.14
    assert s.val_bool is True or s.val_bool == 1


def test_struct_instance_arrays():
    particle = Particle()
    data = np.random.uniform(size=[224, 224])
    particle.pixels = data

    assert particle.pixels.shape == (224, 224)
    assert np.allclose(particle.pixels, data)


def test_struct_instance_nested():
    particle = Particle()
    ctf = CTF()
    ctf.voltage_kv = 300.0
    ctf.amplitude_contrast = 0.1

    foo = Foo()
    foo.a = np.complex64(1 + 2j)
    foo.b = 2.5
    ctf.foo = foo

    particle.ctf = ctf

    assert particle.ctf.voltage_kv == 300.0
    assert particle.ctf.amplitude_contrast == 0.1
    assert particle.ctf.foo.a == np.complex64(1 + 2j)
    assert particle.ctf.foo.b == 2.5


def test_non_schema_attribute_assignment():
    s = SimpleStruct()
    s.custom_field = "test_value"
    assert s.custom_field == "test_value"


def test_init():
    
    pixels = np.random.uniform(size=[64, 64])
    particle = Particle(
        pixels=pixels,
        ctf=CTF(
            voltage_kv=300.0,
            amplitude_contrast=0.0,
            foo=Foo(
                a=np.complex64(1 + 2j),
                b=42.0,
            )
        ),
        foo_2=Foo(
            a=np.complex64(1 + 2j),
            b=12.0,
        )
    )

    assert np.allclose(particle.pixels, pixels)

    assert particle.ctf.voltage_kv == 300.0
    assert particle.ctf.amplitude_contrast == 0.0
    assert particle.ctf.foo.a == np.complex64(1 + 2j)
    assert particle.ctf.foo.b == 42.0

    assert particle.foo_2.a == np.complex64(1 + 2j)
    assert particle.foo_2.b == 12.0


if __name__ == "__main__":
    test_init()
