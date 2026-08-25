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
    H: B.Dim = B.Dim()
    W: B.Dim = B.Dim()

    pixels: B.Array[float] = B.Array(shape=(H, W))
    ctf: CTF
    foo_2: Foo

class Class2D(B.Struct):

    H: B.Dim = B.Dim()
    W: B.Dim = B.Dim()

    particle: Particle = Particle(H=H, W=W)
    average: B.Array[float] = B.Array(shape=(H, W))


def test_simple_struct():
    struct = SimpleStruct()
    assert struct.is_static()
    assert struct.schema.is_static
    assert set(struct.schema.fields.keys()) == {"val_int", "val_float", "val_bool"}
    assert struct.schema.fields["val_int"].shape == ()
    struct.schema.print_tree()


def test_nested_struct():
    struct = CTF()
    assert struct.is_static()
    assert struct.schema.is_static
    assert set(struct.schema.fields.keys()) == {"voltage_kv", "amplitude_contrast", "foo"}
    assert struct.schema.fields["foo"].children is not None
    assert set(struct.schema.fields["foo"].children.fields.keys()) == {"a", "b"}
    struct.schema.print_tree()


def test_dynamic_init():
    class2D = Class2D(H=128, W=128)
    assert class2D.is_static()
    assert class2D.schema.is_static
    assert class2D.H == 128
    assert class2D.W == 128

    # Check average array shape
    assert class2D.schema.fields["average"].shape == (128, 128)
    assert class2D.schema.fields["average"].is_static

    # Check nested particle struct and its pixels array shape
    particle_entry = class2D.schema.fields["particle"]
    assert particle_entry.children is not None
    assert particle_entry.children.fields["pixels"].shape == (128, 128)
    assert particle_entry.children.fields["pixels"].is_static
    assert particle_entry.is_static

    class2D.schema.print_tree()


def test_unhydrated_dynamic_init():
    class2D = Class2D()
    assert not class2D.is_static()
    assert not class2D.schema.is_static
    assert class2D.schema.fields["average"].shape == (None, None)
    assert not class2D.schema.fields["average"].is_static

    particle_entry = class2D.schema.fields["particle"]
    assert particle_entry.children is not None
    assert particle_entry.children.fields["pixels"].shape == (None, None)
    assert not particle_entry.children.fields["pixels"].is_static


def test_instance_isolation():
    c1 = Class2D(H=64, W=64)
    c2 = Class2D(H=256, W=256)

    assert c1.schema.fields["average"].shape == (64, 64)
    assert c1.schema.fields["particle"].children.fields["pixels"].shape == (64, 64)

    assert c2.schema.fields["average"].shape == (256, 256)
    assert c2.schema.fields["particle"].children.fields["pixels"].shape == (256, 256)


def test_struct_with_default_dim_value():
    class FixedBox(B.Struct):
        size: B.Dim = B.Dim(64)
        pixels: B.Array[float] = B.Array(shape=(size, size))

    # Default uses 64
    fb_default = FixedBox()
    assert fb_default.is_static()
    assert fb_default.schema.fields["pixels"].shape == (64, 64)

    # Override with 128
    fb_override = FixedBox(size=128)
    assert fb_override.is_static()
    assert fb_override.schema.fields["pixels"].shape == (128, 128)


def test_partial_hydration():
    class Volume(B.Struct):
        D: B.Dim = B.Dim()
        H: B.Dim = B.Dim()
        W: B.Dim = B.Dim()
        data: B.Array[float] = B.Array(shape=(D, H, W))

    v = Volume(D=32)
    assert not v.is_static()
    assert v.schema.fields["data"].shape == (32, None, None)
    assert not v.schema.fields["data"].is_static


def test_dimension_name_collision_scoping():
    class InnerParticle(B.Struct):
        H: B.Dim = B.Dim(64)
        pixels: B.Array[float] = B.Array(shape=(H, H))

    class OuterMicrograph(B.Struct):
        H: B.Dim = B.Dim()
        # InnerParticle uses its own default H=64; not bound to OuterMicrograph.H
        particle: InnerParticle = InnerParticle()
        image: B.Array[float] = B.Array(shape=(H, H))

    micrograph = OuterMicrograph(H=4096)

    # Outer micrograph image must be 4096 x 4096
    assert micrograph.schema.fields["image"].shape == (4096, 4096)

    # Inner particle pixels MUST remain 64 x 64 (not polluted by OuterMicrograph.H=4096)
    particle_entry = micrograph.schema.fields["particle"]
    assert particle_entry.children.fields["pixels"].shape == (64, 64)


def test_multi_level_alias_chaining():
    class Level1Particle(B.Struct):
        H: B.Dim = B.Dim()
        pixels: B.Array[float] = B.Array(shape=(H, H))

    class Level2Class2D(B.Struct):
        box_size: B.Dim = B.Dim()
        particle: Level1Particle = Level1Particle(H=box_size)
        average: B.Array[float] = B.Array(shape=(box_size, box_size))

    class Level3Experiment(B.Struct):
        N: B.Dim = B.Dim()
        class2d: Level2Class2D = Level2Class2D(box_size=N)

    exp = Level3Experiment(N=256)
    assert exp.is_static()
    assert exp.schema.fields["class2d"].children.fields["average"].shape == (256, 256)
    assert exp.schema.fields["class2d"].children.fields["particle"].children.fields["pixels"].shape == (256, 256)


def test_explicit_nested_override():
    c = Class2D(H=256, W=256, particle=Particle(H=64, W=64))
    assert c.is_static()
    assert c.schema.fields["average"].shape == (256, 256)
    assert c.schema.fields["particle"].children.fields["pixels"].shape == (64, 64)


def test_struct_inheritance_with_dims():
    class BaseRecord(B.Struct):
        H: B.Dim = B.Dim()
        pixels: B.Array[float] = B.Array(shape=(H, H))

    class ExtendedRecord(BaseRecord):
        C: B.Dim = B.Dim()
        channels: B.Array[float] = B.Array(shape=(BaseRecord.H, BaseRecord.H, C))

    record = ExtendedRecord(H=64, C=3)
    assert record.is_static()
    assert record.schema.fields["pixels"].shape == (64, 64)
    assert record.schema.fields["channels"].shape == (64, 64, 3)


def test_dynamic_rebinding_with_dim_instance():
    runtime_dim = B.Dim()
    p = Particle(H=runtime_dim, W=runtime_dim)
    assert not p.is_static()
    assert p.schema.fields["pixels"].shape == (None, None)

    hydrated_p = p.hydrate({runtime_dim: 128})
    assert hydrated_p.is_static()
    assert hydrated_p.schema.fields["pixels"].shape == (128, 128)


def test_multi_array_shared_dims_mixed_ranks():
    class MixedStruct(B.Struct):
        batch: B.Dim = B.Dim()
        H: B.Dim = B.Dim()
        scalar: float
        vector: B.Array[int] = B.Array(shape=(batch,))
        image: B.Array[float] = B.Array(shape=(batch, H, H))
        fixed_cube: B.Array[float] = B.Array(shape=(batch, 3, H, 64))

    mixed = MixedStruct(batch=10, H=32)
    assert mixed.is_static()
    assert mixed.schema.fields["scalar"].shape == ()
    assert mixed.schema.fields["vector"].shape == (10,)
    assert mixed.schema.fields["image"].shape == (10, 32, 32)
    assert mixed.schema.fields["fixed_cube"].shape == (10, 3, 32, 64)


def test_dim_validation_and_type_errors():
    with pytest.raises(TypeError):
        Particle(H="invalid_string")

    with pytest.raises(TypeError):
        Particle(H=[128])

def test_invalid_field_overrides_raise():
    # 1. Rejecting wrong struct type in struct field
    with pytest.raises(TypeError, match="Expected field of type 'Particle', but got 'CTF'"):
        Class2D(particle=CTF())

    # 2. Rejecting primitive in struct field
    with pytest.raises(TypeError, match="Expected field of type 'Particle', but got 'str'"):
        Class2D(particle="not a struct")

    # 3. Rejecting primitive in array field
    with pytest.raises(TypeError, match="Expected an Array marker specification, but got 'str'"):
        Particle(pixels="not an array")

def test_hydrate_fixed_dim_with_none_fails():
    class FixedBox(B.Struct):
        size: B.Dim = B.Dim(64)
        pixels: B.Array[float] = B.Array(shape=(size, size))

    # 1. Attempting to override a fixed dimension (size=64) with None should fail
    with pytest.raises((ValueError, TypeError)):
        FixedBox(size=None)

    class FixedParticle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(64, 64))

    # 2. Attempting to override a fixed static array with dynamic None shape should fail
    with pytest.raises(ValueError):
        FixedParticle(pixels=B.Array[float](shape=(None, None)))


if __name__ == "__main__":
    test_dynamic_init()