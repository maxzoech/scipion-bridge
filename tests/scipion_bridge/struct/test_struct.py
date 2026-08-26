from typing import Optional, Tuple
import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.schema import Entry, Schema, _ArrayEntryBase


# def assert_array_entry(
#     entry: Entry,
#     expected_shape: Tuple[Optional[int], ...],
#     is_static: Optional[bool] = None,
# ) -> None:
#     """Narrows Entry to _ArrayEntryBase and validates shape and static status."""
#     assert isinstance(entry, _ArrayEntryBase)
#     assert entry.shape == expected_shape
#     if is_static is not None:
#         assert entry.is_static == is_static


# def get_child_struct(entry: Entry) -> Schema:
#     """Narrows Entry and returns its children Schema."""
#     assert entry.children is not None
#     return entry.children


# class SimpleStruct(B.Struct):
#     val_int: int
#     val_float: float
#     val_bool: bool


# class Foo(B.Struct):
#     a: np.complex64
#     b: float


# class CTF(B.Struct):
#     voltage_kv: float
#     amplitude_contrast: float
#     foo: Foo


# class Particle(B.Struct):
#     H: B.Dim = B.Dim()
#     W: B.Dim = B.Dim()

#     pixels: B.Array[float] = B.Array(shape=(H, W))
#     ctf: CTF
#     foo_2: Foo


# class Class2D(B.Struct):

#     H: B.Dim = B.Dim()
#     W: B.Dim = B.Dim()

#     particle: Particle = Particle(H=H, W=W)
#     average: B.Array[float] = B.Array(shape=(H, W))


# def test_simple_struct():
#     struct = SimpleStruct()
#     assert struct.schema.is_static
#     assert set(struct.schema.fields.keys()) == {"val_int", "val_float", "val_bool"}
#     assert_array_entry(struct.schema.fields["val_int"], (1,))
#     struct.schema.print_tree()


# def test_nested_struct():
#     struct = CTF()
#     assert struct.schema.is_static
#     assert set(struct.schema.fields.keys()) == {"voltage_kv", "amplitude_contrast", "foo"}
#     foo_schema = get_child_struct(struct.schema.fields["foo"])
#     assert set(foo_schema.fields.keys()) == {"a", "b"}
#     struct.schema.print_tree()


# def test_dynamic_init():
#     class2D = Class2D(H=128, W=128)
#     assert class2D.schema.is_static
#     assert class2D.H == 128
#     assert class2D.W == 128

#     # Check average array shape
#     assert_array_entry(class2D.schema.fields["average"], (128, 128), is_static=True)

#     # Check nested particle struct and its pixels array shape
#     particle_schema = get_child_struct(class2D.schema.fields["particle"])
#     assert_array_entry(particle_schema.fields["pixels"], (128, 128), is_static=True)

#     class2D.schema.print_tree()


# def test_unspecialized_dynamic_init():
#     class2D = Class2D()
#     assert not class2D.schema.is_static
#     assert_array_entry(class2D.schema.fields["average"], (None, None), is_static=False)

#     particle_schema = get_child_struct(class2D.schema.fields["particle"])
#     assert_array_entry(particle_schema.fields["pixels"], (None, None), is_static=False)


# def test_instance_isolation():
#     c1 = Class2D(H=64, W=64)
#     c2 = Class2D(H=256, W=256)

#     assert_array_entry(c1.schema.fields["average"], (64, 64))
#     assert_array_entry(get_child_struct(c1.schema.fields["particle"]).fields["pixels"], (64, 64))

#     assert_array_entry(c2.schema.fields["average"], (256, 256))
#     assert_array_entry(get_child_struct(c2.schema.fields["particle"]).fields["pixels"], (256, 256))


# def test_struct_with_default_dim_value():
#     class FixedBox(B.Struct):
#         size: B.Dim = B.Dim(64)
#         pixels: B.Array[float] = B.Array(shape=(size, size))

#     # Default uses 64
#     fb_default = FixedBox()
#     assert fb_default.schema.is_static
#     assert_array_entry(fb_default.schema.fields["pixels"], (64, 64))

#     # Override with 128
#     fb_override = FixedBox(size=128)
#     assert fb_override.schema.is_static
#     assert_array_entry(fb_override.schema.fields["pixels"], (128, 128))


# def test_partial_specialization():
#     class Volume(B.Struct):
#         D: B.Dim = B.Dim()
#         H: B.Dim = B.Dim()
#         W: B.Dim = B.Dim()
#         data: B.Array[float] = B.Array(shape=(D, H, W))

#     v = Volume(D=32)
#     assert not v.schema.is_static
#     assert_array_entry(v.schema.fields["data"], (32, None, None), is_static=False)


# def test_dimension_name_collision_scoping():
#     class InnerParticle(B.Struct):
#         H: B.Dim = B.Dim(64)
#         pixels: B.Array[float] = B.Array(shape=(H, H))

#     class OuterMicrograph(B.Struct):
#         H: B.Dim = B.Dim()
#         # InnerParticle uses its own default H=64; not bound to OuterMicrograph.H
#         particle: InnerParticle = InnerParticle()
#         image: B.Array[float] = B.Array(shape=(H, H))

#     micrograph = OuterMicrograph(H=4096)

#     # Outer micrograph image must be 4096 x 4096
#     assert_array_entry(micrograph.schema.fields["image"], (4096, 4096))

#     # Inner particle pixels MUST remain 64 x 64 (not polluted by OuterMicrograph.H=4096)
#     assert_array_entry(get_child_struct(micrograph.schema.fields["particle"]).fields["pixels"], (64, 64))


# def test_multi_level_alias_chaining():
#     class Level1Particle(B.Struct):
#         H: B.Dim = B.Dim()
#         pixels: B.Array[float] = B.Array(shape=(H, H))

#     class Level2Class2D(B.Struct):
#         box_size: B.Dim = B.Dim()
#         particle: Level1Particle = Level1Particle(H=box_size)
#         average: B.Array[float] = B.Array(shape=(box_size, box_size))

#     class Level3Experiment(B.Struct):
#         N: B.Dim = B.Dim()
#         class2d: Level2Class2D = Level2Class2D(box_size=N)

#     exp = Level3Experiment(N=256)
#     assert exp.schema.is_static
#     l2_schema = get_child_struct(exp.schema.fields["class2d"])
#     assert_array_entry(l2_schema.fields["average"], (256, 256))
#     l1_schema = get_child_struct(l2_schema.fields["particle"])
#     assert_array_entry(l1_schema.fields["pixels"], (256, 256))


# def test_explicit_nested_override():
#     c = Class2D(H=256, W=256, particle=Particle(H=64, W=64))
#     assert c.schema.is_static
#     assert_array_entry(c.schema.fields["average"], (256, 256))
#     assert_array_entry(get_child_struct(c.schema.fields["particle"]).fields["pixels"], (64, 64))


# def test_struct_inheritance_with_dims():
#     class BaseRecord(B.Struct):
#         H: B.Dim = B.Dim()
#         pixels: B.Array[float] = B.Array(shape=(H, H))

#     class ExtendedRecord(BaseRecord):
#         C: B.Dim = B.Dim()
#         channels: B.Array[float] = B.Array(shape=(BaseRecord.H, BaseRecord.H, C))

#     record = ExtendedRecord(H=64, C=3)
#     assert record.schema.is_static
#     assert_array_entry(record.schema.fields["pixels"], (64, 64))
#     assert_array_entry(record.schema.fields["channels"], (64, 64, 3))


# def test_dynamic_rebinding_with_dim_instance():
#     runtime_dim = B.Dim()
#     p = Particle(H=runtime_dim, W=runtime_dim)
#     assert not p.schema.is_static
#     assert_array_entry(p.schema.fields["pixels"], (None, None))

#     specialized_p = p.specialize({runtime_dim: 128})
#     assert specialized_p.schema.is_static
#     assert_array_entry(specialized_p.schema.fields["pixels"], (128, 128))


# def test_specialize_multiple_dims_and_immutability():
#     runtime_h = B.Dim()
#     runtime_w = B.Dim()
#     p = Particle(H=runtime_h, W=runtime_w)
#     assert not p.schema.is_static

#     specialized_p = p.specialize({runtime_h: 128, runtime_w: 128})
#     assert specialized_p.schema.is_static
#     assert_array_entry(specialized_p.schema.fields["pixels"], (128, 128))
#     # Original remains unmodified (immutability)
#     assert not p.schema.is_static


# def test_specialize_default():
#     class FixedBox(B.Struct):
#         size: B.Dim = B.Dim(64)
#         pixels: B.Array[float] = B.Array(shape=(size, size))

#     box = FixedBox()
#     default_box = box.default()
#     assert default_box.schema.is_static
#     assert_array_entry(default_box.schema.fields["pixels"], (64, 64))


# def test_multi_array_shared_dims_mixed_ranks():
#     class MixedStruct(B.Struct):
#         batch: B.Dim = B.Dim()
#         H: B.Dim = B.Dim()
#         scalar: float
#         vector: B.Array[int] = B.Array(shape=(batch,))
#         image: B.Array[float] = B.Array(shape=(batch, H, H))
#         fixed_cube: B.Array[float] = B.Array(shape=(batch, 3, H, 64))

#     mixed = MixedStruct(batch=10, H=32)
#     assert mixed.schema.is_static
#     assert_array_entry(mixed.schema.fields["scalar"], (1,))
#     assert_array_entry(mixed.schema.fields["vector"], (10,))
#     assert_array_entry(mixed.schema.fields["image"], (10, 32, 32))
#     assert_array_entry(mixed.schema.fields["fixed_cube"], (10, 3, 32, 64))


#     mixed.print_schema()

# def test_dim_validation_and_type_errors():
#     with pytest.raises(TypeError):
#         Particle(H="invalid_string")

#     with pytest.raises(TypeError):
#         Particle(H=[128])


# def test_invalid_field_overrides_raise():
#     # 1. Rejecting wrong struct type in struct field
#     with pytest.raises(TypeError, match="Expected field of type 'Particle', but got 'CTF'"):
#         Class2D(particle=CTF())

#     # 2. Rejecting primitive in struct field
#     with pytest.raises(TypeError, match="Expected field of type 'Particle', but got 'str'"):
#         Class2D(particle="not a struct")

#     # 3. Rejecting primitive in array field
#     with pytest.raises(TypeError, match="Expected an Array marker specification, but got 'str'"):
#         Particle(pixels="not an array")


# def test_override_fixed_dim_with_none_fails():
#     class FixedBox(B.Struct):
#         size: B.Dim = B.Dim(64)
#         pixels: B.Array[float] = B.Array(shape=(size, size))

#     # 1. Attempting to override a fixed dimension (size=64) with None should fail
#     with pytest.raises((ValueError, TypeError)):
#         FixedBox(size=None)

#     class FixedParticle(B.Struct):
#         pixels: B.Array[float] = B.Array(shape=(64, 64))

#     # 2. Attempting to override a fixed static array with dynamic None shape should fail
#     with pytest.raises(ValueError):
#         FixedParticle(pixels=B.Array[float](shape=(None, None)))


# def test_struct_definition_and_init_validation_errors():
#     # 1. Untyped class definitions
#     with pytest.raises(TypeError, match="contains class-level attributes missing type annotations"):
#         class UntypedStruct(B.Struct):
#             x = 10

#     # 2. Missing default Array specification
#     with pytest.raises(ValueError, match="missing a default Array specification"):
#         class MissingArraySpec(B.Struct):
#             pixels: B.Array[float]

#     # 3. Invalid default value for nested SchemaConvertible field
#     with pytest.raises(TypeError, match="expects a default value of type 'Particle'"):
#         class InvalidNestedDefault(B.Struct):
#             particle: Particle = "invalid_default" # type: ignore

#     # 4. Non-convertible / unsupported type annotation
#     with pytest.raises(TypeError, match="Invalid type annotation"):
#         class UnsupportedTypeStruct(B.Struct):
#             handler: object

#     # 5. Unexpected keyword argument in Struct constructor
#     with pytest.raises(TypeError, match="got unexpected keyword argument"):
#         Particle(unexpected_param=123)


# def test_relaxed_type_annotations_syntax():
#     class DynamicParticle(B.Struct):
#         H = B.Dim()
#         pixels = B.Array[float](shape=(H, H))
#         voltage_kv: float

#     class Class2D(B.Struct):
#         H = B.Dim()
#         num_particles = B.Arg()
#         average = B.Array[float](shape=(H, H))
#         particles = B.Set[DynamicParticle](
#             num_particles,
#             H=H,
#         )

#     c = Class2D(H=64, num_particles=10)
#     assert c.schema.is_static
#     assert assert_array_entry(c.schema.fields["average"], (64, 64)) is None
#     particles_entry = c.schema.fields["particles"]
#     assert get_child_struct(particles_entry).fields["pixels"].shape == (64, 64)


def test_annotation_only_struct_and_array_rules():
    # Value-only struct works with annotation only
    class SimpleLeaf(B.Struct):

        val_1: float
        val_2: int
        val_3: np.complex64

    class ParentWithLeaf(B.Struct):
        bar = B.Dim()

        leaf: SimpleLeaf

    parent = ParentWithLeaf()

    assert parent.schema.is_static
    assert "leaf" in parent.schema.fields

    parent.print_schema()

    # Array with annotation only must raise ValueError (requires shape/rank)
    with pytest.raises(ValueError, match="Missing required argument 'shape' for Array. Expected a tuple of dimensions"):
        class InvalidArrayAnnotation(B.Struct):
            average: B.Array[float]


if __name__ == "__main__":
    test_annotation_only_struct_and_array_rules()