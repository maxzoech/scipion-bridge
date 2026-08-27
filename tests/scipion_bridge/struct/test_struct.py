from typing import Optional, Tuple
import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.schema import Entry, Schema, _ArrayEntryBase

def assert_array_entry(
    entry: Entry,
    expected_shape: Tuple[Optional[int], ...],
    is_static: Optional[bool] = None,
) -> None:
    """Narrows Entry to _ArrayEntryBase and validates shape and static status."""
    assert isinstance(entry, _ArrayEntryBase)
    assert entry.shape == expected_shape
    if is_static is not None:
        assert entry.is_static == is_static


def get_child_struct(entry: Entry) -> Schema:
    """Narrows Entry and returns its children Schema."""
    assert entry.children is not None
    return entry.children


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


def test_simple_struct():
    class SimpleStruct(B.Struct):
        val_int: int
        val_float: float
        val_bool: bool

    struct = SimpleStruct()
    assert struct.schema.is_static
    assert set(struct.schema.fields.keys()) == {"val_int", "val_float", "val_bool"}
    assert_array_entry(struct.schema.fields["val_int"], (1,))
    struct.schema.print_tree()


def test_nested_struct():
    class Foo(B.Struct):
        a: np.complex64
        b: float


    class CTF(B.Struct):
        voltage_kv: float
        amplitude_contrast: float
        foo: Foo

    struct = CTF()
    assert struct.schema.is_static
    assert set(struct.schema.fields.keys()) == {"voltage_kv", "amplitude_contrast", "foo"}
    foo_schema = get_child_struct(struct.schema.fields["foo"])
    assert set(foo_schema.fields.keys()) == {"a", "b"}
    struct.schema.print_tree()


def test_basic_inheritance():
    class BaseRecord(B.Struct):
        id: int
        weight: float

    class ExtendedRecord(BaseRecord):
        is_active: bool
        pixels: B.Array[float] = B.Array(shape=(64, 64))

    record = ExtendedRecord()
    assert record.schema.is_static
    assert set(record.schema.fields.keys()) == {"id", "weight", "is_active", "pixels"}
    assert_array_entry(record.schema.fields["id"], (1,))
    assert_array_entry(record.schema.fields["weight"], (1,))
    assert_array_entry(record.schema.fields["is_active"], (1,))
    assert_array_entry(record.schema.fields["pixels"], (64, 64), is_static=True)


def test_multilevel_inheritance():
    class Level1(B.Struct):
        val_a: int
        dim1 = B.Dim()

    class Level2(Level1):
        val_b: float
        arr1 = B.Array[float](shape=(Level1.dim1, Level1.dim1))

    class Level3(Level2):
        val_c: bool
        dim1 = B.Dim(32)
        arr2 = B.Array[int](shape=(Level1.dim1, 10))

    level3 = Level3()
    assert level3.schema.is_static
    assert set(level3.schema.fields.keys()) == {"val_a", "val_b", "val_c", "arr1", "arr2"}
    assert_array_entry(level3.schema.fields["arr1"], (32, 32), is_static=True)
    assert_array_entry(level3.schema.fields["arr2"], (32, 10), is_static=True)
    assert_array_entry(level3.schema.fields["val_a"], (1,), is_static=True)


def test_inheritance_dimension_chaining_and_specialization():
    class BaseFoo(B.Struct):
        batch_size = B.Arg()

    class Foo(BaseFoo):
        batch_size = B.Arg(128)
        box_size = B.Arg(None)
        pixels = B.Array[float](shape=(BaseFoo.batch_size, box_size, box_size))

    class FooFlex(Foo):
        embeddings = B.Array[float](shape=(BaseFoo.batch_size, None))

    assert not Foo.schema.is_static
    assert isinstance(Foo.schema.fields["pixels"], _ArrayEntryBase)
    assert Foo.schema.fields["pixels"].shape == (128, None, None)

    assert not FooFlex.schema.is_static
    assert set(FooFlex.schema.fields.keys()) == {"pixels", "embeddings"}
    assert isinstance(FooFlex.schema.fields["pixels"], _ArrayEntryBase)
    assert FooFlex.schema.fields["pixels"].shape == (128, None, None)
    assert isinstance(FooFlex.schema.fields["embeddings"], _ArrayEntryBase)
    assert FooFlex.schema.fields["embeddings"].shape == (128, None)

    class FooStatic(Foo):
        box_size = B.Arg(64)

    assert FooStatic.schema.is_static
    assert isinstance(FooStatic.schema.fields["pixels"], _ArrayEntryBase)
    assert FooStatic.schema.fields["pixels"].shape == (128, 64, 64)


def test_inheritance_dimension_validation_errors():
    class FixedBase(B.Struct):
        dim = B.Dim(64)

    # 1. Overriding fixed dimension with None raises ValueError
    with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'dim' \(value=64\) with None"):
        class InvalidChild(FixedBase):
            dim = B.Dim(None)

    # 2. Overriding dimension with invalid default value type raises TypeError
    with pytest.raises(TypeError, match="was assigned an invalid default value of type"):
        class InvalidAnnotatedChild(FixedBase):
            dim: B.Dim = 12.34  # type: ignore

    # 3. Unannotated invalid attribute in subclass raises TypeError
    with pytest.raises(TypeError, match="contains class-level attributes missing type annotations"):
        class UnannotatedChild(FixedBase):
            dim = "invalid_dimension"


def test_inheritance_with_nested_structs_and_sets():
    class Inner(B.Struct):
        feature: float
        vector = B.Array[float](shape=(10,))

    class ContainerBase(B.Struct):
        inner: Inner
        items: B.Set[Inner]

    class ContainerChild(ContainerBase):
        name_id: int
        extra_items: B.Set[Inner]

    child = ContainerChild()
    assert set(child.schema.fields.keys()) == {"inner", "items", "name_id", "extra_items"}

    inner_schema = get_child_struct(child.schema.fields["inner"])
    assert set(inner_schema.fields.keys()) == {"feature", "vector"}
    assert_array_entry(inner_schema.fields["vector"], (10,), is_static=True)

    items_entry = child.schema.fields["items"]
    assert items_entry.children is not None
    assert set(items_entry.children.fields.keys()) == {"feature", "vector"}


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

#     specialized_p = p.specialize({runtime_dim: B.Dim(128)})
#     assert specialized_p.schema.is_static
#     assert_array_entry(specialized_p.schema.fields["pixels"], (128, 128))


# def test_specialize_multiple_dims_and_immutability():
#     runtime_h = B.Dim()
#     runtime_w = B.Dim()
#     p = Particle(H=runtime_h, W=runtime_w)
#     assert not p.schema.is_static

#     specialized_p = p.specialize({runtime_h: B.Dim(128), runtime_w: B.Dim(128)})
#     assert specialized_p.schema.is_static
#     assert_array_entry(specialized_p.schema.fields["pixels"], (128, 128))
#     # Original remains unmodified (immutability)
#     assert not p.schema.is_static


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
#     # 1. Rejecting struct field in kwargs
#     with pytest.raises(TypeError, match="got unexpected keyword argument"):
#         Class2D(particle=CTF())

#     # 2. Rejecting primitive in kwargs
#     with pytest.raises(TypeError, match="got unexpected keyword argument"):
#         Class2D(particle="not a struct")

#     # 3. Rejecting array field in kwargs
#     with pytest.raises(TypeError, match="got unexpected keyword argument"):
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

#     # 2. Attempting to pass array field as keyword argument should fail
#     with pytest.raises(TypeError, match="got unexpected keyword argument"):
#         FixedParticle(pixels=B.Array[float](shape=(None, None)))


# def test_struct_definition_and_init_validation_errors():
#     # 1. Untyped class definitions
#     with pytest.raises(TypeError, match="contains class-level attributes missing type annotations"):
#         class UntypedStruct(B.Struct):
#             x = 10

#     # 2. Missing default Array specification
#     with pytest.raises(ValueError, match="Missing required argument 'shape' for Array. Exp"):
#         class MissingArraySpec(B.Struct):
#             pixels: B.Array[float]

#     # 3. Invalid default value for nested SchemaConvertible field
#     with pytest.raises(TypeError, match="was assigned an invalid default value of type"):
#         class InvalidNestedDefault(B.Struct):
#             particle: Particle = "invalid_default" # type: ignore

#     # 4. Non-convertible / unsupported type annotation
#     with pytest.raises(TypeError, match="Unsupported field type"):
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
#     assert_array_entry(c.schema.fields["average"], (64, 64))
#     particles_entry = c.schema.fields["particles"]
#     assert_array_entry(get_child_struct(particles_entry).fields["pixels"], (64, 64))


# def test_annotation_only_struct_and_array_rules():
#     # Value-only struct works with annotation only
#     class SimpleLeaf(B.Struct):

#         val_1: float
#         val_2: int
#         val_3: np.complex64

#     class ParentWithLeaf(B.Struct):
#         bar = B.Dim()

#         leaf: SimpleLeaf

#     parent = ParentWithLeaf()

#     assert parent.schema.is_static
#     assert "leaf" in parent.schema.fields

#     parent.print_schema()

#     # Array with annotation only must raise ValueError (requires shape/rank)
#     with pytest.raises(ValueError, match="Missing required argument 'shape' for Array. Expected a tuple of dimensions"):
#         class InvalidArrayAnnotation(B.Struct):
#             average: B.Array[float]


if __name__ == "__main__":
    test_basic_inheritance()