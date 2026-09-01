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


def test_simple_struct():
    class SimpleStruct(B.Struct):
        val_int: int
        val_float: float
        val_bool: bool

    struct = SimpleStruct()
    struct.print_schema()

    assert struct.schema().is_static
    assert set(struct.schema().fields.keys()) == {"val_int", "val_float", "val_bool"}
    assert_array_entry(struct.schema().fields["val_int"], (1,))


def test_nested_struct():
    class Foo(B.Struct):
        a: np.complex64
        b: float


    class CTF(B.Struct):
        voltage_kv: float
        amplitude_contrast: float
        foo: Foo

    struct = CTF()
    assert struct.schema().is_static
    assert set(struct.schema().fields.keys()) == {"voltage_kv", "amplitude_contrast", "foo"}
    foo_schema = get_child_struct(struct.schema().fields["foo"])
    assert set(foo_schema.fields.keys()) == {"a", "b"}
    struct.schema().print_tree()


def test_basic_inheritance():
    class BaseRecord(B.Struct):
        id: int
        weight: float

    class ExtendedRecord(BaseRecord):
        is_active: bool
        pixels: B.Array[float] = B.Array(shape=(64, 64))

    record = ExtendedRecord()
    assert record.schema().is_static
    assert set(record.schema().fields.keys()) == {"id", "weight", "is_active", "pixels"}
    assert_array_entry(record.schema().fields["id"], (1,))
    assert_array_entry(record.schema().fields["weight"], (1,))
    assert_array_entry(record.schema().fields["is_active"], (1,))
    assert_array_entry(record.schema().fields["pixels"], (64, 64), is_static=True)


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
    assert level3.schema().is_static
    assert set(level3.schema().fields.keys()) == {"val_a", "val_b", "val_c", "arr1", "arr2"}
    assert_array_entry(level3.schema().fields["arr1"], (32, 32), is_static=True)
    assert_array_entry(level3.schema().fields["arr2"], (32, 10), is_static=True)
    assert_array_entry(level3.schema().fields["val_a"], (1,), is_static=True)


def test_inheritance_dimension_chaining_and_specialization():
    class BaseFoo(B.Struct):
        batch_size = B.Arg()

    class Foo(BaseFoo):
        batch_size = B.Arg(128)
        box_size = B.Arg(None)
        pixels = B.Array[float](shape=(BaseFoo.batch_size, box_size, box_size))

    class FooFlex(Foo):
        embeddings = B.Array[float](shape=(BaseFoo.batch_size, None))

    assert not Foo.schema().is_static
    assert_array_entry(Foo.schema().fields["pixels"], (128, None, None))

    assert not FooFlex.schema().is_static
    assert set(FooFlex.schema().fields.keys()) == {"pixels", "embeddings"}
    assert_array_entry(FooFlex.schema().fields["pixels"], (128, None, None))
    assert_array_entry(FooFlex.schema().fields["embeddings"], (128, None))

    class FooStatic(Foo):
        box_size = B.Arg(64)

    assert FooStatic.schema().is_static
    assert_array_entry(FooStatic.schema().fields["pixels"], (128, 64, 64))


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
    assert set(child.schema().fields.keys()) == {"inner", "items", "name_id", "extra_items"}

    inner_schema = get_child_struct(child.schema().fields["inner"])
    assert set(inner_schema.fields.keys()) == {"feature", "vector"}
    assert_array_entry(inner_schema.fields["vector"], (10,), is_static=True)

    items_entry = child.schema().fields["items"]
    assert items_entry.children is not None
    assert set(items_entry.children.fields.keys()) == {"feature", "vector"}


def test_class_vs_instance_get_semantics():
    class Record(B.Struct):
        dim = B.Dim(64)

    # Access on class returns the Arg/Dim descriptor instance
    assert isinstance(Record.dim, B.Arg)
    assert Record.dim.value == 64
    assert Record.dim.name == "dim"

    # Access on instance returns the resolved integer value
    inst = Record()
    assert inst.dim == 64

def test_unassigned_class_vs_instance_get():
    class DynamicRecord(B.Struct):
        dim = B.Dim()

    assert isinstance(DynamicRecord.dim, B.Arg)
    assert DynamicRecord.dim.value is None

    inst = DynamicRecord()
    assert inst.dim is None

def test_instance_set_raises_attribute_error():
    class Record(B.Struct):
        dim = B.Dim(64)

    inst = Record()
    with pytest.raises(AttributeError, match="class-level schema parameter"):
        inst.dim = 128

def test_dim_equality_with_ints_and_dims():
    d1 = B.Dim(64)
    d2 = B.Dim(64)
    d3 = B.Dim(128)
    d_none1 = B.Dim()
    d_none2 = B.Dim()

    assert d1 == d2
    assert d1 == 64
    assert 64 == d1
    assert d1 != d3
    assert d1 != 128
    assert d_none1 == d_none2
    assert d_none1 == None
    assert d1 != d_none1

def test_int_conversion():
    d = B.Dim(42)
    assert int(d) == 42

    d_unassigned = B.Dim()
    with pytest.raises(TypeError, match="Cannot convert unassigned Dim to int"):
        int(d_unassigned)

def test_repr_and_str():
    d = B.Dim(64, name="H")
    assert str(d) == "H:64"
    assert "Arg(64, name='H')" in repr(d)

    d_unbound = B.Dim(name="W")
    assert str(d_unbound) == "W"


def test_sibling_subclasses_do_not_mutate_base_dimension():
    """Edge case: subclassing must not mutate the parent class descriptor."""
    class Base(B.Struct):
        H = B.Dim()
        pixels = B.Array[float](shape=(H, H))

    assert Base.H.value is None

    # Create Subclass A with H=64
    class SubA(Base):
        H = B.Dim(64)
        latents = B.Array[float](shape=(H,))

    assert SubA.H.value == 64
    # Base.H must remain unassigned (None), NOT mutated to 64
    assert Base.H.value is None, "BUG: Defining SubA mutated Base.H!"

    SubA.print_schema()


def test_sibling_subclasses_do_not_poison_each_other():
    """Edge case: defining multiple siblings with different values."""
    class Base(B.Struct):
        H = B.Dim()
        pixels = B.Array[float](shape=(H, H))

    class Sub64(Base):
        H = B.Dim(64)

    class Sub128(Base):
        H = B.Dim(128)

    class SubDynamic(Base):
        H = B.Dim()

    assert Sub64.H.value == 64
    assert Sub128.H.value == 128
    assert SubDynamic.H.value is None
    assert Base.H.value is None

def test_multi_tier_inheritance_isolation():
    class Root(B.Struct):
        dim = B.Dim()

    class MidA(Root):
        dim = B.Dim(32)

    class MidB(Root):
        dim = B.Dim(64)

    class LeafA(MidA):
        pass

    class LeafB(MidB):
        pass

    assert Root.dim.value is None
    assert MidA.dim.value == 32
    assert MidB.dim.value == 64
    assert LeafA.dim.value == 32
    assert LeafB.dim.value == 64

def test_diamond_inheritance_dimension_resolution():
    class Root(B.Struct):
        dim = B.Dim()

    class Left(Root):
        dim = B.Dim(16)

    class Right(Root):
        dim = B.Dim(16)

    class Diamond(Left, Right):
        pass

    assert Diamond.dim.value == 16
    assert Root.dim.value is None


def test_override_fixed_dimension_with_none_rejected():
    class FixedBase(B.Struct):
        dim = B.Dim(64)

    with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'dim' \(value=64\) with None"):
        class InvalidSub(FixedBase):
            dim = B.Dim(None)

def test_override_dimension_with_invalid_type_rejected():
    with pytest.raises(TypeError, match="Expected Dim, int, or None"):
        B.Dim("invalid_string")  # type: ignore

    with pytest.raises(TypeError, match="Expected Dim, int, or None"):
        B.Dim([64])  # type: ignore

def test_validate_method_type_checks():
    d = B.Dim(64, name="H")
    with pytest.raises(TypeError, match="Expected Dim, int, or None"):
        d.validate("string_value")

    with pytest.raises(ValueError, match="Cannot override fixed dimension"):
        d.validate(None)


def test_dimension_name_scoping_no_crosstalk():
    class Inner(B.Struct):
        H = B.Dim(64)
        pixels = B.Array[float](shape=(H, H))

    class Outer(B.Struct):
        H = B.Dim()
        inner: Inner
        image = B.Array[float](shape=(H, H))

    assert Inner.H.value == 64
    assert Outer.H.value is None
    assert_array_entry(Inner.schema().fields["pixels"], (64, 64))
    assert_array_entry(Outer.schema().fields["image"], (None, None))


def test_schema_specialization():
    class Foo(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, H))

    # 1. Test specialization via subclass definition
    class Foo128(Foo, specializations={"H": 128}):
        pass

    # 2. Test specialization via .static() factory method
    FooStatic128 = Foo.static(H=128)

    # Verify base class remained untouched
    assert Foo.H.value is None
    assert not Foo.schema().is_static
    assert_array_entry(Foo.schema().fields["pixels"], (None, None), is_static=False)

    # Verify class-level subclass specialization
    assert Foo128.H.value == 128
    assert Foo128.schema().is_static
    assert_array_entry(Foo128.schema().fields["pixels"], (128, 128), is_static=True)

    # Verify dynamic .static() specialization
    assert FooStatic128.H.value == 128
    assert FooStatic128.schema().is_static
    assert_array_entry(FooStatic128.schema().fields["pixels"], (128, 128), is_static=True)

    # Verify instance-level schema propagation
    inst_128 = Foo128()
    inst_static = FooStatic128()
    assert_array_entry(inst_128.schema().fields["pixels"], (128, 128), is_static=True)
    assert_array_entry(inst_static.schema().fields["pixels"], (128, 128), is_static=True)


def test_multidim_partial_and_full_specialization():
    class TensorData(B.Struct):
        H = B.Dim(None)
        W = B.Dim(None)
        C = B.Dim(None)
        image = B.Array[float](shape=(H, W, C))
        mask = B.Array[bool](shape=(H, W))

    # Partial specialization: only specialize channel C
    TensorRGB = TensorData.static(C=3)
    assert TensorRGB.C.value == 3
    assert TensorRGB.H.value is None
    assert TensorRGB.W.value is None
    assert not TensorRGB.schema().is_static
    assert_array_entry(TensorRGB.schema().fields["image"], (None, None, 3), is_static=False)
    assert_array_entry(TensorRGB.schema().fields["mask"], (None, None), is_static=False)

    # Further specialize dimensions H and W
    TensorFixed = TensorRGB.static(H=64, W=128)
    assert TensorFixed.H.value == 64
    assert TensorFixed.W.value == 128
    assert TensorFixed.C.value == 3
    assert TensorFixed.schema().is_static
    assert_array_entry(TensorFixed.schema().fields["image"], (64, 128, 3), is_static=True)
    assert_array_entry(TensorFixed.schema().fields["mask"], (64, 128), is_static=True)


def test_multi_level_inheritance_specialization():
    class BaseVolume(B.Struct):
        Z = B.Dim(None)
        Y = B.Dim(None)
        X = B.Dim(None)
        data = B.Array[float](shape=(Z, Y, X))

    # Level 1: Specialize Z via subclass definition
    class SlabVolume(BaseVolume, specializations={"Z": 1}):
        pass

    assert SlabVolume.Z.value == 1
    assert SlabVolume.Y.value is None
    assert SlabVolume.X.value is None
    assert not SlabVolume.schema().is_static
    assert_array_entry(SlabVolume.schema().fields["data"], (1, None, None), is_static=False)

    # Level 2: Specialize Y via subclass definition
    class SquareSlab(SlabVolume, specializations={"Y": 256}):
        pass

    assert SquareSlab.Z.value == 1
    assert SquareSlab.Y.value == 256
    assert SquareSlab.X.value is None
    assert not SquareSlab.schema().is_static
    assert_array_entry(SquareSlab.schema().fields["data"], (1, 256, None), is_static=False)

    # Level 3: Fully specialize via .static()
    Cube256 = SquareSlab.static(X=256)
    assert Cube256.Z.value == 1
    assert Cube256.Y.value == 256
    assert Cube256.X.value == 256
    assert Cube256.schema().is_static
    assert_array_entry(Cube256.schema().fields["data"], (1, 256, 256), is_static=True)


def test_nested_struct_specialization():
    class Patch(B.Struct):
        size = B.Dim(None)
        pixels = B.Array[float](shape=(size, size))

    Patch64 = Patch.static(size=64)

    class Container(B.Struct):
        N = B.Dim(None)
        patch = Patch64()

        weights = B.Array[float](shape=(N,))

    # Outer struct is partially dynamic due to N
    assert not Container.schema().is_static
    patch_schema = get_child_struct(Container.schema().fields["patch"])
    assert patch_schema.is_static is True
    assert_array_entry(patch_schema.fields["pixels"], (64, 64), is_static=True)
    assert_array_entry(Container.schema().fields["weights"], (None,), is_static=False)

    # Fully specializing outer struct
    ContainerFixed = Container.static(N=10)
    assert ContainerFixed.schema().is_static is True
    fixed_patch_schema = get_child_struct(ContainerFixed.schema().fields["patch"])
    assert fixed_patch_schema.is_static is True
    assert_array_entry(fixed_patch_schema.fields["pixels"], (64, 64), is_static=True)
    assert_array_entry(ContainerFixed.schema().fields["weights"], (10,), is_static=True)


def test_specialization_error_handling():
    class Foo(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, H))

    # Unknown dimension name
    with pytest.raises(TypeError, match="unexpected dimension argument: 'UNKNOWN'"):
        Foo.static(UNKNOWN=64)

    with pytest.raises(TypeError, match="Unknown schema overwrite argument 'UNKNOWN'"):
        class InvalidSub(Foo, specializations={"UNKNOWN": 64}):
            pass

    # Overriding fixed dimension with None is rejected
    Foo64 = Foo.static(H=64)
    with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'H' \(value=64\) with None"):
        Foo64.static(H=None) # type: ignore

    with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'H' \(value=64\) with None"):
        class InvalidSubFixed(Foo64, specializations={"H": None}): # type: ignore
            pass

    # Invalid type for dimension
    with pytest.raises(TypeError, match="Expected Dim, int, or None"):
        Foo.static(H="invalid")  # type: ignore


def test_set_of_specialized_struct_schema():
    class Particle(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, H))

    Particle128 = Particle.static(H=128)
    set_schema = B.Set[Particle128].schema()
    assert set_schema.is_static is True
    assert "pixels" in set_schema.fields
    assert_array_entry(set_schema.fields["pixels"], (128, 128), is_static=True)


def test_struct_containing_specialized_set():
    class Particle(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, H))

    Particle64 = Particle.static(H=64)

    class Micrograph(B.Struct):
        W = B.Dim(None)
        raw = B.Array[float](shape=(W, W))
        particles = B.Set[Particle64](capacity=10)

    assert not Micrograph.schema().is_static
    MicrographFixed = Micrograph.static(W=1024)
    assert MicrographFixed.schema().is_static is True
    assert_array_entry(MicrographFixed.schema().fields["raw"], (1024, 1024), is_static=True)
    particles_schema = get_child_struct(MicrographFixed.schema().fields["particles"])
    assert_array_entry(particles_schema.fields["pixels"], (64, 64), is_static=True)

    MicrographFixed.print_schema()


def test_basic_set_storage():

    class Data(B.Struct):
        pixels: B.Array[float] = B.Array[float](shape=(128, 128,))
        bar: float

    data = Data()
    noise = np.random.uniform(size=[128, 128])

    data.bar = 42.0
    data.pixels = noise
    
    assert np.allclose(data.pixels, noise)


def test_array_instantiation_validation():
    # 1. Missing shape raises ValueError
    with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
        B.Array()

    with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
        B.Array[float]()

    # 2. Non-sequence shape raises TypeError
    with pytest.raises(TypeError, match="Expected shape to be a tuple or list of dimensions"):
        B.Array[float](shape=128)  # type: ignore

    # 3. Negative dimension raises ValueError
    with pytest.raises(ValueError, match="Array dimension cannot be negative"):
        B.Array[float](shape=(-1, 64))

    # 4. Shape with list input works
    arr = B.Array[float](shape=[64, 64])
    assert len(arr.shape_spec) == 2
    assert arr.shape_spec[0].value == 64
    assert arr.shape_spec[1].value == 64


def test_array_unassigned_annotation_raises_helpful_error():
    with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
        class BadStruct(B.Struct):
            pixels: B.Array[float]


def test_array_class_access_semantics():
    class Particle(B.Struct):
        H = B.Dim(64)
        pixels = B.Array[np.float32](shape=(H, H))

    # Class-level access yields an Array bound to the owner class
    arr = Particle.pixels
    assert isinstance(arr, B.Array)
    assert arr.dtype == np.dtype(np.float32)
    assert arr.shape == (64, 64)
    assert repr(arr) == "Array[float32](shape=(64, 64), owner=Particle)"
    entry = arr.convert_to_entry()
    assert isinstance(entry, _ArrayEntryBase)
    assert entry.shape == (64, 64)

    # Calling schema() on uninstantiated Array class raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Cannot get schema directly from an uninstantiated Array class"):
        B.Array.schema()

    # Calling default() raises ValueError
    with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
        B.Array.default()


def test_array_missing_dtype_error():
    # Defining a Struct with an Array lacking a dtype specification raises TypeError during class creation
    with pytest.raises(TypeError, match="missing a dtype specification"):
        class UnspecifiedArrayStruct(B.Struct):
            pixels = B.Array(shape=(64, 64))


def test_array_schema_convertible_methods():
    arr = B.Array[float](shape=(10, 10))

    with pytest.raises(NotImplementedError, match="Cannot get schema directly from an uninstantiated Array class"):
        B.Array.schema()

    entry = arr.convert_to_entry()
    assert isinstance(entry, _ArrayEntryBase)
    assert entry.shape == (10, 10)
    assert repr(arr) == "Array[float64](shape=(10, 10))"


def test_array_descriptor_error_handling():
    class Record(B.Struct):
        dim = B.Dim()
        pixels = B.Array[float](shape=(dim, dim))

    inst = Record()

    # 1. Reading dynamic (non-static) array raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Dynamic array access on instance is not supported yet"):
        _ = inst.pixels

    # 2. Writing dynamic (non-static) array raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Dynamic array assignment on instance is not supported yet"):
        inst.pixels = np.ones((10, 10))

    # 3. Accessing/setting Array descriptor on non-Struct instance raises TypeError
    class NonStruct:
        pixels = B.Array[float](shape=(10,))

    non_struct_inst = NonStruct()
    with pytest.raises(TypeError, match="on non-Struct instance"):
        _ = non_struct_inst.pixels # type: ignore

    with pytest.raises(TypeError, match="on non-Struct instance"):
        non_struct_inst.pixels = np.ones(10)


if __name__ == "__main__":
    from scipion_bridge.backend.standalone.container import configure_default_env
        
    # Wire the container for 'scipion_bridge'
    container = configure_default_env()

    test_basic_set_storage()