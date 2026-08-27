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


if __name__ == "__main__":
    test_basic_inheritance()