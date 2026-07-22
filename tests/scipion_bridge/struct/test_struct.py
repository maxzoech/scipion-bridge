import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.schema import (
    _ArrayEntry,
    _ArrayLocation,
    Schema
)


class SimpleStruct(B.Struct):
    val_int: int
    val_float: float
    val_bool: bool


class NestedChild(B.Struct):
    x: int
    y: np.float64


class NestedParent(B.Struct):
    name_id: int
    child: NestedChild


class DeepNested(B.Struct):
    parent: NestedParent
    tag: int


class Foo(B.Struct):
    a: np.complex64
    b: float


class CTF(B.Struct):
    voltage_kv: float
    amplitude_contrast: float
    foo: Foo


class Particle(B.Struct):
    pixels: B.Array[float]
    ctf: CTF
    foo_2: Foo


def test_schema_creation_basic():
    schema = SimpleStruct.schema()
    assert isinstance(schema, Schema)
    assert schema.entries() == {"val_int", "val_float", "val_bool"}

    entry_int = schema.fields["val_int"]
    assert isinstance(entry_int, _ArrayEntry)
    assert entry_int.dtype == np.dtype(int)
    assert entry_int.storage == _ArrayLocation.AUTOMATIC

    entry_float = schema.fields["val_float"]
    assert isinstance(entry_float, _ArrayEntry)
    assert entry_float.dtype == np.dtype(float)
    assert entry_float.storage == _ArrayLocation.AUTOMATIC

    entry_bool = schema.fields["val_bool"]
    assert isinstance(entry_bool, _ArrayEntry)
    assert entry_bool.dtype == np.dtype(bool)
    assert entry_bool.storage == _ArrayLocation.AUTOMATIC


def test_schema_creation_nested():
    parent_schema = NestedParent.schema()
    assert isinstance(parent_schema, Schema)
    assert parent_schema.entries() == {"name_id", "child"}

    assert isinstance(parent_schema.fields["name_id"], _ArrayEntry)
    assert parent_schema.fields["name_id"].dtype == np.dtype(int)

    child_cls = parent_schema.fields["child"]
    assert child_cls == NestedChild
    child_schema = child_cls.schema()
    assert child_schema.entries() == {"x", "y"}
    assert isinstance(child_schema.fields["x"], _ArrayEntry)
    assert child_schema.fields["x"].dtype == np.dtype(int)
    assert isinstance(child_schema.fields["y"], _ArrayEntry)
    assert child_schema.fields["y"].dtype == np.dtype(np.float64)


def test_untyped_attribute_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        class UntypedStruct(B.Struct):
            x = 10
        UntypedStruct.schema()

    err_msg = str(exc_info.value)
    assert "declares attributes without type annotations." in err_msg
    assert "UntypedStruct" in err_msg


def test_incompatible_attribute_single_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        class SingleIncompatibleStruct(B.Struct):
            bad_attr: object
        SingleIncompatibleStruct.schema()

    err_msg = str(exc_info.value)
    assert "The attribute 'bad_attr' cannot be declared in struct" in err_msg
    assert "SingleIncompatibleStruct" in err_msg
    assert "because it does not support array serialization." in err_msg


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


def test_array_generic_and_edge_cases():
    class ArrayStruct(B.Struct):
        arr_bare: B.Array
        arr_typed: B.Array[int]

    schema = ArrayStruct.schema()
    assert "arr_bare" in schema.entries()
    assert "arr_typed" in schema.entries()

    with pytest.raises(TypeError) as exc_info:
        class BadArrayStruct(B.Struct):
            arr_bad: B.Array[object]
        BadArrayStruct.schema()

    assert "arr_bad" in str(exc_info.value)


def test_supports_array_storage_direct():
    from scipion_bridge.core.struct.schema import _supports_array_storage
    assert _supports_array_storage(NestedChild) is not None
    assert _supports_array_storage(B.Array) is True
    assert _supports_array_storage(B.Array[int]) is True
    assert _supports_array_storage(int) is True
    assert _supports_array_storage(object) is False
    assert _supports_array_storage("invalid_type_obj") is False


def test_unassigned_attribute_raises_attribute_error():
    s = SimpleStruct()
    with pytest.raises(AttributeError) as exc_info:
        _ = s.val_int
    assert "SimpleStruct" in str(exc_info.value)
    assert "val_int" in str(exc_info.value)

