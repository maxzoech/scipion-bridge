import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.entries import (
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _ArrayLocation,
    _StructEntry,
)
from scipion_bridge.core.struct.schema import (
    Schema,
    create_schema,
    _supports_array_storage,
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


def test_schema_creation_basic():
    schema = create_schema(SimpleStruct)
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
    parent_schema = create_schema(NestedParent)
    assert isinstance(parent_schema, Schema)
    assert parent_schema.entries() == {"name_id", "child"}

    assert isinstance(parent_schema.fields["name_id"], _ArrayEntry)
    assert parent_schema.fields["name_id"].dtype == np.dtype(int)

    child_entry = parent_schema.fields["child"]
    assert isinstance(child_entry, _StructEntry)
    assert child_entry.struct_cls == NestedChild

    child_schema = child_entry.schema
    assert child_schema.entries() == {"x", "y"}
    assert isinstance(child_schema.fields["x"], _ArrayEntry)
    assert child_schema.fields["x"].dtype == np.dtype(int)
    assert isinstance(child_schema.fields["y"], _ArrayEntry)
    assert child_schema.fields["y"].dtype == np.dtype(np.float64)


def test_schema_creation_deep_nested():
    deep_schema = create_schema(DeepNested)
    assert isinstance(deep_schema, Schema)
    assert deep_schema.entries() == {"parent", "tag"}
    assert isinstance(deep_schema.fields["tag"], _ArrayEntry)

    parent_entry = deep_schema.fields["parent"]
    assert isinstance(parent_entry, _StructEntry)
    assert parent_entry.struct_cls == NestedParent
    assert parent_entry.schema.entries() == {"name_id", "child"}


def test_untyped_attribute_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        class UntypedStruct(B.Struct):
            x = 10
        create_schema(UntypedStruct)

    err_msg = str(exc_info.value)
    assert "declares attributes without type annotations." in err_msg
    assert "UntypedStruct" in err_msg


def test_incompatible_attribute_single_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        class SingleIncompatibleStruct(B.Struct):
            bad_attr: object
        create_schema(SingleIncompatibleStruct)

    err_msg = str(exc_info.value)
    assert "The attribute 'bad_attr' cannot be declared in struct" in err_msg
    assert "SingleIncompatibleStruct" in err_msg
    assert "because it does not support array serialization." in err_msg


def test_incompatible_attribute_multiple_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        class MultiIncompatibleStruct(B.Struct):
            bad1: object
            bad2: dict
        create_schema(MultiIncompatibleStruct)

    err_msg = str(exc_info.value)
    assert "The attributes 'bad1 and bad2' cannot be declared in struct" in err_msg
    assert "MultiIncompatibleStruct" in err_msg


def test_incompatible_nested_attribute_raises_type_error():
    class InvalidChild(B.Struct):
        invalid_field: object

    with pytest.raises(TypeError) as exc_info:
        class InvalidParent(B.Struct):
            child: InvalidChild
        create_schema(InvalidParent)

    err_msg = str(exc_info.value)
    assert "cannot be declared in struct" in err_msg


def test_schema_entries():
    schema = create_schema(SimpleStruct)
    entries = schema.entries()
    assert isinstance(entries, set)
    assert entries == {"val_int", "val_float", "val_bool"}


def test_array_generic_and_edge_cases():
    class ArrayStruct(B.Struct):
        arr_bare: B.Array[float]
        arr_typed: B.Array[int]

    schema = create_schema(ArrayStruct)
    assert "arr_bare" in schema.entries()
    assert "arr_typed" in schema.entries()

    with pytest.raises(TypeError) as exc_info:
        class BadArrayStruct(B.Struct):
            arr_bad: B.Array[object]
        create_schema(BadArrayStruct)

    assert "arr_bad" in str(exc_info.value)


def test_supports_array_storage_direct():
    assert _supports_array_storage(NestedChild) is not None
    assert _supports_array_storage(int) is True
    assert _supports_array_storage(object) is False
    assert _supports_array_storage("invalid_type_obj") is False


def test_schema_is_static():
    assert SimpleStruct.schema().is_static is True
    assert DeepNested.schema().is_static is True

    class DynamicArrayStruct(B.Struct):
        pixels: B.Array[float]

    assert DynamicArrayStruct.schema().is_static is False

    class ParentWithDynamicChild(B.Struct):
        child: DynamicArrayStruct
        x: int

    assert ParentWithDynamicChild.schema().is_static is False


def test_array_entry_is_static():
    static_entry = _ArrayEntry(
        dtype=np.dtype(float),
        storage=_ArrayLocation.AUTOMATIC,
        min_shape=(10, 10),
        max_shape=(10, 10),
        preferred_shape=None,
    )
    assert static_entry.is_static is True

    dynamic_entry = _ArrayEntry(
        dtype=np.dtype(float),
        storage=_ArrayLocation.AUTOMATIC,
        min_shape=None,
        max_shape=None,
        preferred_shape=None,
    )
    assert dynamic_entry.is_static is False

    mismatched_entry = _ArrayEntry(
        dtype=np.dtype(float),
        storage=_ArrayLocation.AUTOMATIC,
        min_shape=(5,),
        max_shape=(10,),
        preferred_shape=None,
    )
    assert mismatched_entry.is_static is False


def test_format_entry():
    arr_entry = _ArrayEntry(
        dtype=np.dtype(float),
        storage=_ArrayLocation.AUTOMATIC,
        min_shape=(10,),
        max_shape=(10,),
        preferred_shape=None,
    )
    assert arr_entry.format_entry("my_arr") == "my_arr: Array[float64](storage: auto, min: (10,), max: (10,))"

    arr_set_entry = _ArraySetEntry(
        dtype=np.dtype(int),
        storage=_ArrayLocation.AUTOMATIC,
        shape=(64, 64),
    )
    assert arr_set_entry.format_entry("my_set") == "my_set: ArraySet[int64](storage: auto, shape: (64, 64))"

    ragged_entry = _RaggedArraySetEntry(
        dtype=np.dtype(float),
        storage=_ArrayLocation.AUTOMATIC,
        min_shape=(1,),
        max_shape=None,
        preferred_shape=None,
    )
    assert ragged_entry.format_entry("my_ragged") == "my_ragged: RaggedArraySet[float64](storage: auto, min: (1,))"





def test_tree_iter():
    schema = create_schema(DeepNested)
    leaves = dict(schema.tree_iter())

    assert "parent.name_id" in leaves
    assert "parent.child.x" in leaves
    assert "parent.child.y" in leaves
    assert "tag" in leaves

    assert isinstance(leaves["tag"], _ArrayEntry)


def test_iter_leaves():
    schema = create_schema(DeepNested)
    leaves = dict(schema.iter_leaves())
    
    assert "parent.name_id" in leaves
    assert "parent.child.x" in leaves
    assert "parent.child.y" in leaves
    assert "tag" in leaves
    
    assert isinstance(leaves["tag"], _ArrayEntry)

def test_map_leaves():
    schema = create_schema(DeepNested)
    
    # Map leaves to their entry's names
    mapped = schema.map_leaves(lambda path, entry: entry.entry_name)
    
    expected = {
        "parent.name_id": "Array",
        "parent.child.x": "Array",
        "parent.child.y": "Array",
        "tag": "Array",
    }
    assert mapped == expected
