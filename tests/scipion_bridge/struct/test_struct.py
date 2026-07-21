import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.schema import (
    ArrayEntry,
    ArrayLocation,
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


def test_schema_creation_basic():
    schema = SimpleStruct._scipion_bridge_schema
    assert isinstance(schema, Schema)
    assert set(schema.tree.keys()) == {"val_int", "val_float", "val_bool"}

    entry_int = schema.tree["val_int"]
    assert isinstance(entry_int, ArrayEntry)
    assert entry_int.dtype == np.dtype(int)
    assert entry_int.storage == ArrayLocation.AUTOMATIC

    entry_float = schema.tree["val_float"]
    assert isinstance(entry_float, ArrayEntry)
    assert entry_float.dtype == np.dtype(float)
    assert entry_float.storage == ArrayLocation.AUTOMATIC

    entry_bool = schema.tree["val_bool"]
    assert isinstance(entry_bool, ArrayEntry)
    assert entry_bool.dtype == np.dtype(bool)
    assert entry_bool.storage == ArrayLocation.AUTOMATIC


def test_schema_creation_nested():
    parent_schema = NestedParent._scipion_bridge_schema
    assert isinstance(parent_schema, Schema)
    assert set(parent_schema.tree.keys()) == {"name_id", "child"}

    assert isinstance(parent_schema.tree["name_id"], ArrayEntry)
    assert parent_schema.tree["name_id"].dtype == np.dtype(int)

    child_tree = parent_schema.tree["child"]
    assert isinstance(child_tree, dict)
    assert set(child_tree.keys()) == {"x", "y"}
    assert child_tree["x"] == ArrayEntry(np.dtype(int), ArrayLocation.AUTOMATIC)
    assert child_tree["y"] == ArrayEntry(np.dtype(np.float64), ArrayLocation.AUTOMATIC)


def test_schema_creation_deep_nested():
    deep_schema = DeepNested._scipion_bridge_schema
    assert isinstance(deep_schema, Schema)
    assert set(deep_schema.tree.keys()) == {"parent", "tag"}
    assert isinstance(deep_schema.tree["tag"], ArrayEntry)

    parent_tree = deep_schema.tree["parent"]
    assert isinstance(parent_tree, dict)
    assert set(parent_tree.keys()) == {"name_id", "child"}
    assert isinstance(parent_tree["child"], dict)
    assert set(parent_tree["child"].keys()) == {"x", "y"}

def test_untyped_attribute_raises_type_error():
    with pytest.raises(TypeError) as exc_info:

        class UntypedStruct(B.Struct):
            x = 10

    err_msg = str(exc_info.value)
    assert "declares attributes without type annotations." in err_msg
    assert "UntypedStruct" in err_msg


def test_incompatible_attribute_single_raises_type_error():
    with pytest.raises(TypeError) as exc_info:

        class SingleIncompatibleStruct(B.Struct):
            bad_attr: object

    err_msg = str(exc_info.value)
    assert "The attribute 'bad_attr' cannot be declared in struct" in err_msg
    assert "SingleIncompatibleStruct" in err_msg
    assert "because it does not support array serialization." in err_msg
