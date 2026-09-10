import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.struct.schema import (
    Schema,
    Entry,
    ArrayEntryBase,
    ArrayEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
    SchemaEntry,
    SchemaSetEntry,
)


def test_schema_construction_and_static_property():
    class Simple(B.Struct):
        val_int: int
        val_float: float
        val_bool: bool

    schema = Simple.schema()
    assert isinstance(schema, Schema)
    assert schema.dtype is Simple
    assert schema.is_static is True
    assert set(schema.fields.keys()) == {"val_int", "val_float", "val_bool"}

    int_entry = schema.fields["val_int"]
    assert isinstance(int_entry, ArrayEntry)
    assert int_entry.dtype == np.dtype(int)
    assert int_entry.shape == (1,)
    assert int_entry.is_static is True


def test_schema_nested_struct_and_set():
    class Child(B.Struct):
        x: float
        pixels = B.Array[float](shape=(32, 32))

    class Parent(B.Struct):
        tag: int
        child: Child
        items = B.Set[Child](capacity=10)

    schema = Parent.schema()
    assert schema.is_static is True
    assert set(schema.fields.keys()) == {"tag", "child", "items"}

    # StructEntry
    child_entry = schema.fields["child"]
    assert isinstance(child_entry, SchemaEntry)
    assert child_entry.is_static is True
    assert child_entry.children is not None
    assert set(child_entry.children.fields.keys()) == {"x", "pixels"}

    # SchemaSetEntry
    items_entry = schema.fields["items"]
    assert isinstance(items_entry, SchemaSetEntry)
    assert items_entry.capacity == 10
    assert items_entry.is_static is True
    assert items_entry.children is not None


def test_schema_dynamic_shape_is_not_static():
    class Dynamic(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, 32))

    assert Dynamic.schema().is_static is False
    pixels_entry = Dynamic.schema().fields["pixels"]
    assert isinstance(pixels_entry, ArrayEntry)
    assert pixels_entry.is_static is False


def test_schema_lookup_and_lookup_array():
    class SubChild(B.Struct):
        val: int
        tensor = B.Array[np.float32](shape=(4, 4))

    class Container(B.Struct):
        sub: SubChild
        items = B.Set[SubChild](capacity=5)

    schema = Container.schema()

    # 1. Lookup with root prefix
    assert schema.lookup(("root", "sub", "val")) is not None
    assert isinstance(schema.lookup_array(("root", "sub", "val")), ArrayEntryBase)

    # 2. Lookup without root prefix
    val_entry = schema.lookup(("sub", "val"))
    assert val_entry is not None
    assert val_entry.dtype == np.dtype(int)

    # 3. Lookup intermediate SchemaEntry
    sub_entry = schema.lookup(("sub",))
    assert isinstance(sub_entry, SchemaEntry)
    # lookup_array returns None for non-array entries
    assert schema.lookup_array(("sub",)) is None

    # 4. Lookup nested set field
    tensor_entry = schema.lookup_array(("items", "tensor"))
    assert isinstance(tensor_entry, ArraySetEntry)
    assert tensor_entry.shape == (4, 4)

    # 5. Nonexistent paths return None
    assert schema.lookup(("nonexistent",)) is None
    assert schema.lookup(("sub", "missing")) is None
    assert schema.lookup_array(("items", "missing")) is None
    assert schema.lookup(()) is None


def test_schema_tree_iter():
    class Leaf(B.Struct):
        a: int
        b: float

    class Root(B.Struct):
        leaf: Leaf
        tag: str

    leaves = dict(Root.schema().tree_iter())
    assert ("leaf", "a") in leaves
    assert ("leaf", "b") in leaves
    assert ("tag",) in leaves
    for path, entry in leaves.items():
        assert isinstance(entry, ArrayEntryBase)
        assert isinstance(path, tuple)


def test_schema_to_set_schema():
    class Item(B.Struct):
        val: float
        pixels = B.Array[float](shape=(16, 16))

    set_schema = Item.schema().to_set_schema(capacity=8)
    assert set_schema.capacity == 8
    assert set_schema.is_static is True

    val_entry = set_schema.fields["val"]
    assert isinstance(val_entry, ArraySetEntry)
    assert val_entry.capacity == 8

    pixels_entry = set_schema.fields["pixels"]
    assert isinstance(pixels_entry, ArraySetEntry)
    assert pixels_entry.shape == (16, 16)


def test_format_entry_formatting():
    entry_static = ArrayEntry(dtype=np.dtype("float32"), shape=(10, 10))
    formatted = entry_static.format_entry("image")
    # Verify no trailing unmatched closing parenthesis
    assert formatted.endswith("[10, 10]")
    assert "(" not in formatted
    assert ")" not in formatted