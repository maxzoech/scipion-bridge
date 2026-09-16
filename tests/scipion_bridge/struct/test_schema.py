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
from scipion_bridge.core.struct.key_path import KeyPath



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


def test_joint_tree_iter_set_and_struct():
    """Verify joint traversal of Set schema + Struct schema (element assignment)."""
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        embeddings: B.Array[float] = B.Array(shape=(8,))

    set_schema = B.Set[Particle].schema()
    struct_schema = Particle.schema()

    results = list(set_schema.tree_iter(struct_schema))
    assert len(results) == 2

    # 1. pixels
    path1, set_entry1, struct_entry1 = results[0]
    assert path1 == KeyPath(root=()).append("pixels")
    assert isinstance(set_entry1, ArraySetEntry)
    assert isinstance(struct_entry1, ArrayEntry)
    assert set_entry1.shape == (4, 4)
    assert struct_entry1.shape == (4, 4)

    # 2. embeddings
    path2, set_entry2, struct_entry2 = results[1]
    assert path2 == KeyPath(root=()).append("embeddings")
    assert isinstance(set_entry2, ArraySetEntry)
    assert isinstance(struct_entry2, ArrayEntry)
    assert set_entry2.shape == (8,)
    assert struct_entry2.shape == (8,)


def test_joint_tree_iter_set_and_set():
    """Verify joint traversal of Set schema + Set schema (subset assignment)."""
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        embeddings: B.Array[float] = B.Array(shape=(8,))

    target_schema = Particle.schema().to_set_schema(capacity=10)
    source_schema = Particle.schema().to_set_schema(capacity=5)

    results = list(target_schema.tree_iter(source_schema))
    assert len(results) == 2

    path1, target_entry1, source_entry1 = results[0]
    assert path1 == KeyPath(root=()).append("pixels")
    assert isinstance(target_entry1, ArraySetEntry)
    assert isinstance(source_entry1, ArraySetEntry)
    assert target_entry1.capacity == 10
    assert source_entry1.capacity == 5

    path2, target_entry2, source_entry2 = results[1]
    assert path2 == KeyPath(root=()).append("embeddings")
    assert isinstance(target_entry2, ArraySetEntry)
    assert isinstance(source_entry2, ArraySetEntry)


def test_joint_tree_iter_nested():
    """Verify joint traversal through nested Struct and Set schemas."""
    class Header(B.Struct):
        version: int

    class Frame(B.Struct):
        header: Header
        pixels: B.Array[float] = B.Array(shape=(16, 16))

    set_schema = B.Set[Frame].schema()
    struct_schema = Frame.schema()

    results = list(set_schema.tree_iter(struct_schema))
    assert len(results) == 2

    by_path = {p.path: (p, set_e, struct_e) for p, set_e, struct_e in results}

    # header.version
    assert ("header", "version") in by_path
    p_header, set_v, struct_v = by_path[("header", "version")]
    assert p_header == KeyPath(root=()).append("header").append("version")
    assert isinstance(set_v, ArraySetEntry)
    assert isinstance(struct_v, ArrayEntry)

    # pixels
    assert ("pixels",) in by_path
    p_pixels, set_p, struct_p = by_path[("pixels",)]
    assert p_pixels == KeyPath(root=()).append("pixels")
    assert isinstance(set_p, ArraySetEntry)
    assert isinstance(struct_p, ArrayEntry)


def test_joint_tree_iter_strict_errors():
    """Verify strict validation raises ValueError on key mismatch and TypeError on structural mismatch."""
    class Foo(B.Struct):
        a: int
        b: float

    class BarMissing(B.Struct):
        a: int

    class BarExtra(B.Struct):
        a: int
        b: float
        c: int

    class BarBranchMismatch(B.Struct):
        class Child(B.Struct):
            x: int
        a: Child
        b: float

    # Key mismatch (missing field)
    with pytest.raises(ValueError, match="Schema field mismatch"):
        list(Foo.schema().tree_iter(BarMissing.schema()))

    # Key mismatch (extra field)
    with pytest.raises(ValueError, match="Schema field mismatch"):
        list(Foo.schema().tree_iter(BarExtra.schema()))

    # Structural mismatch (branch vs leaf)
    with pytest.raises(TypeError, match="Structural mismatch at field 'a'"):
        list(Foo.schema().tree_iter(BarBranchMismatch.schema()))


def test_joint_tree_iter_three_schemas():
    """Verify tree_iter with N > 1 additional schemas."""
    class Item(B.Struct):
        data: int

    s1 = Item.schema()
    s2 = Item.schema()
    s3 = B.Set[Item].schema()

    results = list(s1.tree_iter(s2, s3))
    assert len(results) == 1
    path, e1, e2, e3 = results[0]
    assert path == KeyPath(root=()).append("data")
    assert isinstance(e1, ArrayEntry)
    assert isinstance(e2, ArrayEntry)
    assert isinstance(e3, ArraySetEntry)