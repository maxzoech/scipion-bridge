"""Unit tests for Collection container, schema, descriptors, and Set interplay."""

import pytest

from scipion_bridge.core.struct import (
    Collection,
    CollectionEntry,
    SchemaEntry,
    Set,
    Struct,
)


class Point(Struct):
    x: float
    y: float


class MinorCluster(Struct):
    sub_id: int
    score: float


class MajorCluster(Struct):
    major_id: int
    subclusters: Collection[MinorCluster] = Collection[MinorCluster](size=3)


def test_collection_schema_and_caching() -> None:
    coll = Collection[Point](size=10)
    entry = coll.convert_to_entry()

    assert isinstance(entry, CollectionEntry)
    assert entry.size == 10
    assert entry.is_static is True

    # Check cached_property memoization
    children1 = entry.children
    children2 = entry.children
    assert children1 is not None
    assert children1 is children2
    assert len(children1.fields) == 10

    # Ensure all slots point to the exact same element entry (shared heap pointer)
    first_field = children1.fields["0"]
    for i in range(10):
        assert children1.fields[str(i)] is first_field

    # Check formatting
    formatted = entry.format_entry("points")
    assert formatted == "points: Collection[Point](size: 10)"


def test_collection_type_validation() -> None:
    # Non-Struct element types must raise TypeError
    with pytest.raises(TypeError, match="Element of Collection has to be a Struct"):
        _ = Collection[int](size=5)

    with pytest.raises(TypeError, match="Element of Collection has to be a Struct"):
        _ = Collection[Collection[Point]](size=5)  # type: ignore


def test_collection_crud_and_bounds() -> None:
    coll = Collection[Point](size=5)

    assert len(coll) == 5
    assert coll.initialized_indices() == []
    assert coll.is_initialized(0) is False
    assert (0 in coll) is False

    # Out of bounds
    with pytest.raises(IndexError):
        _ = coll[-1]

    with pytest.raises(IndexError):
        _ = coll[5]

    # Uninitialized access raises KeyError
    with pytest.raises(KeyError, match="Collection slot 0 is not initialized"):
        _ = coll[0]

    # Write items
    p0 = Point(x=1.5, y=2.5)
    coll[0] = p0

    assert coll.is_initialized(0) is True
    assert (0 in coll) is True
    assert coll.is_initialized(1) is False
    assert (1 in coll) is False

    read_p0 = coll[0]
    assert read_p0.x == 1.5
    assert read_p0.y == 2.5

    # Write another slot
    p3 = Point(x=10.0, y=20.0)
    coll[3] = p3

    assert coll.initialized_indices() == [0, 3]
    assert coll.keys() == [0, 3]

    items = list(coll)
    assert len(items) == 2
    assert items[0].x == 1.5
    assert items[1].x == 10.0

    item_dict = dict(coll.items())
    assert 0 in item_dict and 3 in item_dict
    assert item_dict[3].y == 20.0


def test_collection_constructor_items() -> None:
    p1 = Point(x=1.0, y=1.0)
    p3 = Point(x=3.0, y=3.0)

    # Initializing from dict
    coll_dict = Collection[Point](size=5, items={1: p1, 3: p3})
    assert coll_dict.initialized_indices() == [1, 3]
    assert coll_dict[1].x == 1.0
    assert coll_dict[3].y == 3.0

    # Initializing from sequence
    coll_seq = Collection[Point](size=4, items=[p1, p3])
    assert coll_seq.initialized_indices() == [0, 1]
    assert coll_seq[0].x == 1.0
    assert coll_seq[1].x == 3.0


def test_collection_to_set() -> None:
    coll = Collection[Point](size=10)
    coll[2] = Point(x=1.0, y=2.0)
    coll[7] = Point(x=3.0, y=4.0)

    s = coll.to_set()
    assert isinstance(s, Set)
    assert len(s) == 2
    assert s[0].x == 1.0
    assert s[0].y == 2.0
    assert s[1].x == 3.0
    assert s[1].y == 4.0

    # Empty collection to set
    empty_coll = Collection[Point](size=5)
    empty_set = empty_coll.to_set()
    assert len(empty_set) == 0


def test_struct_with_collection_descriptor() -> None:
    # Schema check
    schema = MajorCluster.schema()
    assert "major_id" in schema.fields
    assert "subclusters" in schema.fields
    sub_entry = schema.fields["subclusters"]
    assert isinstance(sub_entry, CollectionEntry)
    assert sub_entry.size == 3

    # Instance check
    cluster = MajorCluster(major_id=42)
    assert cluster.major_id == 42
    assert isinstance(cluster.subclusters, Collection)
    assert len(cluster.subclusters) == 3
    assert cluster.subclusters.initialized_indices() == []

    # Assign member in collection using standard indexing
    m0 = MinorCluster(sub_id=1, score=0.88)
    cluster.subclusters[0] = m0

    assert cluster.subclusters.is_initialized(0) is True
    assert cluster.subclusters[0].sub_id == 1
    assert cluster.subclusters[0].score == pytest.approx(0.88)

    # Assign entire collection to descriptor
    other_coll = Collection[MinorCluster](size=3)
    other_coll[1] = MinorCluster(sub_id=2, score=0.99)
    cluster.subclusters = other_coll

    assert cluster.subclusters.is_initialized(1) is True
    assert cluster.subclusters[1].sub_id == 2
    assert cluster.subclusters[1].score == pytest.approx(0.99)

    # Size mismatch on assignment
    wrong_size_coll = Collection[MinorCluster](size=5)
    with pytest.raises(ValueError, match="Collection size mismatch"):
        cluster.subclusters = wrong_size_coll


def test_set_of_structs_with_collection() -> None:
    # Set of MajorCluster where MajorCluster contains a Collection[MinorCluster]
    c1 = MajorCluster(major_id=1)
    c1.subclusters[0] = MinorCluster(sub_id=10, score=0.5)

    c2 = MajorCluster(major_id=2)
    c2.subclusters[0] = MinorCluster(sub_id=20, score=0.6)

    # Schema vectorization via CollectionEntry.to_set_entry()
    major_schema = MajorCluster.schema().to_set_schema(capacity=2)
    sub_coll_entry = major_schema.fields["subclusters"]
    assert isinstance(sub_coll_entry, CollectionEntry)
    assert sub_coll_entry.size == 3
    assert isinstance(sub_coll_entry.element_entry, SchemaEntry)

    # Set instantiation
    cluster_set = Set[MajorCluster]([c1, c2])
    assert len(cluster_set) == 2
    assert cluster_set[0].major_id == 1
    assert cluster_set[0].subclusters[0].sub_id == 10
    assert cluster_set[1].major_id == 2
    assert cluster_set[1].subclusters[0].sub_id == 20

    cluster_set.print_schema()


def test_hierarchical_collection_struct() -> None:
    # Collection of top-level MajorClusters, each containing Collection[MinorCluster]
    clusters = Collection[MajorCluster](size=2)

    assert len(clusters) == 2
    assert clusters.initialized_indices() == []

    # Assign top-level slot 0
    m_cluster = MajorCluster(major_id=10)
    m_cluster.subclusters[1] = MinorCluster(sub_id=101, score=0.77)
    clusters[0] = m_cluster

    assert clusters.is_initialized(0) is True
    assert clusters.is_initialized(1) is False

    # Read back using standard indexing
    read_major = clusters[0]
    assert read_major.major_id == 10
    assert read_major.subclusters.is_initialized(1) is True
    assert read_major.subclusters.is_initialized(0) is False
    assert read_major.subclusters[1].sub_id == 101
    assert read_major.subclusters[1].score == pytest.approx(0.77)

    # Mutate nested collection using standard indexing
    clusters[0].subclusters[2] = MinorCluster(sub_id=102, score=0.88)
    assert clusters[0].subclusters[2].sub_id == 102
    assert clusters[0].subclusters[2].score == pytest.approx(0.88)


def test_collection_schema_print_tree_single_struct_entry(
    capsys: pytest.CaptureFixture,
) -> None:
    schema = MajorCluster.schema()
    schema.print_tree()
    captured = capsys.readouterr().out

    # Verify that slot 0 is printed in full, followed by ellipsis and collapsed last slot
    assert "subclusters: Collection[MinorCluster](size: 3)" in captured
    assert "0 (struct)" in captured
    assert "..." in captured
    assert "2 (struct, collapsed)" in captured
    assert "sub_id" in captured
    assert "score" in captured


def test_standalone_collection_schema_and_item_type() -> None:
    # Test item_type classmethod
    assert Collection[MinorCluster].item_type() is MinorCluster
    with pytest.raises(AssertionError, match="Cannot retrieve item_type"):
        Collection.item_type()

    # Test class schema raises NotImplementedError
    with pytest.raises(NotImplementedError):
        Collection[MinorCluster].schema()


def test_collection_descriptor_self_assignment() -> None:
    class Cluster(Struct):
        classes: Collection[MinorCluster] = Collection(size=5)

    c = Cluster()
    c.classes[0] = MinorCluster(sub_id=1, score=0.5)
    c.classes[2] = MinorCluster(sub_id=3, score=0.9)
    assert c.classes.initialized_indices() == [0, 2]

    # Self-assignment must preserve data and not clear/destroy the collection
    c.classes = c.classes
    assert c.classes.initialized_indices() == [0, 2]
    assert c.classes[0].sub_id == 1
    assert c.classes[2].sub_id == 3


def test_collection_setitem_self_assignment() -> None:
    coll = Collection[MinorCluster](size=5)
    coll[0] = MinorCluster(sub_id=10, score=0.42)
    assert coll.is_initialized(0) is True

    # Assigning slot to itself must preserve data
    coll[0] = coll[0]
    assert coll.is_initialized(0) is True
    assert coll[0].sub_id == 10
    assert coll[0].score == pytest.approx(0.42)


if __name__ == "__main__":
    test_set_of_structs_with_collection()
