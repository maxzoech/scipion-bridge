"""Element ids of Sets and Structs: the hidden id column (in-process)."""

import pickle

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.struct import concat, identity, references
from scipion_bridge.core.struct.cast import cast
from scipion_bridge.core.struct.id_column import IdColumn, tag_column
from scipion_bridge.core.struct.references import PicklingMode
from scipion_bridge.core.struct.source import ArraySource, Reference

F = b"F" * 16
G = b"G" * 16


class Image(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(4, 4))
    defocus: float


class ScoredImage(Image):
    score: float


class Group(B.Struct):
    members: B.Set[Image]
    group_id: int


class CountingSource(ArraySource):
    def __init__(self, array):
        super().__init__(array)
        self.reads = 0

    def _read(self, index):
        self.reads += 1
        return super()._read(index)


def _rows(n):
    return np.arange(n * 16, dtype=np.float32).reshape(n, 4, 4)


def _images(n=6, namespace=F, reference=False):
    """A Set of images with ids ``(namespace, 0..n-1)``."""
    s = B.Set[Image](capacity=n)
    s["defocus"] = np.arange(n, dtype=np.float64)
    source = CountingSource(_rows(n))
    if reference:
        references.set_reference(s, ("pixels",), Reference(source, range(n)))
    else:
        s["pixels"] = _rows(n)
    if namespace is not None:
        identity.set_ids(s, IdColumn.of(namespace, range(n)))
    return s, source


def _ids(obj):
    column = identity.element_ids(obj)
    return None if column is None else column.elements()


def _f(*positions):
    return [(F, p) for p in positions]


# -- The id column -------------------------------------------------------------
def test_segments_stay_compact_through_views_and_concat():
    column = IdColumn.concat([IdColumn.of(F, range(0, 4)), IdColumn.of(F, range(4, 8))])

    assert len(column.segments) == 1
    assert column.narrow(slice(2, 6)).elements() == _f(2, 3, 4, 5)
    assert column.narrow(np.array([7, 0])).elements() == _f(7, 0)
    assert column.narrow(np.arange(8) % 2 == 0).elements() == _f(0, 2, 4, 6)
    assert column.narrow(-1).elements() == _f(7)


def test_assigning_rows_grows_the_column_with_unassigned_elements():
    column = IdColumn().assign(2, IdColumn.of(G, range(5, 6)))

    assert column.elements() == [None, None, (G, 5)]
    assert not column.assigned and column.any_assigned


def test_integer_tags_derive_namespaces_and_stay_segments():
    views = IdColumn.of(F, np.repeat(np.arange(3), 2))

    tagged = tag_column(views, "view", np.tile(np.arange(2), 3))

    elements = tagged.elements()
    assert len(set(elements)) == 6
    assert [position for _, position in elements] == [0, 0, 1, 1, 2, 2]
    assert len({namespace for namespace, _ in elements}) == 2
    assert tag_column(views, "view", np.tile(np.arange(2), 3)).elements() == elements


def test_other_tags_are_hashed_per_element():
    tagged = tag_column(IdColumn.of(F, range(3)), "seed", np.array([0.5, 0.1, 0.2]))

    assert tagged.segments[0].explicit is not None
    assert len(set(tagged.elements())) == 3


def _fragments(n_stacks, rows=3):
    """Short runs of ``n_stacks`` stacks: ``rows`` rows of each."""
    return [IdColumn.of(bytes([k]) * 16, range(rows)) for k in range(n_stacks)]


def _elements_of(columns):
    return [e for column in columns for e in column.elements()]


def test_short_runs_of_many_namespaces_are_one_mixed_segment():
    fragments = _fragments(100)

    column = IdColumn.concat(fragments)

    assert len(column.segments) == 1 and column.segments[0].mixed
    assert column.elements() == _elements_of(fragments)


def test_long_runs_stay_positional_between_mixed_segments():
    long_run = IdColumn.of(F, range(100))

    column = IdColumn.concat([*_fragments(5), long_run, *_fragments(5)])

    assert [s.mixed for s in column.segments] == [True, False, True]
    assert column.elements()[15:115] == long_run.elements()


def test_mixed_segments_narrow_assign_and_join():
    fragments = _fragments(10)
    column = IdColumn.concat(fragments)
    elements = column.elements()
    mask = np.arange(len(column)) % 3 == 0

    assert column.narrow(mask).elements() == [e for e, m in zip(elements, mask) if m]
    assert column.narrow(np.array([29, 0, 4])).elements() == [
        elements[29],
        elements[0],
        elements[4],
    ]
    assert IdColumn.concat([column, column]).elements() == elements * 2
    assigned = column.assign(slice(3, 6), IdColumn.of(G, range(3)))
    assert assigned.elements()[3:6] == [(G, 0), (G, 1), (G, 2)]
    assert len(assigned.segments) == 1


def test_equal_ids_have_equal_digests_however_they_are_stored():
    fragments = _fragments(10)
    mixed = IdColumn.concat(fragments)
    as_runs = IdColumn(tuple(s for f in fragments for s in f.segments))

    assert mixed.digest_parts() == as_runs.digest_parts()


def test_tags_of_mixed_segments_equal_the_tags_of_positional_ones():
    fragments = _fragments(10)
    values = np.arange(30) % 2

    mixed = tag_column(IdColumn.concat(fragments), "view", values)
    one_by_one = [
        tag_column(fragment, "view", values[3 * k : 3 * k + 3])
        for k, fragment in enumerate(fragments)
    ]

    assert mixed.elements() == _elements_of(one_by_one)


def test_mixed_segments_survive_pickling():
    column = IdColumn.concat(_fragments(10))

    assert pickle.loads(pickle.dumps(column)).elements() == column.elements()


# -- Ids follow the rows -------------------------------------------------------
@pytest.mark.parametrize(
    "select, expected",
    [
        (lambda s: s[2:5], _f(2, 3, 4)),
        (lambda s: s[np.array([True, False, True, False, False, True])], _f(0, 2, 5)),
        (lambda s: s[[3, 0]], _f(3, 0)),
        (lambda s: s[4], _f(4)),
        (lambda s: s[1:5][1:3], _f(2, 3)),
    ],
)
def test_views_select_the_ids_of_their_rows(select, expected):
    s, _ = _images()

    assert _ids(select(s)) == expected


def test_writing_values_keeps_the_ids():
    s, _ = _images()

    s["defocus"] = np.zeros(6)
    s[1:3]["pixels"] = np.zeros((2, 4, 4), dtype=np.float32)

    assert _ids(s) == _f(*range(6))


def test_concat_joins_ids_and_marks_sets_without_ids():
    a, _ = _images(3)
    b, _ = _images(2, namespace=None)

    joined = concat([a, b, a[1:]])

    assert _ids(joined) == [*_f(0, 1, 2), None, None, *_f(1, 2)]


def test_sets_without_ids_have_none():
    s, _ = _images(namespace=None)

    assert identity.element_ids(s) is None
    assert identity.element_ids(s[1:3]) is None


def test_a_view_of_the_members_of_every_group_has_no_single_column():
    groups = B.Set[Group]([Group(members=_images(2, namespace=None)[0], group_id=0)])
    identity.set_ids(groups, IdColumn.of(G, range(1)))

    with pytest.raises(NotImplementedError, match="index the outer Set first"):
        identity.element_ids(groups["members"])


# -- Row writes ----------------------------------------------------------------
def test_writing_rows_gives_them_the_ids_of_what_was_written():
    s, _ = _images()
    target, _ = _images(5, namespace=G)

    target[0] = s[3]
    target[1:3] = s[4:6]
    target[np.array([False, False, False, True, True])] = s[[1, 0]]

    assert _ids(target) == [(F, 3), (F, 4), (F, 5), (F, 1), (F, 0)]


def test_writing_rows_without_ids_unassigns_them():
    target, _ = _images(3, namespace=G)
    plain, _ = _images(3, namespace=None)

    target[1] = plain[0]

    assert _ids(target) == [(G, 0), None, (G, 2)]


def test_a_set_built_from_elements_keeps_their_ids():
    s, _ = _images()
    fresh = Image(pixels=np.zeros((4, 4), dtype=np.float32), defocus=1.0)

    built = B.Set[Image]([s[4], fresh, s[1]])

    assert _ids(built) == [(F, 4), None, (F, 1)]


# -- Copies that keep identity: Struct fields and Collection slots -------------
def test_a_set_in_a_struct_field_keeps_its_ids_and_references():
    s, source = _images(reference=True)
    mask = np.array([True, False, True, True, False, False])

    group = Group(members=s[mask], group_id=1)

    assert _ids(group.members) == _f(0, 2, 3)
    assert references.column_states(group)[("members", "pixels")].state == "reference"
    assert source.reads == 0
    np.testing.assert_array_equal(group.members["pixels"], _rows(6)[mask])


def test_read_references_stay_cached_references_in_a_struct_field():
    s, source = _images(reference=True)
    references.prefetch(s)

    group = Group(members=s[2:4], group_id=1)

    state = references.column_states(group)[("members", "pixels")]
    assert (state.state, state.rows) == ("cached", 2)
    assert source.reads == 1


def test_a_struct_in_a_collection_slot_keeps_ids_and_references():
    s, source = _images(reference=True)
    collection = B.Collection[Group](size=3)

    collection[1] = Group(members=s[[5, 1]], group_id=1)

    assert _ids(collection[1].members) == _f(5, 1)
    states = references.column_states(collection[1])
    assert states[("members", "pixels")].state == "reference"
    assert source.reads == 0
    np.testing.assert_array_equal(collection[1].members["pixels"], _rows(6)[[5, 1]])


def test_a_collection_from_items_keeps_ids():
    s, _ = _images()

    collection = B.Collection[Group](
        size=2,
        items=[Group(members=s[:2], group_id=0), Group(members=s[2:5], group_id=1)],
    )

    assert _ids(collection[0].members) == _f(0, 1)
    assert _ids(collection[1].members) == _f(2, 3, 4)


# -- The guard: ids of nested Sets below two axes ------------------------------
def test_a_set_of_groups_from_members_with_ids_raises():
    s, _ = _images()

    with pytest.raises(NotImplementedError, match="Collection"):
        B.Set[Group]([Group(members=s[:2], group_id=0)])

    groups = B.Set[Group](
        [Group(members=_images(2, namespace=None)[0], group_id=0)],
    )
    with pytest.raises(NotImplementedError, match="Collection"):
        groups[0] = Group(members=s[:2], group_id=0)
    with pytest.raises(NotImplementedError, match="Collection"):
        groups[0].members = s[:2]


def test_a_set_of_groups_from_members_without_ids_copies():
    members, _ = _images(3, namespace=None)

    groups = B.Set[Group]([Group(members=members, group_id=0)])

    np.testing.assert_array_equal(groups[0].members["defocus"].ravel(), [0, 1, 2])


# -- Casts, tags and pickling --------------------------------------------------
def test_casts_share_the_ids():
    s, _ = _images()

    scored = cast(s[1:4], B.Set[ScoredImage])

    assert _ids(scored) == _f(1, 2, 3)


def test_tagged_views_are_new_elements_linked_to_theirs():
    s, _ = _images(3)
    views = s[np.repeat(np.arange(3), 2)]

    tagged = views.tag("view", np.tile(np.arange(2), 3))

    elements = _ids(tagged)
    assert len(set(elements)) == 6
    assert [position for _, position in elements] == [0, 0, 1, 1, 2, 2]
    assert _ids(views) == _f(0, 0, 1, 1, 2, 2)
    np.testing.assert_array_equal(tagged["pixels"], views["pixels"])


def test_tagging_elements_without_ids_keeps_none():
    s, _ = _images(namespace=None)

    assert identity.element_ids(s.tag("view", 0)) is None


def test_a_struct_is_tagged_like_an_element():
    s, _ = _images()

    view = s[2].tag("view", 3)

    (element,) = _ids(view)
    assert element[1] == 2 and element[0] != F


def test_ids_rng_is_reproducible_per_element():
    s, _ = _images(3)

    first = [rng.random() for rng in B.ids_rng(s, "augment")]
    again = [rng.random() for rng in B.ids_rng(s, "augment")]
    other = [rng.random() for rng in B.ids_rng(s, "crop")]

    assert first == again
    assert len(set(first)) == 3
    assert first != other


def test_ids_rng_needs_ids():
    s, _ = _images(namespace=None)

    with pytest.raises(ValueError, match="seeded from element ids"):
        B.ids_rng(s, "augment")


@pytest.mark.parametrize("mode", [PicklingMode.TRANSFER, PicklingMode.SPILL])
def test_ids_survive_pickling(mode):
    s, _ = _images(reference=True)
    collection = B.Collection[Group](size=2)
    collection[0] = Group(members=s[[4, 2]], group_id=0)

    with references.pickling(mode):
        restored_view = pickle.loads(pickle.dumps(s[1:4]))
        restored_element = pickle.loads(pickle.dumps(s[5]))
        restored_collection = pickle.loads(pickle.dumps(collection))

    assert _ids(restored_view) == _f(1, 2, 3)
    assert _ids(restored_element) == _f(5)
    assert _ids(restored_collection[0].members) == _f(4, 2)


def test_collections_and_structs_pickle_their_references():
    s, source = _images(reference=True)
    references.prefetch(s)
    collection = B.Collection[Group](size=3)
    collection[2] = Group(members=s[[4, 2]], group_id=2)
    group = Group(members=s[1:3], group_id=1)

    with references.pickling(PicklingMode.SPILL):
        spilled_collection = pickle.dumps(collection)
        spilled_group = pickle.dumps(group)
    transferred = pickle.loads(pickle.dumps(collection))

    restored = pickle.loads(spilled_collection)
    state = references.column_states(restored[2])[("members", "pixels")]
    assert state.state == "reference"
    np.testing.assert_array_equal(restored[2].members["pixels"], _rows(6)[[4, 2]])
    restored_group = pickle.loads(spilled_group)
    np.testing.assert_array_equal(restored_group.members["pixels"], _rows(6)[1:3])
    assert (
        references.column_states(transferred[2])[("members", "pixels")].state
        == "cached"
    )
    assert len(spilled_collection) < len(pickle.dumps(collection))
