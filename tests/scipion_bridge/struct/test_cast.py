"""Casting Sets and Structs to other types without copying their columns."""

import pickle

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.struct import identity, references
from scipion_bridge.core.struct.id_column import IdColumn

from scipion_bridge.core.struct.references import PicklingMode
from scipion_bridge.core.struct.source import ArraySource, Reference


class CTF(B.Struct):
    defocus: float
    angle: float


class Image(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(4, 4))
    ctf: CTF
    sampling_rate: float


class ScoredImage(Image):
    score: float


class Thumbnail(B.Struct):
    """Unrelated to Image, but shares its pixels."""

    pixels: B.Array[np.float32] = B.Array(shape=(4, 4))
    label: int


class CountingSource(ArraySource):
    """An in-memory source that counts its reads."""

    def __init__(self, array):
        super().__init__(array)
        self.reads = 0

    def _read(self, index):
        self.reads += 1
        return super()._read(index)


def _rows(n):
    return np.arange(n * 16, dtype=np.float32).reshape(n, 4, 4)


def _images(n=6, reference=False):
    """A Set of images; its pixels reference a source if ``reference``."""
    s = B.Set[Image](capacity=n)
    s["ctf"]["defocus"] = np.arange(n, dtype=np.float64)
    s["sampling_rate"] = np.full(n, 1.5)
    source = CountingSource(_rows(n))
    if reference:
        references.set_reference(s, ("pixels",), Reference(source, range(n)))
    else:
        s["pixels"] = _rows(n)
    return s, source


def _values(s, *paths):
    return {path: np.asarray(_read(s, path)) for path in paths}


def _read(s, path):
    value = s
    for name in path.split("."):
        value = value[name]
    return value


# -- Sharing columns -----------------------------------------------------------
def test_a_cast_shares_the_columns_and_adds_uninitialized_fields():
    s, _ = _images()

    scored = B.astype(s, B.Set[ScoredImage])

    assert isinstance(scored, B.Set[ScoredImage]) and len(scored) == len(s)
    assert scored.capacity == s.capacity
    np.testing.assert_array_equal(scored["pixels"], s["pixels"])
    np.testing.assert_array_equal(scored["ctf"]["defocus"], s["ctf"]["defocus"])
    assert np.shares_memory(scored["pixels"], s["pixels"])
    assert not scored["pixels"].flags.writeable
    assert ("score",) not in references.column_states(scored)
    # Uninitialized fields stay so.
    assert ("ctf", "angle") not in references.column_states(scored)


def test_fields_the_target_lacks_are_dropped():
    s, _ = _images()
    scored = B.astype(s, B.Set[ScoredImage])
    scored["score"] = np.ones(len(s))

    back = B.astype(scored, B.Set[Image])

    assert set(references.column_states(back)) == {
        ("pixels",),
        ("ctf", "defocus"),
        ("sampling_rate",),
    }


def test_types_outside_each_others_hierarchy_cast_by_their_common_fields():
    s, _ = _images()

    thumbnails = B.astype(s, B.Set[Thumbnail])

    assert set(references.column_states(thumbnails)) == {("pixels",)}
    np.testing.assert_array_equal(thumbnails["pixels"], s["pixels"])


# -- References ----------------------------------------------------------------
def test_unread_references_stay_unread():
    s, source = _images(reference=True)

    scored = B.astype(s, B.Set[ScoredImage])

    assert source.reads == 0
    assert references.column_states(scored)[("pixels",)].state == "reference"
    np.testing.assert_array_equal(scored["pixels"], _rows(6))
    assert source.reads == 1


def test_read_references_stay_references_with_their_digests():
    s, source = _images(reference=True)
    references.prefetch(s)

    scored = B.astype(s, B.Set[ScoredImage])

    states = references.column_states(scored)
    assert states[("pixels",)] == references.column_states(s)[("pixels",)]
    assert states[("pixels",)].state == "cached"
    (reference,) = scored.storage.references().values()
    (original,) = s.storage.references().values()
    np.testing.assert_array_equal(reference.digests, original.digests)
    assert source.reads == 1


@pytest.mark.parametrize(
    "select",
    [
        lambda s: s[2:5],
        lambda s: s[np.array([True, False, True, True, False, False])],
        lambda s: s[[3, 0]],
    ],
)
def test_views_are_cast_with_their_rows(select):
    s, _ = _images(reference=True)
    references.prefetch(s)
    eager, _ = _images()
    view = select(s)

    scored = B.astype(view, B.Set[ScoredImage])

    assert len(scored) == len(view)
    paths = ("pixels", "ctf.defocus", "sampling_rate")
    expected = _values(select(eager), *paths)
    for path, value in _values(scored, *paths).items():
        np.testing.assert_array_equal(value, expected[path])
    (reference,) = scored.storage.references().values()
    assert len(reference) == len(view)


def test_an_element_of_a_set_is_cast_to_a_struct():
    s, _ = _images()

    element = B.astype(s[3], ScoredImage)

    assert isinstance(element, ScoredImage)
    np.testing.assert_array_equal(element.pixels, _rows(6)[3])
    assert element.ctf.defocus == 3.0
    assert not element.is_initialized("score")


def test_structs_cast_to_structs():
    image = Image(
        pixels=np.ones((4, 4), dtype=np.float32),
        ctf=CTF(defocus=2.0, angle=0.5),
        sampling_rate=1.0,
    )

    scored = B.astype(image, ScoredImage)

    assert (scored.ctf.defocus, scored.ctf.angle, scored.sampling_rate) == (
        2.0,
        0.5,
        1.0,
    )
    np.testing.assert_array_equal(scored.pixels, image.pixels)


class Member(B.Struct):
    value: float


class Group(B.Struct):
    members: B.Set[Member]
    group_id: int


class LabelledGroup(Group):
    label: int


def test_nested_sets_are_shared_with_their_rows():
    groups = B.Set[Group](
        [
            Group(
                members=B.Set[Member]([Member(value=10 * g + k) for k in range(n)]),
                group_id=g,
            )
            for g, n in enumerate([2, 3, 1])
        ],
    )

    labelled = B.astype(groups, B.Set[LabelledGroup])

    for g, n in enumerate([2, 3, 1]):
        values = np.asarray(labelled[g].members["value"]).ravel().tolist()
        assert values == [10.0 * g + k for k in range(n)]
    assert np.asarray(labelled["group_id"]).ravel().tolist() == [0, 1, 2]


# -- Rejected casts ------------------------------------------------------------
class OtherPixels(B.Struct):
    pixels: B.Array[np.float64] = B.Array(shape=(4, 4))


class OtherShape(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(8, 8))


class FlatCTF(B.Struct):
    ctf: float


class SetOfMembers(B.Struct):
    members: B.Set[Member]


class StructOfMembers(B.Struct):
    members: Member


class Unrelated(B.Struct):
    weight: float


@pytest.mark.parametrize(
    "target, message",
    [
        (
            B.Set[OtherPixels],
            "'pixels' is float32 of shape \\[4, 4\\] in one type and float64",
        ),
        (
            B.Set[OtherShape],
            "'pixels' is float32 of shape \\[4, 4\\] in one type and float32 of shape \\[8, 8\\]",
        ),
        (B.Set[FlatCTF], "'ctf' is a nested Struct in one type and a leaf"),
        (B.Set[Unrelated], "share no fields"),
        (Image, "a Set casts to a Set type"),
    ],
)
def test_casts_that_would_convert_values_are_rejected(target, message):
    s, _ = _images()

    with pytest.raises(TypeError, match=message):
        B.astype(s, target)


def test_a_nested_set_does_not_cast_to_a_nested_struct():
    s = B.Set[SetOfMembers](capacity=1)

    with pytest.raises(
        TypeError, match="'members' is a nested Set in one type and a nested Struct"
    ):
        B.astype(s, B.Set[StructOfMembers])


def test_targets_must_be_set_types_or_structs():
    s, _ = _images()

    for target in (B.Set, int, 3):
        with pytest.raises(TypeError, match="Set type .* or a Struct"):
            B.astype(s, target)


def test_a_view_of_the_members_of_every_group_is_rejected():
    groups = B.Set[Group](
        [Group(members=B.Set[Member]([Member(value=1.0)]), group_id=0)]
    )

    with pytest.raises(TypeError, match="index the outer Set first"):
        B.astype(groups["members"], B.Set[Member])


class Classes(B.Struct):
    classes: B.Collection[Image] = B.Collection(size=2)


class MoreClasses(B.Struct):
    classes: B.Collection[Image] = B.Collection(size=3)


class LabelledClasses(Classes):
    label: int


def test_collection_fields_of_the_same_size_are_shared():
    classes = Classes()
    classes.classes[1] = Image(
        pixels=np.ones((4, 4), dtype=np.float32),
        ctf=CTF(defocus=2.0, angle=0.0),
        sampling_rate=1.0,
    )

    labelled = B.astype(classes, LabelledClasses)

    assert labelled.classes[1].ctf.defocus == 2.0
    assert not labelled.classes.is_initialized(0)


def test_collection_fields_of_another_size_are_rejected():
    with pytest.raises(TypeError, match="Collection of size 2 in one type and 3"):
        B.astype(Classes(), MoreClasses)


def test_casting_collections_is_not_implemented():
    collection = B.Collection[Image](size=2)

    with pytest.raises(NotImplementedError):
        B.astype(collection, ScoredImage)
    with pytest.raises(NotImplementedError):
        B.astype(_images()[0], B.Collection[Image])


# -- No aliasing ---------------------------------------------------------------
@pytest.mark.parametrize(
    "write",
    [
        lambda r: r.__setitem__("sampling_rate", np.zeros(6)),
        lambda r: r["ctf"].__setitem__("defocus", np.zeros(6)),
        lambda r: r[1:3].__setitem__("pixels", np.zeros((2, 4, 4), dtype=np.float32)),
        lambda r: r.__setitem__(
            0,
            ScoredImage(
                pixels=np.zeros((4, 4), dtype=np.float32),
                ctf=CTF(defocus=0.0, angle=0.0),
                sampling_rate=0.0,
                score=0.0,
            ),
        ),
    ],
)
def test_writing_to_a_cast_leaves_the_input_unchanged(write):
    s, _ = _images()
    before = _values(s, "pixels", "ctf.defocus", "sampling_rate")

    write(B.astype(s, B.Set[ScoredImage]))

    for path, value in _values(s, "pixels", "ctf.defocus", "sampling_rate").items():
        np.testing.assert_array_equal(value, before[path])


def test_writing_in_place_into_a_shared_buffer_raises():
    s, _ = _images()
    scored = B.astype(s, B.Set[ScoredImage])

    with pytest.raises(ValueError, match="read-only"):
        scored["pixels"][0] = 1.0


def test_writing_into_a_reference_makes_it_owned_in_the_cast_only():
    s, _ = _images(reference=True)
    references.prefetch(s)
    scored = B.astype(s, B.Set[ScoredImage])

    scored[1:3]["pixels"] = np.zeros((2, 4, 4), dtype=np.float32)

    assert references.column_states(scored)[("pixels",)].state == "owned"
    assert references.column_states(s)[("pixels",)].state == "cached"
    np.testing.assert_array_equal(s["pixels"], _rows(6))


# -- Pickling ------------------------------------------------------------------
def test_casts_pickle_with_their_references():
    s, _ = _images(reference=True)
    references.prefetch(s)
    scored = B.astype(s, B.Set[ScoredImage])

    transferred = pickle.loads(pickle.dumps(scored))
    with references.pickling(PicklingMode.SPILL):
        spilled = pickle.dumps(scored)
    restored = pickle.loads(spilled)

    assert references.column_states(transferred)[("pixels",)].state == "cached"
    assert references.column_states(restored)[("pixels",)].state == "reference"
    np.testing.assert_array_equal(restored["pixels"], _rows(6))
    assert len(spilled) < len(pickle.dumps(scored))


# -- Dropping fields -------------------------------------------------------------
class ClassMember(B.Struct):
    """The members of a class: their CTF, without their pixels."""

    ctf: CTF


def test_dropping_the_pixels_keeps_the_ids_and_the_other_fields():
    s, _ = _images(64)
    identity.set_ids(s, IdColumn.of(b"F" * 16, range(64)))

    members = B.astype(s, B.Set[ClassMember])

    assert _ids(members) == _ids(s)
    np.testing.assert_array_equal(members["ctf"]["defocus"], s["ctf"]["defocus"])
    restored = pickle.loads(pickle.dumps(members))
    assert len(pickle.dumps(members)) < _rows(64).nbytes
    assert _ids(restored) == _ids(s)


def _ids(s):
    return identity.element_ids(s).ids().tobytes()
