"""Arrow-based pickling shared by Set, Struct and Collection."""

import pickle

import awkward as ak
import cloudpickle
import numpy as np
import pyarrow as pa
import pytest
import ray
import ray.cloudpickle

import scipion_bridge as B
from scipion_bridge.single_particle.particle import Class2D, Particle


class Frame(B.Struct):
    pixels: B.Array[float] = B.Array(shape=(2, 2))
    frame_id: int


class Movie(B.Struct):
    frames: B.Set[Frame] = B.Set[Frame](capacity=3)
    movie_id: int


_SERIALIZERS = {
    "pickle": (pickle.dumps, pickle.loads),
    "cloudpickle": (cloudpickle.dumps, cloudpickle.loads),
    "ray.cloudpickle": (ray.cloudpickle.dumps, ray.cloudpickle.loads),
    "ray.put": (ray.put, ray.get),
}


def _particles(n: int, offset: float) -> B.Set[Particle]:
    return B.Set[Particle](
        [Particle(pixels=np.full((4, 4), offset + i, np.float32)) for i in range(n)],
    )


def _class(class_id: int, n: int, representative: bool = True) -> Class2D:
    cls = Class2D(class_id=class_id, particles=_particles(n, 10.0 * class_id))
    if representative:
        cls.representative = Particle(pixels=np.ones((4, 4), np.float32) * class_id)
    return cls


def _classes(size: int, filled: dict) -> B.Collection[Class2D]:
    collection = B.Collection[Class2D](size=size)
    for index, cls in filled.items():
        collection[index] = cls
    return collection


def _assert_class_equal(actual: Class2D, expected: Class2D) -> None:
    assert actual.initialized_fields() == expected.initialized_fields()
    assert actual.class_id == expected.class_id
    assert len(actual.particles) == len(expected.particles)
    assert np.array_equal(
        np.asarray(actual.particles["pixels"]),
        np.asarray(expected.particles["pixels"]),
    )
    if expected.is_initialized("representative"):
        assert np.array_equal(
            actual.representative.pixels,
            expected.representative.pixels,
        )


# -- Struct ------------------------------------------------------------------


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_owning_struct_roundtrip(serializer):
    dumps, loads = _SERIALIZERS[serializer]
    cls = _class(3, n=2)

    restored = loads(dumps(cls))

    assert type(restored) is Class2D
    _assert_class_equal(restored, cls)
    restored.class_id = 4  # The restored storage is writable.
    assert restored.class_id == 4


def test_struct_with_uninitialized_fields_roundtrip():
    restored = pickle.loads(pickle.dumps(Class2D(class_id=7)))

    assert restored.initialized_fields() == ["class_id"]
    assert restored.class_id == 7


def test_regular_array_fields_are_restored_as_numpy():
    restored = pickle.loads(pickle.dumps(_class(1, n=2)))

    assert isinstance(restored.representative.pixels, np.ndarray)


def test_struct_with_nested_set_of_static_rows_roundtrip():
    movie = Movie(movie_id=5)
    movie.frames = B.Set[Frame](
        [Frame(pixels=np.full((2, 2), i, float), frame_id=i) for i in range(3)],
    )

    restored = pickle.loads(pickle.dumps(movie))

    assert restored.movie_id == 5
    assert np.array_equal(restored.frames["pixels"], movie.frames["pixels"])


def test_set_element_view_roundtrip():
    particles = _particles(50, 0.0)

    restored = pickle.loads(pickle.dumps(particles[7]))

    assert not restored.storage.is_view
    assert np.array_equal(restored.pixels, np.full((4, 4), 7.0, np.float32))


def test_struct_to_arrow_is_a_single_row():
    batch = _class(2, n=3).to_arrow()

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 1


# -- Collection --------------------------------------------------------------


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_sparse_collection_roundtrip(serializer):
    dumps, loads = _SERIALIZERS[serializer]
    collection = _classes(
        5, {1: _class(1, n=3), 3: _class(3, n=2, representative=False)}
    )

    restored = loads(dumps(collection))

    assert type(restored) is B.Collection[Class2D]
    assert restored.size == 5
    assert restored.initialized_indices() == [1, 3]
    _assert_class_equal(restored[1], collection[1])
    _assert_class_equal(restored[3], collection[3])
    assert not restored[3].is_initialized("representative")


def test_empty_collection_keeps_its_size():
    restored = pickle.loads(pickle.dumps(B.Collection[Class2D](size=4)))

    assert restored.size == 4
    assert restored.initialized_indices() == []


def test_collection_with_instance_element_type_roundtrip():
    collection = B.Collection(size=2, dtype=Class2D)
    collection[0] = _class(0, n=1)

    restored = pickle.loads(pickle.dumps(collection))

    assert restored.dtype is Class2D
    _assert_class_equal(restored[0], collection[0])


def test_collection_to_arrow_has_one_row_per_slot():
    batch = _classes(5, {2: _class(2, n=1)}).to_arrow()

    batch.validate(full=True)
    assert batch.num_rows == 5
    assert batch.column("class_id").is_valid().to_pylist() == [
        False,
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_collection_item_view_roundtrip(serializer):
    dumps, loads = _SERIALIZERS[serializer]
    collection = _classes(3, {0: _class(0, n=2), 2: _class(2, n=4)})

    restored = loads(dumps(collection[2]))

    assert type(restored) is Class2D
    assert not restored.storage.is_view
    _assert_class_equal(restored, collection[2])


def test_collection_item_view_does_not_pickle_other_slots():
    small = _classes(20, {0: _class(0, n=2)})
    large = _classes(
        20, {0: _class(0, n=2), **{i: _class(i, n=200) for i in range(1, 20)}}
    )

    assert len(pickle.dumps(large[0])) == len(pickle.dumps(small[0]))
    assert len(pickle.dumps(large[0])) < len(pickle.dumps(large)) / 50


# Awkward ignores the offset of the validity bitmap of sliced Arrow arrays,
# reading valid rows after a null row as missing and null rows after a valid
# row as present. Restoring a Collection converts whole columns, so slot
# patterns mixing empty and filled slots must restore exactly.
@pytest.mark.parametrize(
    "slots",
    [
        [False, True],
        [True, False],
        [False, False, True, False],
        [True, False, True, False, True],
    ],
    ids=["empty-filled", "filled-empty", "only-third", "alternating"],
)
@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_collection_slot_patterns_roundtrip(slots, serializer):
    dumps, loads = _SERIALIZERS[serializer]
    collection = _classes(
        len(slots),
        {
            index: _class(index, n=index + 1)
            for index, filled in enumerate(slots)
            if filled
        },
    )

    restored = loads(dumps(collection))

    assert restored.initialized_indices() == collection.initialized_indices()
    for index in collection.initialized_indices():
        _assert_class_equal(restored[index], collection[index])


@pytest.mark.parametrize(
    "with_representative",
    [
        [False, True, False],
        [True, False, True],
    ],
    ids=["missing-present-missing", "present-missing-present"],
)
def test_collection_field_patterns_roundtrip(with_representative):
    # A field missing in one slot is a null row in its column; neighbouring
    # slots must keep their own state of that field.
    collection = _classes(
        len(with_representative),
        {
            index: _class(index, n=2, representative=present)
            for index, present in enumerate(with_representative)
        },
    )

    restored = pickle.loads(pickle.dumps(collection))

    assert [
        restored[index].is_initialized("representative")
        for index in range(len(with_representative))
    ] == with_representative
    for index in range(len(with_representative)):
        _assert_class_equal(restored[index], collection[index])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "awkward ignores the validity bitmap offset of sliced Arrow arrays; "
        "when this passes, the bug is fixed upstream."
    ),
)
def test_awkward_reads_validity_of_sliced_arrow_arrays():
    column = pa.array([None, 1.0, None, 2.0], type=pa.float32())

    rows = [ak.to_list(ak.from_arrow(column.slice(i, 1)))[0] for i in range(4)]

    assert rows == column.to_pylist()


# -- Set ---------------------------------------------------------------------


def test_set_slice_roundtrip_does_not_include_parent_rows():
    particles = _particles(200, 0.0)

    restored = pickle.loads(pickle.dumps(particles[10:12]))

    assert len(restored) == 2
    assert np.array_equal(restored[0].pixels, np.full((4, 4), 10.0, np.float32))
    assert len(pickle.dumps(particles[10:12])) < 0.15 * len(pickle.dumps(particles))


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_ragged_set_columns_keep_their_type(serializer):
    # PyArrow's pickling drops the nullability of nested fields; without
    # restoring it, regular rows would come back as Awkward option types.
    dumps, loads = _SERIALIZERS[serializer]
    particles = _particles(3, 0.0)
    pixels = particles["pixels"]

    restored = loads(dumps(particles))

    assert ak.type(restored["pixels"]) == ak.type(pixels)
    assert type(restored[1].pixels) is type(particles[1].pixels)
    assert np.array_equal(restored[1].pixels, np.full((4, 4), 1.0, np.float32))


def test_uninitialized_rows_stay_missing():
    particles = B.Set[Particle](capacity=3)
    particles[0] = Particle(pixels=np.ones((4, 4), np.float32))
    particles[2] = Particle(pixels=np.zeros((2, 2), np.float32))

    restored = pickle.loads(pickle.dumps(particles))

    assert ak.to_list(restored["pixels"]) == ak.to_list(particles["pixels"])
    assert ak.type(restored["pixels"]) == ak.type(particles["pixels"])


# -- Descriptors -------------------------------------------------------------


@pytest.mark.parametrize(
    "descriptor",
    [Class2D.particles, Class2D.representative, Particle.pixels, Movie.frames],
)
def test_field_descriptors_are_not_data(descriptor):
    assert descriptor.is_descriptor


def test_data_instances_are_not_descriptors():
    assert not _particles(1, 0.0).is_descriptor
    assert not _class(0, n=1).is_descriptor
    assert not B.Collection[Class2D](size=1).is_descriptor


def test_set_descriptor_keeps_default_pickling():
    restored = pickle.loads(pickle.dumps(Class2D.particles))

    assert restored.name == "particles"
    assert restored.is_descriptor
