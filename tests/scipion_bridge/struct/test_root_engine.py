import pickle
from typing import Any

import awkward as ak
import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.struct.exceptions import UninitializedFieldError
from scipion_bridge.core.struct.key_path import KeyPath
from scipion_bridge.core.struct.schema import (
    ArraySetEntry,
    Entry,
    RaggedArraySetEntry,
)
from scipion_bridge.core.struct.storage import RootEngine, _BaseStorage
from scipion_bridge.single_particle.particle import Class2D, Particle


class Sample(B.Struct):
    image: B.Array[np.float32] = B.Array(shape=(2, 2))
    embedding: B.Array[float] = B.Array(shape=(None,))
    score: float


class Bucket(B.Struct):
    samples: B.Set[Sample]


def _samples(n: int) -> "B.Set[Sample]":
    return B.Set[Sample](
        [
            Sample(
                image=np.full((2, 2), i, dtype=np.float32),
                embedding=np.arange(i + 1, dtype=float),
                score=float(i),
            )
            for i in range(n)
        ],
    )


def _pending_samples() -> "B.Set[Sample]":
    """Three samples whose embedding column has a pending, unwritten row 1."""
    samples = B.Set[Sample]()
    samples["score"] = np.arange(3, dtype=float).reshape(3, 1)
    samples[0].embedding = np.zeros(2)
    samples[2].embedding = np.ones(1)
    return samples


def test_empty_mask_selection_reads_empty_columns():
    empty = _samples(4)[np.zeros(4, dtype=bool)]

    assert len(empty) == 0
    assert len(empty["embedding"]) == 0
    assert np.asarray(empty["image"]).shape == (0, 2, 2)


@pytest.mark.parametrize("mask", [[False] * 4, [True, False, True, False]])
def test_masked_selection_assigns_to_set_field(mask):
    # Mirrors Class2D(particles=particles[assignments == k]) for a class
    # without particles.
    bucket = Bucket(samples=_samples(4)[np.array(mask)])

    assert len(bucket.samples) == sum(mask)
    assert len(bucket.samples["embedding"]) == sum(mask)


def test_unwritten_element_raises_while_pending_and_after_materialization():
    samples = _pending_samples()

    with pytest.raises(UninitializedFieldError):
        _ = samples[1].embedding

    assert ak.to_list(samples["embedding"]) == [[0.0, 0.0], None, [1.0]]

    with pytest.raises(UninitializedFieldError):
        _ = samples[1].embedding


def test_empty_rows_write_to_static_field_is_padded_to_capacity():
    engine = RootEngine()
    key = KeyPath().append("image")

    engine.write(key, ArraySetEntry(dtype=np.dtype(np.float32), shape=(2, 2)), [])
    assert engine.read(
        key, ArraySetEntry(dtype=np.dtype(np.float32), shape=(2, 2))
    ).shape == (0, 2, 2)

    padded = ArraySetEntry(dtype=np.dtype(np.float32), shape=(2, 2), capacity=3)
    engine.write(key, padded, [])
    assert engine.read(key, padded).shape == (3, 2, 2)


def test_empty_rows_write_to_dynamic_field():
    engine = RootEngine()
    key = KeyPath().append("embedding")
    entry = RaggedArraySetEntry(dtype=np.dtype(float), shape=(None,))

    engine.write(key, entry, [])

    assert len(engine.read(key, entry)) == 0
    assert engine.get_length(KeyPath()) == 0


def test_full_column_length_is_checked_against_pending_sibling():
    engine = RootEngine()
    embedding = RaggedArraySetEntry(dtype=np.dtype(float), shape=(None,))
    score = ArraySetEntry(dtype=np.dtype(float), shape=(1,))

    engine.write(
        KeyPath().narrow_index(2, length=3).append("embedding"),
        embedding,
        np.zeros(2),
    )

    with pytest.raises(ValueError, match="Length mismatch"):
        engine.write(KeyPath().append("score"), score, np.ones((5, 1)))


def test_full_column_length_is_checked_against_capacity():
    engine = RootEngine()
    score = ArraySetEntry(dtype=np.dtype(float), shape=(1,), capacity=2)

    with pytest.raises(ValueError, match="exceeds capacity"):
        engine.write(KeyPath().append("score"), score, np.ones((3, 1)))


def test_storage_without_all_operations_cannot_be_instantiated():
    class ReadOnly(_BaseStorage):
        def read(self, key: KeyPath, entry: Entry) -> Any:
            return None

    with pytest.raises(TypeError, match="abstract"):
        ReadOnly()  # type: ignore[abstract]


def test_pickle_round_trip_of_empty_selection():
    empty = _samples(4)[np.zeros(4, dtype=bool)]

    restored = pickle.loads(pickle.dumps(empty, protocol=pickle.HIGHEST_PROTOCOL))

    assert len(restored) == 0
    assert len(restored["embedding"]) == 0
    assert np.asarray(restored["image"]).shape == (0, 2, 2)


@pytest.mark.parametrize("materialize", [False, True], ids=["pending", "masked-row"])
def test_pickle_round_trip_keeps_missing_row_and_stays_writable(materialize):
    samples = _pending_samples()
    if materialize:
        _ = samples["embedding"]

    restored = pickle.loads(pickle.dumps(samples, protocol=pickle.HIGHEST_PROTOCOL))

    with pytest.raises(UninitializedFieldError):
        _ = restored[1].embedding

    restored[1].embedding = np.full(2, 5.0)
    restored["score"] = np.full((3, 1), 7.0)

    assert ak.to_list(restored["embedding"]) == [[0.0, 0.0], [5.0, 5.0], [1.0]]
    assert np.all(restored["score"] == 7.0)


# -- Regular data in dynamic fields --------------------------------------------


def _particles(n: int) -> "B.Set[Particle]":
    particles = B.Set[Particle](capacity=n)
    particles["pixels"] = np.arange(n * 12, dtype=np.float32).reshape(n, 3, 4)
    return particles


def test_regular_rows_written_one_by_one_are_stored_as_numpy():
    particles = B.Set[Particle](capacity=3)
    for i in range(3):
        particles[i].pixels = np.full((3, 4), i, dtype=np.float32)

    pixels = particles["pixels"]

    assert isinstance(pixels, np.ndarray)
    assert pixels.shape == (3, 3, 4)
    assert pixels.dtype == np.float32
    assert pixels[:, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_row_of_another_shape_switches_the_column_to_awkward():
    particles = _particles(3)

    particles[1].pixels = np.zeros((2, 2), dtype=np.float32)

    pixels = particles["pixels"]
    assert isinstance(pixels, ak.Array)
    assert [np.asarray(row).shape for row in pixels] == [(3, 4), (2, 2), (3, 4)]


def test_regular_awkward_column_write_is_stored_as_numpy():
    particles = B.Set[Particle](capacity=2)

    particles["pixels"] = ak.Array(np.ones((2, 3, 4), dtype=np.float32))

    assert isinstance(particles["pixels"], np.ndarray)


def test_regular_rows_written_as_a_list_are_stored_as_numpy():
    particles = B.Set[Particle]([Particle(pixels=np.ones((3, 4), np.float32))] * 2)

    assert isinstance(particles["pixels"], np.ndarray)
    assert np.asarray(particles["pixels"]).shape == (2, 3, 4)


def test_missing_rows_keep_the_column_awkward():
    samples = _pending_samples()

    embedding = samples["embedding"]

    assert isinstance(embedding, ak.Array)
    assert ak.to_list(embedding) == [[0.0, 0.0], None, [1.0]]
    with pytest.raises(UninitializedFieldError):
        _ = samples[1].embedding


@pytest.mark.parametrize(
    "index",
    [slice(1, 3), np.array([0, 2]), np.array([True, False, True])],
    ids=["slice", "int-array", "bool-mask"],
)
def test_selection_of_numpy_column_reads_numpy(index):
    particles = _particles(3)

    selected = particles[index]["pixels"]

    assert isinstance(selected, np.ndarray)
    assert selected.dtype == np.float32
    assert np.array_equal(selected, np.asarray(particles["pixels"])[index])


def test_masked_selection_assigned_to_class2d_stays_numpy():
    particles = _particles(5)
    assignments = np.array([0, 1, 0, 1, 0])

    classes = [
        Class2D(class_id=k, particles=particles[assignments == k]) for k in range(3)
    ]

    for k, cls in enumerate(classes):
        pixels = cls.particles["pixels"]
        assert isinstance(pixels, np.ndarray)
        assert np.array_equal(pixels, np.asarray(particles["pixels"])[assignments == k])
    assert np.asarray(classes[2].particles["pixels"]).shape == (0, 3, 4)


def test_concat_of_numpy_columns_stays_numpy():
    first, second = _particles(2), _particles(3)

    joined = B.concat([first, second[1:]])

    pixels = joined["pixels"]
    assert isinstance(pixels, np.ndarray)
    assert pixels.dtype == np.float32
    assert np.array_equal(pixels[:2], np.asarray(first["pixels"]))
    assert np.array_equal(pixels[2:], np.asarray(second["pixels"])[1:])


def test_concat_of_numpy_columns_with_different_row_shapes_gives_awkward():
    small = B.Set[Particle](capacity=2)
    small["pixels"] = np.zeros((2, 2, 2), dtype=np.float32)

    pixels = B.concat([_particles(1), small])["pixels"]

    assert isinstance(pixels, ak.Array)
    assert [np.asarray(row).shape for row in pixels] == [(3, 4), (2, 2), (2, 2)]


def test_concat_of_numpy_and_awkward_columns_gives_awkward():
    ragged = B.Set[Particle](
        [
            Particle(pixels=np.zeros((1, 1), np.float32)),
            Particle(pixels=np.ones((2, 2), np.float32)),
        ],
    )
    assert isinstance(ragged["pixels"], ak.Array)

    pixels = B.concat([_particles(1), ragged])["pixels"]

    assert isinstance(pixels, ak.Array)
    assert [np.asarray(row).shape for row in pixels] == [(3, 4), (1, 1), (2, 2)]
