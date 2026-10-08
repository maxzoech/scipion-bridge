"""Arrow export/import and Arrow-based pickling of Sets."""

import pickle

import awkward as ak
import cloudpickle
import numpy as np
import pyarrow as pa
import pytest
import ray
import ray.cloudpickle

import scipion_bridge as B
from scipion_bridge.core.struct.exceptions import UninitializedFieldError
from scipion_bridge.single_particle.particle import Particle


class Sample(B.Struct):
    image: B.Array[np.float32] = B.Array(shape=(4, 4))
    embedding: B.Array[float] = B.Array(shape=(None,))
    score: float


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
}


def _make_samples(n: int) -> B.Set[Sample]:
    samples = B.Set[Sample](capacity=n)
    samples["image"] = np.arange(n * 16, dtype=np.float32).reshape(n, 4, 4)
    samples["embedding"] = ak.Array(
        [np.arange(i % 3 + 1, dtype=float) for i in range(n)]
    )
    samples["score"] = np.arange(n, dtype=float).reshape(n, 1)
    return samples


def _make_movies(n: int) -> B.Set[Movie]:
    movies = B.Set[Movie](capacity=n)
    movies["movie_id"] = np.arange(n).reshape(n, 1)
    movies["frames"]["pixels"] = np.arange(n * 3 * 4, dtype=float).reshape(n, 3, 2, 2)
    return movies


def _assert_samples_equal(actual: B.Set[Sample], expected: B.Set[Sample]) -> None:
    assert len(actual) == len(expected)
    assert np.array_equal(actual["image"], expected["image"])
    assert ak.to_list(actual["embedding"]) == ak.to_list(expected["embedding"])
    assert np.array_equal(actual["score"], expected["score"])


def test_to_arrow_has_one_column_per_initialized_field():
    batch = _make_samples(5).to_arrow()

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 5
    assert batch.schema.names == ["image", "embedding", "score"]


def test_arrow_roundtrip_static_and_ragged_fields():
    samples = _make_samples(6)

    restored = B.Set[Sample].from_arrow(samples.to_arrow())

    _assert_samples_equal(restored, samples)
    assert restored.capacity == samples.capacity


def test_arrow_roundtrip_nested_struct_fields():
    particles = B.Set[Particle](
        [
            Particle(
                pixels=np.full((3, 3), i, dtype=np.float32), sampling_rate=float(i)
            )
            for i in range(4)
        ]
    )

    batch = particles.to_arrow()
    restored = B.Set[Particle].from_arrow(batch)

    assert batch.schema.names == ["pixels", "sampling_rate"]
    assert np.array_equal(
        ak.to_numpy(restored["pixels"]), ak.to_numpy(particles["pixels"])
    )
    assert np.array_equal(restored["sampling_rate"], particles["sampling_rate"])


def test_arrow_roundtrip_nested_set_fields():
    movies = _make_movies(4)

    batch = movies.to_arrow()
    restored = B.Set[Movie].from_arrow(batch)

    assert isinstance(batch.column("frames"), pa.StructArray)
    assert np.array_equal(restored["frames"]["pixels"], movies["frames"]["pixels"])
    assert np.array_equal(restored["movie_id"], movies["movie_id"])


def test_arrow_roundtrip_keeps_uninitialized_fields_uninitialized():
    samples = B.Set[Sample](capacity=3)
    samples["score"] = np.ones((3, 1))

    restored = B.Set[Sample].from_arrow(samples.to_arrow())

    assert np.array_equal(restored["score"], samples["score"])
    with pytest.raises(UninitializedFieldError):
        restored["image"]


def test_arrow_roundtrip_keeps_dynamic_capacity():
    samples = B.Set[Sample]()
    samples["score"] = np.ones((3, 1))

    restored = B.Set[Sample].from_arrow(samples.to_arrow())

    assert restored.capacity is None
    assert len(restored) == 3


def test_sliced_set_exports_only_visible_rows():
    samples = _make_samples(10)

    sliced = samples[2:7]
    batch = sliced.to_arrow()
    restored = B.Set[Sample].from_arrow(batch)

    assert batch.num_rows == 5
    _assert_samples_equal(restored, sliced)


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_pickle_roundtrip(serializer):
    dumps, loads = _SERIALIZERS[serializer]
    samples = _make_samples(5)

    restored = loads(dumps(samples))

    assert type(restored) is B.Set[Sample]
    _assert_samples_equal(restored, samples)


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_pickle_roundtrip_nested_set(serializer):
    dumps, loads = _SERIALIZERS[serializer]
    movies = _make_movies(3)

    restored = loads(dumps(movies))

    assert np.array_equal(restored["frames"]["pixels"], movies["frames"]["pixels"])


def test_pickled_slice_does_not_include_parent_storage():
    # Large enough that the fixed size of the pickled schema does not matter.
    samples = _make_samples(2500)

    full_size = len(pickle.dumps(samples, protocol=5))
    slice_size = len(pickle.dumps(samples[:250], protocol=5))

    assert slice_size < 0.15 * full_size


@pytest.mark.usefixtures("ray_cluster")
def test_write_after_zero_copy_restore():
    samples = _make_samples(4)

    restored = ray.get(ray.put(samples))
    assert not restored["score"].flags.writeable

    restored[0].score = 42.0

    assert restored["score"][0, 0] == 42.0
    assert samples["score"][0, 0] == 0.0


def test_set_field_descriptor_keeps_default_pickling():
    descriptor = Movie.__dict__["frames"]

    restored = cloudpickle.loads(cloudpickle.dumps(descriptor))

    assert restored.name == "frames"
    assert restored.capacity == 3
