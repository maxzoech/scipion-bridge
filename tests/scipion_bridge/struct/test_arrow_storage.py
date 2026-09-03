import numpy as np
import pytest
import pyarrow as pa

import scipion_bridge as B
from scipion_bridge.core.struct import RaggedArrayView
from scipion_bridge.core.struct.utils.dask_serialization import serialize_set, deserialize_set


class Metadata(B.Struct):
    tag: int
    latent = B.Array[np.float32](shape=(None,))


class Particle(B.Struct):
    pixels = B.Array[np.float32](shape=(32, 32))
    metadata: Metadata


def test_arrow_sequence_constructor():
    p1 = Particle(
        pixels=np.ones((32, 32), dtype=np.float32),
        metadata=Metadata(tag=1, latent=np.array([1.0, 2.0], dtype=np.float32)),
    )
    p2 = Particle(
        pixels=np.zeros((32, 32), dtype=np.float32),
        metadata=Metadata(tag=2, latent=np.array([3.0, 4.0, 5.0], dtype=np.float32)),
    )

    particles = B.Set[Particle]([p1, p2])
    assert len(particles) == 2
    assert particles.capacity == 2

    # Verify elements
    assert np.array_equal(particles[0].pixels, p1.pixels)
    assert particles[0].metadata.tag == 1
    assert np.array_equal(particles[0].metadata.latent, p1.metadata.latent)

    assert np.array_equal(particles[1].pixels, p2.pixels)
    assert particles[1].metadata.tag == 2
    assert np.array_equal(particles[1].metadata.latent, p2.metadata.latent)


def test_arrow_record_batch_conversion_and_freeze():
    p1 = Particle(
        pixels=np.ones((32, 32), dtype=np.float32),
        metadata=Metadata(tag=10, latent=np.ones(64, dtype=np.float32)),
    )
    p2 = Particle(
        pixels=np.zeros((32, 32), dtype=np.float32),
        metadata=Metadata(tag=20, latent=np.ones(128, dtype=np.float32)),
    )

    particles = B.Set[Particle]([p1, p2])
    batch = particles.to_arrow()

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 2
    assert "pixels" in batch.schema.names
    assert "metadata" in batch.schema.names

    # Schema inspection
    pixels_type = batch.schema.field("pixels").type
    assert isinstance(pixels_type, pa.FixedShapeTensorType)
    assert pixels_type.shape == [32, 32]

    meta_type = batch.schema.field("metadata").type
    assert isinstance(meta_type, pa.StructType)

    # Immutability verification
    with pytest.raises(RuntimeError, match="Cannot mutate a frozen Set"):
        particles[0] = p2


def test_arrow_ragged_array_view():
    class Sample(B.Struct):
        latent = B.Array[np.float32](shape=(None,))

    samples = B.Set[Sample](capacity=3)
    samples[0] = Sample(latent=np.array([1.0, 2.0], dtype=np.float32))
    samples[1] = Sample(latent=np.array([3.0, 4.0, 5.0], dtype=np.float32))
    samples[2] = Sample(latent=np.array([6.0], dtype=np.float32))

    latent_view = samples["latent"]
    assert isinstance(latent_view, RaggedArrayView)
    assert len(latent_view) == 3

    assert np.array_equal(latent_view[0], np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(latent_view[1], np.array([3.0, 4.0, 5.0], dtype=np.float32))
    assert np.array_equal(latent_view[2], np.array([6.0], dtype=np.float32))

    # Test list conversion
    as_list = latent_view.to_list()
    assert len(as_list) == 3
    assert all(isinstance(arr, np.ndarray) for arr in as_list)


def test_arrow_2d_slicing():
    class SimpleData(B.Struct):
        pixels = B.Array[np.float32](shape=(16, 16))
        value: int

    data_set = B.Set[SimpleData](capacity=10)
    for i in range(10):
        data_set[i] = SimpleData(
            pixels=np.full((16, 16), i, dtype=np.float32),
            value=i * 10,
        )

    # 2D slice: data_set[2:5, "pixels"]
    sliced_pixels = data_set[2:5, "pixels"]
    assert sliced_pixels.shape == (3, 16, 16)
    assert np.all(sliced_pixels[0] == 2)
    assert np.all(sliced_pixels[1] == 3)
    assert np.all(sliced_pixels[2] == 4)


def test_arrow_set_concatenation():
    p1 = Particle(
        pixels=np.ones((32, 32), dtype=np.float32),
        metadata=Metadata(tag=1, latent=np.ones(10, dtype=np.float32)),
    )
    p2 = Particle(
        pixels=np.full((32, 32), 2.0, dtype=np.float32),
        metadata=Metadata(tag=2, latent=np.ones(20, dtype=np.float32)),
    )
    p3 = Particle(
        pixels=np.full((32, 32), 3.0, dtype=np.float32),
        metadata=Metadata(tag=3, latent=np.ones(30, dtype=np.float32)),
    )

    set1 = B.Set[Particle]([p1])
    set2 = B.Set[Particle]([p2, p3])

    combined = B.Set.concat(set1, set2)
    assert len(combined) == 3
    assert combined.capacity == 3

    assert np.all(combined[0].pixels == 1.0)
    assert np.all(combined[1].pixels == 2.0)
    assert np.all(combined[2].pixels == 3.0)

    assert combined[0].metadata.tag == 1
    assert combined[1].metadata.tag == 2
    assert combined[2].metadata.tag == 3

    assert len(combined[0].metadata.latent) == 10
    assert len(combined[1].metadata.latent) == 20
    assert len(combined[2].metadata.latent) == 30


def test_dask_ipc_roundtrip():
    p1 = Particle(
        pixels=np.full((32, 32), 7.0, dtype=np.float32),
        metadata=Metadata(tag=77, latent=np.full(15, 7.7, dtype=np.float32)),
    )
    particles = B.Set[Particle]([p1])

    header, frames = serialize_set(particles)
    recovered = deserialize_set(header, frames)

    assert len(recovered) == 1
    assert np.all(recovered[0].pixels == 7.0)
    assert recovered[0].metadata.tag == 77
    assert len(recovered[0].metadata.latent) == 15
    assert np.allclose(recovered[0].metadata.latent, 7.7)

