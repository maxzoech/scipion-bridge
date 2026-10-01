import asyncio
import pytest
import numpy as np
import tensorstore as ts
from pathlib import Path
import ray

import scipion_bridge as B
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.backend.ray.sinks.tensorstore_sink import TensorStoreSinkWriter


class Metadata(B.Struct):
    foo: int


class Particle(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(16, 16))
    metadata: Metadata


class DynamicParticle(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(None, None))


class Class2D(B.Struct):
    representative: B.Array[np.float32] = B.Array(shape=(16, 16))
    particles: B.Set[Particle] = B.Set[Particle](capacity=10)


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        import sys

        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            runtime_env={"env_vars": {"PYTHONPATH": ":".join(sys.path)}},
        )
    yield


def test_tensorstore_sink_set_append(tmp_path: Path):
    async def _run():
        zarr_path = tmp_path / "particles.zarr"
        writer = TensorStoreSinkWriter(zarr_path)

        # Batch 1: 3 particles
        batch1 = B.Set[Particle](capacity=3)
        for i in range(3):
            batch1[i] = Particle(
                pixels=np.full((16, 16), float(i), dtype=np.float32),
                metadata=Metadata(foo=i),
            )

        # Batch 2: 2 particles
        batch2 = B.Set[Particle](capacity=2)
        for i in range(2):
            batch2[i] = Particle(
                pixels=np.full((16, 16), float(i + 10), dtype=np.float32),
                metadata=Metadata(foo=i + 10),
            )

        await writer.write(batch1)
        await writer.write(batch2)
        await writer.finalize()

        # Open and verify via TensorStore
        pixels_store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(zarr_path / "pixels")},
            }
        )
        metadata_store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(zarr_path / "metadata/foo")},
            }
        )

        assert pixels_store.shape == (5, 16, 16)
        assert metadata_store.shape == (5, 1)

        pixels_data = await pixels_store.read()
        assert np.all(pixels_data[0] == 0.0)
        assert np.all(pixels_data[2] == 2.0)
        assert np.all(pixels_data[3] == 10.0)
        assert np.all(pixels_data[4] == 11.0)

    asyncio.run(_run())


def test_tensorstore_sink_struct_last_writer_wins(tmp_path: Path):
    async def _run():
        zarr_path = tmp_path / "single_particle.zarr"
        writer = TensorStoreSinkWriter(zarr_path)

        p1 = Particle(
            pixels=np.full((16, 16), 1.0, dtype=np.float32),
            metadata=Metadata(foo=100),
        )
        p2 = Particle(
            pixels=np.full((16, 16), 2.0, dtype=np.float32),
            metadata=Metadata(foo=200),
        )

        await writer.write(p1)
        await writer.write(p2)  # Overwrites p1
        await writer.finalize()

        pixels_store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(zarr_path / "pixels")},
            }
        )
        metadata_store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(zarr_path / "metadata/foo")},
            }
        )

        pixels_data = await pixels_store.read()
        metadata_data = await metadata_store.read()

        assert np.all(pixels_data == 2.0)
        assert np.all(metadata_data == 200)

    asyncio.run(_run())


def test_tensorstore_sink_dynamic_shape_support(tmp_path: Path):
    async def _run():
        zarr_path = tmp_path / "dynamic_particles.zarr"
        writer = TensorStoreSinkWriter(zarr_path)

        # Dynamic shape initialized with concrete dense arrays
        batch = B.Set[DynamicParticle](capacity=2)
        for i in range(2):
            batch[i] = DynamicParticle(
                pixels=np.ones((32, 32), dtype=np.float32) * i,
            )

        await writer.write(batch)
        await writer.finalize()

        pixels_store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(zarr_path / "pixels")},
            }
        )

        assert pixels_store.shape == (2, 32, 32)
        pixels_data = await pixels_store.read()
        assert np.all(pixels_data[0] == 0.0)
        assert np.all(pixels_data[1] == 1.0)

    asyncio.run(_run())


@pytest.mark.usefixtures("ray_cluster")
def test_ray_pipeline_with_tensorstore_sink(tmp_path: Path):
    zarr_path = tmp_path / "pipeline_particles.zarr"
    writer = TensorStoreSinkWriter(zarr_path)

    source = Source("raw_particles")
    sink_node = source.map_batch(lambda p_set: p_set).write_to(  # 1:1 pass-through
        writer
    )

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        batch = B.Set[Particle](capacity=2)
        batch[0] = Particle(
            pixels=np.zeros((16, 16), dtype=np.float32),
            metadata=Metadata(foo=1),
        )
        batch[1] = Particle(
            pixels=np.ones((16, 16), dtype=np.float32),
            metadata=Metadata(foo=2),
        )
        pipeline.send(raw_particles=batch)

    # Verify data on disk
    store = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(zarr_path / "pixels")},
        }
    ).result()
    assert store.shape == (2, 16, 16)
