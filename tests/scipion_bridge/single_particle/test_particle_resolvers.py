from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

import scipion_bridge as B
from scipion_bridge.backend.standalone.container import Container
from scipion_bridge.single_particle import (
    Particle,
    MRCStackProxy,
    ParticleStackProxy,
    StarfileProxy,
)


@pytest.fixture(autouse=True)
def wire_container():
    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ],
    )
    yield container


def test_mrcs_path_cannot_resolve_to_particles(tmp_path: Path):
    file_path = tmp_path / "stack.mrcs"
    data = np.random.randn(8, 24, 24).astype(np.float32)
    with mrcfile.new(file_path) as mrc:
        mrc.set_data(data)

    with pytest.raises(TypeError):
        B.resolve(file_path, astype=B.Set[Particle])


def test_mrc_path_cannot_resolve_to_particles(tmp_path: Path):
    file_path = tmp_path / "stack.mrc"
    data = np.random.randn(4, 32, 32).astype(np.float32)
    with mrcfile.new(file_path) as mrc:
        mrc.set_data(data)

    with pytest.raises(TypeError):
        B.resolve(file_path, astype=B.Set[Particle])


def test_mrc_stack_proxy_cannot_resolve_to_particles(tmp_path: Path):
    file_path = tmp_path / "stack.mrcs"
    data = np.random.randn(4, 16, 16).astype(np.float32)
    with mrcfile.new(file_path) as mrc:
        mrc.set_data(data)

    proxy = MRCStackProxy(file_path)
    with pytest.raises(
        TypeError,
        match="'MRCStackProxy' could not be resolved as 'Set\\[Particle\\]'",
    ):
        B.resolve(proxy, astype=B.Set[Particle])


def test_resolve_path_mrcs_to_mrc_stack_proxy(tmp_path: Path):
    file_path = tmp_path / "stack.mrcs"
    data = np.random.randn(4, 16, 16).astype(np.float32)
    with mrcfile.new(file_path) as mrc:
        mrc.set_data(data)

    proxy = B.resolve(file_path, astype=MRCStackProxy)
    assert isinstance(proxy, MRCStackProxy)
    assert proxy.path == file_path
    assert proxy.metadata_path is None


def test_resolve_star_path_to_mrc_stack_proxy(tmp_path: Path):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    data = np.random.randn(5, 20, 20).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {"rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(5)]},
    )
    starfile.write(df, star_path)

    proxy = B.resolve(star_path, astype=MRCStackProxy)
    assert isinstance(proxy, MRCStackProxy)
    assert proxy.path == mrc_path
    assert proxy.metadata_path == star_path


def test_resolve_star_proxy_to_mrc_stack_proxy(tmp_path: Path):
    mrc_path = tmp_path / "data_stack.mrcs"
    star_path = tmp_path / "data.star"
    data = np.random.randn(3, 12, 12).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {"rlnImageName": [f"{i + 1:06d}@data_stack.mrcs" for i in range(3)]},
    )
    starfile.write(df, star_path)

    star_proxy = StarfileProxy(star_path)
    mrc_proxy = B.resolve(star_proxy, astype=MRCStackProxy)
    assert isinstance(mrc_proxy, MRCStackProxy)
    assert mrc_proxy.path == mrc_path
    assert mrc_proxy.metadata_path == star_path


def test_resolve_star_path_without_image_name_fails(tmp_path: Path):
    mrc_path = tmp_path / "sample.mrcs"
    star_path = tmp_path / "sample.star"
    data = np.random.randn(2, 10, 10).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    # Star file without rlnImageName
    df = pd.DataFrame({"rlnDefocusU": [10000.0, 10500.0]})
    starfile.write(df, star_path)

    with pytest.raises(ValueError, match="Could not find 'rlnImageName' column"):
        B.resolve(star_path, astype=MRCStackProxy)


def test_resolve_star_path_to_particle_stack_proxy(tmp_path: Path):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    data = np.random.randn(4, 16, 16).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {"rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(4)]},
    )
    starfile.write(df, star_path)

    psp = B.resolve(star_path, astype=ParticleStackProxy)
    assert isinstance(psp, ParticleStackProxy)
    assert psp.metadata.path == star_path
    assert psp.particle_stack.path == mrc_path


def test_resolve_star_path_to_particles_set(tmp_path: Path):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    data = np.random.randn(3, 14, 14).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {
            "rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(3)],
            "rlnDetectorPixelSize": [1.05, 1.05, 1.05],
            "rlnDefocusU": [12000.0, 12500.0, 13000.0],
            "rlnDefocusV": [11500.0, 12000.0, 12500.0],
            "rlnCoordinateX": [100.0, 200.0, 300.0],
            "rlnCoordinateY": [150.0, 250.0, 350.0],
            "rlnVoltage": [300.0, 300.0, 300.0],
        },
    )
    starfile.write(df, star_path)

    particles = B.resolve(star_path, astype=B.Set[Particle])
    assert len(particles) == 3
    assert np.asarray(particles["pixels"]).shape == (3, 14, 14)
    assert np.allclose(np.asarray(particles["pixels"]), data)
    assert np.allclose(np.asarray(particles["sampling_rate"]).flatten(), 1.05)
    assert np.allclose(
        np.asarray(particles["ctf"]["defocus_u"]).flatten(),
        [12000.0, 12500.0, 13000.0],
    )
    assert np.allclose(
        np.asarray(particles["coordinate"]["x"]).flatten(),
        [100.0, 200.0, 300.0],
    )
    assert np.allclose(
        np.asarray(particles["acquisition"]["voltage"]).flatten(),
        [300.0, 300.0, 300.0],
    )


def test_resolve_set_particles_to_mrc_stack():
    data = np.random.randn(5, 16, 16).astype(np.float32)
    p_set = B.Set[Particle](capacity=5)
    p_set["pixels"] = data

    proxy = B.resolve(p_set, astype=MRCStackProxy)
    assert isinstance(proxy, MRCStackProxy)
    assert proxy.path.exists()
    assert proxy.path.suffix == ".mrcs"

    with mrcfile.open(proxy.path) as mrc:
        loaded = np.asarray(mrc.data, dtype=np.float32)
    assert np.allclose(loaded, data)


def test_resolve_roundtrip_particles_to_particle_stack():
    data = np.random.randn(4, 18, 18).astype(np.float32)
    p_set = B.Set[Particle](capacity=4)
    p_set["pixels"] = data
    p_set["sampling_rate"] = np.array([1.2, 1.2, 1.2, 1.2])
    p_set["coordinate"]["x"] = np.array([10.0, 20.0, 30.0, 40.0])
    p_set["coordinate"]["y"] = np.array([15.0, 25.0, 35.0, 45.0])

    proxy_group = B.resolve(p_set, astype=ParticleStackProxy)
    assert isinstance(proxy_group, ParticleStackProxy)
    assert proxy_group.particle_stack.path.exists()
    assert proxy_group.metadata.path.exists()

    recovered_set = B.resolve(proxy_group, astype=B.Set[Particle])
    assert len(recovered_set) == 4
    assert np.allclose(np.asarray(recovered_set["pixels"]), data)
    assert np.allclose(
        np.asarray(recovered_set["sampling_rate"]).flatten(),
        [1.2, 1.2, 1.2, 1.2],
    )
    assert np.allclose(
        np.asarray(recovered_set["coordinate"]["x"]).flatten(),
        [10.0, 20.0, 30.0, 40.0],
    )


def test_invalid_extension_raises_type_error(tmp_path: Path):
    invalid_path = tmp_path / "particles.txt"
    invalid_path.write_text("not an mrc or star file")

    with pytest.raises(TypeError, match="file extension did not match"):
        B.resolve(invalid_path, astype=B.Set[Particle])


def test_mrc_stack_proxy_zero_io_len_and_estimated_nbytes(tmp_path: Path):
    file_path = tmp_path / "stack.mrcs"
    # 6 frames of 20x20 float32 (mode 2 -> 4 bytes per pixel)
    data = np.random.randn(6, 20, 20).astype(np.float32)
    with mrcfile.new(file_path) as mrc:
        mrc.set_data(data)

    proxy = MRCStackProxy(file_path)
    assert len(proxy) == 6
    # 20 * 20 * 4 = 1600 bytes per frame
    assert proxy.estimated_item_nbytes == 1600


def test_particle_stack_proxy_zero_io_len_and_estimated_nbytes(tmp_path: Path):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    data = np.random.randn(8, 24, 24).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {"rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(8)]},
    )
    starfile.write(df, star_path)

    psp = B.resolve(star_path, astype=ParticleStackProxy)
    assert len(psp) == 8
    assert len(psp.metadata) == 8
    assert len(psp.particle_stack) == 8
    # 24 * 24 * 4 = 2304 bytes per item
    assert psp.estimated_item_nbytes == 2304


def test_particle_stack_proxy_mismatched_lengths_raises(tmp_path: Path):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    data = np.random.randn(6, 24, 24).astype(np.float32)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {"rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(8)]},
    )
    starfile.write(df, star_path)

    psp = B.resolve(star_path, astype=ParticleStackProxy)
    assert len(psp.metadata) == 8
    assert len(psp.particle_stack) == 6
    with pytest.raises(ValueError, match="Inconsistent lengths in ParticleStackProxy"):
        _ = len(psp)


def test_resolve_particle_stack_to_particles_sliced(tmp_path: Path):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    n_particles = 10
    data = np.arange(n_particles * 16 * 16, dtype=np.float32).reshape(
        n_particles, 16, 16
    )
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {
            "rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(n_particles)],
            "rlnCoordinateX": [float(i * 10) for i in range(n_particles)],
            "rlnDefocusU": [float(10000 + i * 500) for i in range(n_particles)],
        },
    )
    starfile.write(df, star_path)

    psp = B.resolve(star_path, astype=ParticleStackProxy)

    # 1. Monolithic resolution (slice=None)
    full_particles = B.resolve(psp, astype=B.Set[Particle])
    assert len(full_particles) == 10
    assert np.allclose(np.asarray(full_particles["pixels"]), data)
    assert np.allclose(
        np.asarray(full_particles["coordinate"]["x"]).flatten(),
        [float(i * 10) for i in range(10)],
    )

    # 2. Sliced resolution (slice=slice(2, 6))
    sliced_particles = B.resolve(psp, astype=B.Set[Particle], slice=slice(2, 6))
    assert len(sliced_particles) == 4
    assert np.allclose(np.asarray(sliced_particles["pixels"]), data[2:6])
    assert np.allclose(
        np.asarray(sliced_particles["coordinate"]["x"]).flatten(),
        [20.0, 30.0, 40.0, 50.0],
    )
    assert np.allclose(
        np.asarray(sliced_particles["ctf"]["defocus_u"]).flatten(),
        [11000.0, 11500.0, 12000.0, 12500.0],
    )

    # 3. Single item retrieval via length-1 slice
    single_particle = B.resolve(psp, astype=B.Set[Particle], slice=slice(7, 8))[0]
    assert np.allclose(np.asarray(single_particle.pixels), data[7])
    assert single_particle.coordinate.x == 70.0


def test_resolve_star_path_to_particles_sliced_end_to_end(tmp_path: Path):
    """Verify that resolving a Path with a slice traverses Path -> ParticleStackProxy -> Set[Particle]."""
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    n_particles = 8
    data = np.arange(n_particles * 12 * 12, dtype=np.float32).reshape(
        n_particles, 12, 12
    )
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {
            "rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(n_particles)],
            "rlnCoordinateX": [float(i * 100) for i in range(n_particles)],
        },
    )
    starfile.write(df, star_path)

    # End-to-end resolution from Path
    subset = B.resolve(star_path, astype=B.Set[Particle], slice=slice(3, 7))
    assert len(subset) == 4
    assert np.allclose(np.asarray(subset["pixels"]), data[3:7])
    assert np.allclose(
        np.asarray(subset["coordinate"]["x"]).flatten(),
        [300.0, 400.0, 500.0, 600.0],
    )


def test_resolve_iter_particle_stack_end_to_end(tmp_path: Path):
    """Verify that B.resolve_iter streams chunks end-to-end with matching data."""
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    n_particles = 10
    data = np.arange(n_particles * 16 * 16, dtype=np.float32).reshape(
        n_particles, 16, 16
    )
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {
            "rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(n_particles)],
            "rlnCoordinateX": [float(i * 10) for i in range(n_particles)],
            "rlnDefocusU": [float(10000 + i * 500) for i in range(n_particles)],
        },
    )
    starfile.write(df, star_path)

    # 1. Iterate in fixed chunk_size=3 from Path
    chunks = list(B.resolve_iter(star_path, astype=B.Set[Particle], chunk_size=3))
    assert len(chunks) == 4
    assert [len(c) for c in chunks] == [3, 3, 3, 1]

    # Verify chunk slices
    assert np.allclose(np.asarray(chunks[0]["pixels"]), data[0:3])
    assert np.allclose(np.asarray(chunks[1]["pixels"]), data[3:6])
    assert np.allclose(np.asarray(chunks[2]["pixels"]), data[6:9])
    assert np.allclose(np.asarray(chunks[3]["pixels"]), data[9:10])

    assert np.allclose(
        np.asarray(chunks[0]["coordinate"]["x"]).flatten(),
        [0.0, 10.0, 20.0],
    )
    assert np.allclose(
        np.asarray(chunks[3]["coordinate"]["x"]).flatten(),
        [90.0],
    )

    # 2. Compare against monolithic resolution
    full = B.resolve(star_path, astype=B.Set[Particle])
    concatenated = B.concat(chunks)
    assert len(concatenated) == len(full) == 10
    assert np.allclose(np.asarray(concatenated["pixels"]), np.asarray(full["pixels"]))
    assert np.allclose(
        np.asarray(concatenated["coordinate"]["x"]).flatten(),
        np.asarray(full["coordinate"]["x"]).flatten(),
    )

    # 3. Iterate with auto-chunking (chunk_size=None)
    chunks_auto = list(B.resolve_iter(star_path, astype=B.Set[Particle]))
    assert sum(len(c) for c in chunks_auto) == 10

    # 4. Iterate directly from ParticleStackProxy
    psp = B.resolve(star_path, astype=ParticleStackProxy)
    chunks_from_psp = list(B.resolve_iter(psp, astype=B.Set[Particle], chunk_size=5))
    assert len(chunks_from_psp) == 2
    assert len(chunks_from_psp[0]) == 5
    assert len(chunks_from_psp[1]) == 5
    assert np.allclose(np.asarray(chunks_from_psp[0]["pixels"]), data[0:5])
    assert np.allclose(np.asarray(chunks_from_psp[1]["pixels"]), data[5:10])

    # 5. Non-sliceable target raises TypeError
    with pytest.raises(
        TypeError,
        match="no resolver along the path supports slicing",
    ):
        list(B.resolve_iter(star_path, astype=StarfileProxy))


def test_resolve_particle_stack_to_particles_caches_star_data(tmp_path: Path):
    """Verify that starfile.read is called only once across multiple chunks during iteration."""
    from unittest.mock import patch

    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    n_particles = 10
    data = np.arange(n_particles * 16 * 16, dtype=np.float32).reshape(
        n_particles, 16, 16
    )
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {
            "rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(n_particles)],
            "rlnCoordinateX": [float(i * 10) for i in range(n_particles)],
        },
    )
    starfile.write(df, star_path)

    # Pre-resolve to ParticleStackProxy so we isolate the slice step caching
    psp = B.resolve(star_path, astype=ParticleStackProxy)

    with patch("starfile.read", wraps=starfile.read) as mock_read:
        chunks = list(B.resolve_iter(psp, astype=B.Set[Particle], chunk_size=2))
        assert len(chunks) == 5
        # Across 5 chunks, starfile.read must only be called once!
        assert mock_read.call_count == 1
