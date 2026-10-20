from pathlib import Path

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.backend.ray.loader import coalesce, expand_pattern, shard
from scipion_bridge.single_particle import Particle


def _particles(start: int, n: int) -> B.Set[Particle]:
    particles = B.Set[Particle](capacity=n)
    particles["pixels"] = np.arange(start, start + n, dtype=np.float32)[
        :, None, None
    ] * np.ones((1, 4, 4), dtype=np.float32)
    return particles


def _ids(particles: B.Set[Particle]) -> list[int]:
    return np.asarray(particles["pixels"])[:, 0, 0].astype(int).tolist()


def test_coalesce_merges_small_chunks():
    chunks = [_particles(0, 2), _particles(2, 1), _particles(3, 3), _particles(6, 1)]

    merged = list(coalesce(chunks, chunk_size=3))

    assert [len(c) for c in merged] == [3, 3, 1]
    assert [i for c in merged for i in _ids(c)] == list(range(7))


def test_coalesce_splits_large_chunks():
    chunks = [_particles(0, 1), _particles(1, 8)]

    merged = list(coalesce(chunks, chunk_size=4))

    assert [len(c) for c in merged] == [4, 4, 1]
    assert [i for c in merged for i in _ids(c)] == list(range(9))


def test_coalesce_sizes_chunks_by_target_bytes():
    item_nbytes = _particles(0, 1).to_arrow().nbytes
    chunks = [_particles(i, 1) for i in range(10)]

    merged = list(coalesce(chunks, target_bytes=4 * item_nbytes))

    assert [len(c) for c in merged] == [4, 4, 2]


def test_coalesce_without_chunks():
    assert list(coalesce([], chunk_size=4)) == []


def test_shard_round_robin():
    assert shard(list(range(7)), 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert shard([0, 1], 4) == [[0], [1]]


def test_expand_pattern(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    for name in ["b.xmd", "a.xmd", "a.mrcs", "nested/c.xmd"]:
        (tmp_path / name).touch()

    assert expand_pattern(str(tmp_path / "*.xmd")) == [
        tmp_path / "a.xmd",
        tmp_path / "b.xmd",
    ]
    assert expand_pattern(str(tmp_path / "**" / "*.xmd")) == [
        tmp_path / "a.xmd",
        tmp_path / "b.xmd",
        tmp_path / "nested" / "c.xmd",
    ]

    with pytest.raises(FileNotFoundError):
        expand_pattern(str(tmp_path / "*.star"))
