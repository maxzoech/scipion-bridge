from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import mrcfile
import starfile
import pytest

import scipion_bridge as B
from scipion_bridge import Protocol
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.single_particle.particle import Particle
from scipion_bridge.backend.ray.ray_protocol_runner import RayPipelineRunner


class StreamingParticleProtocol(Protocol):
    particles: B.Input[B.Set[Particle]]

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        source = Source("particles")
        return source.map_batch(lambda p: p)


def _create_sample_particles(tmp_path: Path, n_particles: int = 10):
    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
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
    return star_path, data


def test_ray_protocol_runner_precomputation():
    """Verify that RayPipelineRunner precomputes resolvers on runner creation (Cold Path)."""
    proto = StreamingParticleProtocol()
    runner = RayPipelineRunner(proto)

    assert "particles" in runner.input_resolvers
    resolver = runner.input_resolvers["particles"]
    assert isinstance(resolver, B.ComposedResolver)
    assert resolver.origin == Path
    assert resolver.target == B.Set[Particle]
    # Slicing capability must be present
    assert any(step.requires_slice for step in resolver.steps)


def test_ray_protocol_runner_hot_path_zero_dijkstra(tmp_path, capsys):
    """Verify that no Dijkstra pathfinding occurs on the hot streaming execution path and lengths are printed."""
    star_path, _ = _create_sample_particles(tmp_path, n_particles=4)
    proto = StreamingParticleProtocol()
    reg = B.core.typed.resolve.current_registry()

    with patch.object(
        reg, "find_resolve_func", wraps=reg.find_resolve_func
    ) as mock_dijkstra:
        # Precomputation runs Dijkstra during init
        runner = RayPipelineRunner(proto)
        precompute_call_count = mock_dijkstra.call_count
        assert precompute_call_count >= 1

        # Hot data path: run streaming
        runner.run(particles=star_path, chunk_size=2)

        # Dijkstra pathfinding must NOT be called again on the hot path
        assert mock_dijkstra.call_count == precompute_call_count

    captured = capsys.readouterr()
    assert captured.out == "2\n2\n"


def test_ray_protocol_runner_iter_prints_lengths(tmp_path, capsys):
    """Verify that runner iterates over chunks of the resolved Set and prints each length."""
    star_path, _ = _create_sample_particles(tmp_path, n_particles=10)
    proto = StreamingParticleProtocol()
    runner = RayPipelineRunner(proto, chunk_size=3)

    runner.run(particles=star_path)

    captured = capsys.readouterr()
    # 10 particles with chunk_size 3 yields chunks of 3, 3, 3, 1
    assert captured.out == "3\n3\n3\n1\n"
