from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np
import pandas as pd
import mrcfile
import starfile
import pytest
import ray

import scipion_bridge as B
from scipion_bridge import Protocol
from scipion_bridge.core.streaming.ops import Source, MapOp
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter
from scipion_bridge.single_particle.particle import Particle
from scipion_bridge.backend.ray.ray_protocol_runner import RayPipelineRunner


class StreamingParticleProtocol(Protocol):
    particles: B.Input[B.Set[Particle]]

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        source = Source("particles")
        return source.map_batch(lambda p: {"output": p})


def _create_sample_particles(
    tmp_path: Path, n_particles: int = 10, filename: str = "particles"
):
    mrc_path = tmp_path / f"{filename}.mrcs"
    star_path = tmp_path / f"{filename}.star"
    data = np.arange(n_particles * 16 * 16, dtype=np.float32).reshape(
        n_particles, 16, 16
    )
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {
            "rlnImageName": [
                f"{i + 1:06d}@{filename}.mrcs" for i in range(n_particles)
            ],
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


def test_ray_protocol_runner_hot_path_zero_dijkstra(tmp_path, capsys, monkeypatch):
    """Verify that no Dijkstra pathfinding occurs on the hot streaming execution path and lengths are printed."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
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
        runner.run(particles=star_path)

        # Dijkstra pathfinding must NOT be called again on the hot path
        assert mock_dijkstra.call_count == precompute_call_count

    captured = capsys.readouterr()
    assert captured.out == "2\n2\n"


def test_ray_protocol_runner_iter_prints_lengths(tmp_path, capsys, monkeypatch):
    """Verify that runner iterates over chunks of the resolved Set and prints each length."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "3")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=10)
    proto = StreamingParticleProtocol()
    runner = RayPipelineRunner(proto)

    runner.run(particles=star_path)

    captured = capsys.readouterr()
    # 10 particles with chunk_size 3 yields chunks of 3, 3, 3, 1
    assert captured.out == "3\n3\n3\n1\n"


def test_ray_pipeline_runner_compiles_pipeline():
    """Verify that RayPipelineRunner compiles protocol steps into a valid Pipeline instance."""
    proto = StreamingParticleProtocol()
    runner = RayPipelineRunner(proto)

    assert isinstance(runner.pipeline, Pipeline)
    assert "particles" in runner.pipeline._compiled._sources


class InvalidOutputProtocol(Protocol):
    particles: B.Input[B.Set[Particle]]

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        source = Source("particles")
        # Invalid: returning Set directly rather than a dictionary
        return source.map_batch(lambda p: p)


def test_ray_pipeline_runner_validation_error_on_non_dict_output(tmp_path):
    """Verify that returning a bare non-dict object from steps() triggers ValidationError."""
    star_path, _ = _create_sample_particles(tmp_path, n_particles=4)
    proto = InvalidOutputProtocol()
    runner = RayPipelineRunner(proto)

    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        runner.run(particles=star_path)

    assert "ValidationError" in str(exc_info.value)
    assert "Protocol steps output must be a dictionary" in str(exc_info.value)


@ray.remote
class OutputCollector:
    def __init__(self):
        self.received = []

    def append(self, item):
        self.received.append(item)

    def get_items(self):
        return self.received


def test_ray_pipeline_runner_executes_pipeline_with_custom_sink(tmp_path, monkeypatch):
    """Verify that custom sink receives processed chunks from the Ray pipeline."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "4")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=8)
    collector = OutputCollector.remote()

    proto = StreamingParticleProtocol()
    custom_sink = CallbackSinkWriter(
        lambda out: ray.get(collector.append.remote(len(out["output"]))),
    )
    runner = RayPipelineRunner(proto, sink=custom_sink)

    runner.run(particles=star_path)

    results = ray.get(collector.get_items.remote())
    assert results == [4, 4]


class DualInputProtocol(Protocol):
    particles1: B.Input[B.Set[Particle]]
    particles2: B.Input[B.Set[Particle]]

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        s1 = Source("particles1")
        s2 = Source("particles2")
        op = MapOp(lambda p: {"output": p})
        s1.op(op)
        s2.op(op)
        return op


def test_ray_pipeline_runner_multi_input_interleaving(tmp_path, monkeypatch):
    """Verify that multiple inputs are interleaved in lockstep without stalling."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
    star_path1, _ = _create_sample_particles(tmp_path, n_particles=4, filename="p1")
    star_path2, _ = _create_sample_particles(tmp_path, n_particles=4, filename="p2")

    proto = DualInputProtocol()
    runner = RayPipelineRunner(proto)

    # Track pushes into the pipeline
    sent_order = []
    original_send = runner.pipeline.send

    def _spy_send(**kwargs):
        sent_order.append(list(kwargs.keys())[0])
        original_send(**kwargs)

    runner.pipeline.send = _spy_send

    runner.run(particles1=star_path1, particles2=star_path2)

    # Both streams should alternate in lockstep: p1, p2, p1, p2
    assert sent_order == ["particles1", "particles2", "particles1", "particles2"]


def test_ray_pipeline_runner_target_byte_size_env(tmp_path, monkeypatch):
    """Verify that SCIPION_TARGET_BYTE_SIZE is read from environment during run()."""
    monkeypatch.setenv("SCIPION_TARGET_BYTE_SIZE", "1048576")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=6)

    proto = StreamingParticleProtocol()
    runner = RayPipelineRunner(proto)

    with patch.object(
        runner.input_resolvers["particles"],
        "iter",
        wraps=runner.input_resolvers["particles"].iter,
    ) as mock_iter:
        runner.run(particles=star_path)
        assert mock_iter.call_count == 1
        _, kwargs = mock_iter.call_args
        assert kwargs.get("target_bytes") == 1048576


class MainModuleBoundMethodProtocol(Protocol):
    class ModelType(Enum):
        CRYO_IEF_SMALL = "Cryo-IEF (Base)"

    particles: B.Input[B.Set[Particle]] = B.Input(
        label="Input Particles",
        optional=False,
    )
    model_type: B.Field[ModelType] = B.Field(
        default=ModelType.CRYO_IEF_SMALL,
    )
    chunk_size: B.Field[int] = B.Field(
        default=10_000,
    )

    def _compute_latents(self, particles: B.Set[Particle]):
        return {"output": particles}

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        return self.particles.map(self._compute_latents)


def test_ray_pipeline_runner_with_bound_method_and_main_module(tmp_path, monkeypatch):
    """Verify that protocol with bound methods and __module__ = '__main__' executes in Ray without OSError."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=4)

    # Force __module__ to '__main__' to simulate execution via `python -m ...`
    MainModuleBoundMethodProtocol.__module__ = "__main__"
    MainModuleBoundMethodProtocol.ModelType.__module__ = "__main__"

    proto = MainModuleBoundMethodProtocol()
    runner = RayPipelineRunner(proto)
    runner.run(particles=star_path)


@ray.remote
class BuilderTracker:
    def __init__(self):
        self.call_count = 0

    def increment(self):
        self.call_count += 1
        return self.call_count

    def get_count(self):
        return self.call_count


class RayResourceProtocol(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input(
        label="Input Particles",
        optional=False,
    )

    tracker_handle: Any = None
    model: B.Resource[dict] = B.Resource(
        builder=lambda self: {
            "builder_run": ray.get(self.tracker_handle.increment.remote())
        },
    )

    def _infer(self, particles: B.Set[Particle]):
        model_info = self.model
        assert model_info["builder_run"] == 1
        return {"output": particles}

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        return self.particles.map(self._infer)


def test_ray_pipeline_runner_with_resource(tmp_path, monkeypatch):
    """Verify that Resource builder runs lazily inside the Ray worker actor exactly once across streaming chunks."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=6)

    tracker = BuilderTracker.remote()
    proto = RayResourceProtocol()
    proto.tracker_handle = tracker

    # Before running: builder has not run
    assert ray.get(tracker.get_count.remote()) == 0

    runner = RayPipelineRunner(proto)
    runner.run(particles=star_path)

    # After streaming 3 chunks of 2 particles: builder was called exactly once in worker actor
    assert ray.get(tracker.get_count.remote()) == 1
