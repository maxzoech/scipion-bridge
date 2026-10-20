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
from scipion_bridge.single_particle import XmippParticleStackProxy
from scipion_bridge.single_particle.particle import Particle
from scipion_bridge.backend.ray.ray_protocol_runner import RayPipelineRunner

pytestmark = pytest.mark.usefixtures("ray_cluster")


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
        mrc.set_image_stack()

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


def test_ray_protocol_runner_hot_path_zero_dijkstra(tmp_path, monkeypatch):
    """Verify that no Dijkstra pathfinding occurs on the hot streaming execution path."""
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


def test_ray_protocol_runner_iter_chunk_sizes(tmp_path, monkeypatch):
    """Verify that runner iterates over chunks of the resolved Set and pushes correct chunk sizes."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "3")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=10)
    proto = StreamingParticleProtocol()
    runner = RayPipelineRunner(proto)

    chunk_sizes = []
    original_send = runner.pipeline.send

    def _spy_send(**kwargs):
        chunk_sizes.append(len(kwargs["particles"]))
        original_send(**kwargs)

    runner.pipeline.send = _spy_send
    runner.run(particles=star_path)

    # 10 particles with chunk_size 3 yields chunks of 3, 3, 3, 1
    assert chunk_sizes == [3, 3, 3, 1]


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


# Maps run as Ray tasks, possibly in several worker processes; a long-running
# protocol keeps its process-scope resources in a single process.
@B.resources(task=B.TaskType.LONG_RUNNING)
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


class RaySharedResourceMultiWorkerProtocol(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input(
        label="Input Particles",
        optional=False,
    )

    tracker_handle: Any = None
    shared_volume: B.Resource[np.ndarray] = B.Resource(
        builder=lambda self: (
            ray.get(self.tracker_handle.increment.remote()),
            np.ones((10, 10), dtype=np.float32),
        )[1],
        scope=B.ResourceScope.SHARED,
        dtype=np.ndarray,
    )

    def _step1(self, particles: B.Set[Particle]):
        vol = self.shared_volume
        assert isinstance(vol, np.ndarray)
        assert vol.shape == (10, 10)
        return particles

    def _step2(self, particles: B.Set[Particle]):
        vol = self.shared_volume
        assert isinstance(vol, np.ndarray)
        assert vol.shape == (10, 10)
        return {"output": particles}

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        return self.particles.map(self._step1).map(self._step2)


def test_ray_pipeline_runner_with_shared_resource_multi_worker(tmp_path, monkeypatch):
    """Verify that a SHARED resource is built exactly once even across multiple worker actors."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=6)

    tracker = BuilderTracker.remote()
    proto = RaySharedResourceMultiWorkerProtocol()
    proto.tracker_handle = tracker

    assert ray.get(tracker.get_count.remote()) == 0

    runner = RayPipelineRunner(proto)
    runner.run(particles=star_path)

    # Despite 2 distinct worker actors (step1 and step2) processing 3 chunks each,
    # the SHARED resource builder ran exactly once across the entire cluster!
    assert ray.get(tracker.get_count.remote()) == 1


class RayProcessResourceMultiWorkerProtocol(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input(
        label="Input Particles",
        optional=False,
    )

    tracker_handle: Any = None
    process_res: B.Resource[dict] = B.Resource(
        builder=lambda self: {"val": ray.get(self.tracker_handle.increment.remote())},
        scope=B.ResourceScope.PROCESS,
    )

    def _step1(self, particles: B.Set[Particle]):
        _ = self.process_res
        return particles

    def _step2(self, particles: B.Set[Particle]):
        _ = self.process_res
        return {"output": particles}

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        return self.particles.map(self._step1).map(self._step2)


def test_ray_pipeline_runner_with_process_resource_multi_worker(tmp_path, monkeypatch):
    """Verify that a PROCESS resource runs once per worker actor process (2 times for 2 actors)."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=6)

    tracker = BuilderTracker.remote()
    proto = RayProcessResourceMultiWorkerProtocol()
    proto.tracker_handle = tracker

    runner = RayPipelineRunner(proto)
    runner.run(particles=star_path)

    # 2 distinct worker actors each initialize their own process-local copy:
    assert ray.get(tracker.get_count.remote()) == 2


class RechunkProtocol(Protocol):
    chunked: B.Input[B.Set[Particle]]

    def outputs(self):
        return {"result": B.Set[Particle]}

    def steps(self):
        return Source("chunked").chunk(3).map(lambda p: {"result": p})


def test_ray_pipeline_runner_executes_chained_protocols(tmp_path, monkeypatch):
    """Verify that a chained protocol runs both stages as one Ray pipeline."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "2")
    star_path, _ = _create_sample_particles(tmp_path, n_particles=8)
    collector = OutputCollector.remote()

    chain = StreamingParticleProtocol() | RechunkProtocol()
    custom_sink = CallbackSinkWriter(
        lambda out: ray.get(collector.append.remote(len(out["result"]))),
    )
    runner = RayPipelineRunner(chain, sink=custom_sink)

    assert list(runner.input_resolvers) == ["particles"]

    runner.run(particles=star_path)

    results = ray.get(collector.get_items.remote())
    assert results == [3, 3, 2]


def test_ray_pipeline_runner_chain_validates_intermediate_outputs(tmp_path):
    """Verify that outputs of the first protocol are validated inside the chain."""
    star_path, _ = _create_sample_particles(tmp_path, n_particles=4)
    chain = InvalidOutputProtocol().pipe(
        RechunkProtocol(),
        mapping={"output": "chunked"},
    )
    runner = RayPipelineRunner(chain)

    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        runner.run(particles=star_path)

    assert "Protocol steps output must be a dictionary" in str(exc_info.value)


class Precision(Enum):
    FP16 = "fp16"
    FP32 = "fp32"


class ParameterProtocol(Protocol):
    particles: B.Input[B.Set[Particle]]
    precision: B.Field[Precision] = B.Field(default=Precision.FP32)
    scale: B.Field[int] = B.Field(default=1)

    def outputs(self):
        return {"output": B.Set[Particle], "tag": tuple}

    def _tag(self, particles):
        return {"output": particles, "tag": (self.precision.value, self.scale.value)}

    def steps(self):
        self.driver_values = (self.precision.value, self.scale.value)
        return Source("particles").map_batch(self._tag)


def _run_parameter_protocol(tmp_path, **runner_kwargs):
    star_path, _ = _create_sample_particles(tmp_path, n_particles=4)
    collector = OutputCollector.remote()
    proto = ParameterProtocol()
    runner = RayPipelineRunner(
        proto,
        sink=CallbackSinkWriter(
            lambda out: ray.get(collector.append.remote(out["tag"])),
        ),
        **runner_kwargs,
    )
    runner.run(particles=star_path)
    return proto, set(ray.get(collector.get_items.remote()))


def test_ray_pipeline_runner_serves_parameters(tmp_path):
    """Verify that parameter values reach steps() on the driver and the Ray stages."""
    proto, tags = _run_parameter_protocol(
        tmp_path,
        parameters={"precision": Precision.FP16, "scale": 3},
    )

    assert proto.driver_values == (Precision.FP16, 3)
    assert tags == {(Precision.FP16, 3)}


def test_ray_pipeline_runner_parameters_default(tmp_path):
    """Verify that parameters without a value resolve to their declared default."""
    proto, tags = _run_parameter_protocol(tmp_path, parameters={"scale": 2})

    assert proto.driver_values == (Precision.FP32, 2)
    assert tags == {(Precision.FP32, 2)}


def test_ray_pipeline_runner_rejects_unknown_parameters():
    with pytest.raises(ValueError, match="Unknown parameters"):
        RayPipelineRunner(ParameterProtocol(), parameters={"missing": 1})


class RequiredParameterProtocol(Protocol):
    particles: B.Input[B.Set[Particle]]
    model: B.Field[str] = B.Field(optional=False)

    def outputs(self):
        return {"output": B.Set[Particle]}

    def steps(self):
        return Source("particles").map_batch(lambda p: {"output": p})


def test_ray_pipeline_runner_requires_parameters_on_compile():
    runner = RayPipelineRunner(RequiredParameterProtocol())

    with pytest.raises(ValueError, match="required parameters"):
        runner.pipeline

    runner.set_parameters({"model": "small"})
    assert isinstance(runner.pipeline, Pipeline)

    with pytest.raises(RuntimeError):
        runner.set_parameters({"model": "large"})


def test_ray_pipeline_runner_cli_parameters(tmp_path, monkeypatch):
    """Verify that command line parameters are converted and served to the protocol."""
    star_path, _ = _create_sample_particles(tmp_path, n_particles=4)
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--particles", str(star_path), "--precision", "FP16", "--scale", "5"],
    )
    proto = ParameterProtocol()
    runner = RayPipelineRunner(proto)

    runner.launch_as_terminal_application()

    assert proto.driver_values == (Precision.FP16, 5)


def _create_xmipp_stack(tmp_path: Path, stem: str, first: int, n_particles: int):
    """Write ``<stem>.mrcs`` with images ``first, first + 1, ...`` and its ``.xmd``."""
    ids = np.arange(first, first + n_particles, dtype=np.float32)
    with mrcfile.new(tmp_path / f"{stem}.mrcs") as mrc:
        mrc.set_data(ids[:, None, None] * np.ones((1, 8, 8), dtype=np.float32))
        mrc.set_image_stack()

    rows = [
        f"{10000.0 + particle_id} 1 {i + 1:06d}@Runs/extra/{stem}.mrcs"
        for i, particle_id in enumerate(ids)
    ]
    xmd_path = tmp_path / f"{stem}.xmd"
    xmd_path.write_text(
        "# XMIPP_STAR_1 * \n# \ndata_noname\nloop_\n _ctfDefocusU\n _enabled\n _image\n"
        + "\n".join(rows)
        + "\n",
    )
    return xmd_path


def test_ray_pipeline_runner_loads_file_lists_in_parallel(tmp_path, monkeypatch):
    """Verify that loaders push the regrouped particles of every file into the pipeline."""
    monkeypatch.setenv("SCIPION_CHUNK_SIZE", "4")
    sizes = [3, 1, 5, 2, 4]
    firsts = np.cumsum([0, *sizes[:-1]]).tolist()
    paths = [
        _create_xmipp_stack(tmp_path, f"stack_{i}", first, n)
        for i, (first, n) in enumerate(zip(firsts, sizes))
    ]
    collector = OutputCollector.remote()
    sink = CallbackSinkWriter(
        lambda out: ray.get(
            collector.append.remote(
                (
                    np.asarray(out["output"]["pixels"])[:, 0, 0].tolist(),
                    np.asarray(out["output"]["ctf"]["defocus_u"]).flatten().tolist(),
                ),
            ),
        ),
    )
    runner = RayPipelineRunner(
        StreamingParticleProtocol(),
        intermediate_types={"particles": XmippParticleStackProxy},
        sink=sink,
        num_loaders=2,
    )

    runner.run(particles=paths)

    received = ray.get(collector.get_items.remote())
    ids = sorted(i for pixels, _ in received for i in pixels)
    defocus = sorted(d for _, defoci in received for d in defoci)
    assert ids == list(range(sum(sizes)))
    assert defocus == [10000.0 + i for i in range(sum(sizes))]
    # The loaders get stacks 0, 2, 4 (12 particles) and 1, 3 (3 particles),
    # which they regroup into chunks of 4.
    assert sorted(len(pixels) for pixels, _ in received) == [3, 4, 4, 4]


def test_ray_pipeline_runner_cli_expands_patterns(tmp_path, monkeypatch):
    """Verify that several paths and quoted glob patterns form one list of files."""
    for run in ["run_a", "run_b"]:
        (tmp_path / run).mkdir()
    run_a = [_create_xmipp_stack(tmp_path / "run_a", f"a_{i}", i, 1) for i in range(2)]
    run_b = [_create_xmipp_stack(tmp_path / "run_b", f"b_{i}", i, 1) for i in range(3)]
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--particles",
            str(tmp_path / "run_b" / "*.xmd"),
            str(run_a[1]),
            str(tmp_path / "run_a" / "a_0*.xmd"),
            "--num-loaders",
            "3",
        ],
    )
    runner = RayPipelineRunner(
        StreamingParticleProtocol(),
        intermediate_types={"particles": XmippParticleStackProxy},
    )

    with patch.object(runner, "run") as run:
        runner.launch_as_terminal_application()

    run.assert_called_once_with({"particles": [*run_b, run_a[1], run_a[0]]})
    assert runner.num_loaders == 3
