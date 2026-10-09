"""Declaring compute resources on protocols and assigning them to stages."""

from numpy._core.multiarray import (
    _set_madvise_hugepage,  # pyright: ignore[reportAttributeAccessIssue]
)
import pytest

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import _ray_options
from scipion_bridge.core.environment.compute import (
    NUMPY_HUGEPAGE_ENV_VAR,
    ComputeAssignment,
    configure_numpy_hugepages,
    gpu_claim,
    gpu_memory_env,
    numpy_hugepage_env,
)
from scipion_bridge.core.streaming.ir import IRMap
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import MapElementOp, MapOp, Source, lineage
from scipion_bridge.single_particle.particle import Particle


def _second(item):
    return item[1]


def _identity(item):
    return item


@B.resources(gpus=1)
class Refine(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return (
            self.particles.chunk(2)
            .group_by(
                0,
                lambda stream: stream.map(_second).map_element(_identity),
            )
            .unkey()
            .map(self._output)
        )

    def _output(self, item):
        return {"particles": item.value}


class Untagged(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return self.particles.map(lambda p: {"particles": p})


@B.resources(gpus=0.5, cpus=4, task=B.TaskType.LONG_RUNNING)
class Inference(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return self.particles.map(lambda p: {"particles": p})


class RefineSubclass(Refine):
    pass


def _maps(op):
    return [node for node in lineage(op) if isinstance(node, (MapOp, MapElementOp))]


def test_default_resources():
    assert B.ComputeResources() == B.ComputeResources(
        gpus=0,
        cpus=None,
        task=B.TaskType.EPHEMERAL,
    )


@pytest.mark.parametrize("kwargs", [{"gpus": -1}, {"cpus": -0.5}, {"min_vram": -1}])
def test_negative_resources_raise(kwargs):
    with pytest.raises(ValueError, match="must not be negative"):
        B.ComputeResources(**kwargs)


def test_decorator_sets_resources_on_the_class():
    assert Inference.compute_resources == B.ComputeResources(
        gpus=0.5,
        cpus=4,
        task=B.TaskType.LONG_RUNNING,
    )
    assert Untagged.compute_resources is None


def test_decorator_returns_the_class():
    cls = B.resources(gpus=1)(Untagged)

    try:
        assert cls is Untagged
    finally:
        Untagged.compute_resources = None


def test_subclasses_inherit_resources():
    assert RefineSubclass.compute_resources == Refine.compute_resources


def test_decorator_rejects_other_classes():
    with pytest.raises(TypeError, match="applies to Protocol classes"):
        B.resources(gpus=1)(Particle)  # type: ignore[arg-type]


def test_maps_inside_group_by_are_assigned():
    protocol = Refine()

    steps = protocol._tagged_steps()

    assignment = ComputeAssignment(
        B.ComputeResources(gpus=1),
        group=protocol.protocol_id,
    )
    maps = _maps(steps)
    assert len(maps) == 3
    assert {type(node) for node in maps} == {MapOp, MapElementOp}
    assert all(node.compute == assignment for node in maps)


def test_built_in_ops_are_not_assigned():
    steps = Refine()._tagged_steps()

    built_ins = [
        node for node in lineage(steps) if not isinstance(node, (MapOp, MapElementOp))
    ]
    assert built_ins
    assert not any(hasattr(node, "compute") for node in built_ins)


def test_verify_outputs_stage_is_not_assigned():
    pipeline = Refine().get_pipeline()

    assert isinstance(pipeline, MapOp)
    assert pipeline.compute is None


def test_protocol_without_resources_assigns_nothing():
    assert all(node.compute is None for node in _maps(Untagged()._tagged_steps()))


def test_chained_protocols_keep_their_own_resources():
    first, second = Inference(), Untagged()
    assert Inference.compute_resources is not None

    pipeline = (first | second).get_pipeline()

    assignments = {node.compute for node in _maps(pipeline)}
    assert assignments == {
        None,
        ComputeAssignment(Inference.compute_resources, group=first.protocol_id),
    }


def test_lowering_keeps_the_assignment():
    protocol = Inference()
    assert Inference.compute_resources is not None
    (sink,) = lower([protocol._tagged_steps().sink(print)])

    (stage,) = sink.upstream
    assert isinstance(stage, IRMap)
    assert stage.compute == ComputeAssignment(
        Inference.compute_resources,
        group=protocol.protocol_id,
    )


@B.resources(gpus=1, cpus=2, task=B.TaskType.LONG_RUNNING)
class WithBookkeeping(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return (
            self.particles.map(_identity)
            .map_element(_identity, cpu_only=True)
            .map(lambda p: {"particles": p}, cpu_only=True)
        )


def test_cpu_only_maps_keep_only_the_cpus():
    protocol = WithBookkeeping()
    assert WithBookkeeping.compute_resources is not None

    maps = _maps(protocol._tagged_steps())

    group = protocol.protocol_id
    assert {(node.cpu_only, node.compute) for node in maps} == {
        (False, ComputeAssignment(WithBookkeeping.compute_resources, group=group)),
        (True, ComputeAssignment(B.ComputeResources(cpus=2), group=group)),
    }


@pytest.mark.parametrize(
    "gpus, min_vram, claim",
    [
        (0, 8, 0.5),
        (0, 3, 0.25),
        (0, 1, 0.125),
        (0, 5, 0.5),  # Rounded up to the next fraction.
        (0, 16, 1.0),
        (1, 8, 1.0),  # The count wins over a smaller memory claim.
        (2, 8, 2.0),  # min_vram applies to every GPU.
        (0.25, 8, 0.5),  # The larger claim wins.
        (1, None, 1.0),
    ],
)
def test_gpu_claim_is_the_larger_of_count_and_memory(gpus, min_vram, claim):
    resources = B.ComputeResources(gpus=gpus, min_vram=min_vram)

    assert gpu_claim(resources, gpu_memory=16.0) == claim


def test_memory_larger_than_a_gpu_claims_whole_gpus_with_a_warning(caplog):
    claim = gpu_claim(B.ComputeResources(min_vram=20), gpu_memory=16.0)

    assert claim == 1.0
    assert "does not fit on a GPU with 16" in caplog.text


def test_unknown_gpu_memory_claims_whole_gpus_with_a_warning(caplog):
    claim = gpu_claim(B.ComputeResources(min_vram=4), gpu_memory=None)

    assert claim == 1.0
    assert "Cannot read the GPU memory" in caplog.text


def test_decorator_passes_min_vram():
    @B.resources(min_vram=6)
    class Small(B.Protocol):
        def outputs(self):
            return {}

        def steps(self):
            return Source("x")

    assert Small.compute_resources == B.ComputeResources(min_vram=6)


def test_a_share_of_a_gpu_limits_preallocation():
    assert gpu_memory_env(0.5) == {
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.450",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "SCIPION_BRIDGE_GPU_FRACTION": "0.5",
    }


@pytest.mark.parametrize("num_gpus", [0, 1, 2])
def test_no_or_whole_gpus_leave_preallocation_alone(num_gpus):
    assert gpu_memory_env(num_gpus) == {}


def test_numpy_hugepages_are_off_by_default(monkeypatch):
    monkeypatch.delenv(NUMPY_HUGEPAGE_ENV_VAR, raising=False)

    assert numpy_hugepage_env() == {"NUMPY_MADVISE_HUGEPAGE": "0"}


@pytest.mark.parametrize("value", ["0", "1"])
def test_numpy_hugepages_set_by_the_user_are_passed_through(monkeypatch, value):
    monkeypatch.setenv(NUMPY_HUGEPAGE_ENV_VAR, value)

    assert numpy_hugepage_env() == {"NUMPY_MADVISE_HUGEPAGE": value}


@pytest.mark.parametrize(("value", "hint"), [(None, False), ("0", False), ("1", True)])
def test_configure_numpy_hugepages_switches_numpy_at_runtime(
    monkeypatch,
    numpy_hugepage_hint,
    value,
    hint,
):
    match value:
        case None:
            monkeypatch.delenv(NUMPY_HUGEPAGE_ENV_VAR, raising=False)
        case _:
            monkeypatch.setenv(NUMPY_HUGEPAGE_ENV_VAR, value)
    _set_madvise_hugepage(not hint)

    configure_numpy_hugepages()

    # _set_madvise_hugepage returns the previous state of the hint.
    assert _set_madvise_hugepage(hint) is hint


def test_ray_options_turn_numpy_hugepages_off_next_to_the_other_variables(
    monkeypatch,
):
    monkeypatch.delenv(NUMPY_HUGEPAGE_ENV_VAR, raising=False)

    options = _ray_options(B.ComputeResources(gpus=0.5, cpus=2), 0.5)

    assert options["runtime_env"]["env_vars"] == {
        "NUMPY_MADVISE_HUGEPAGE": "0",
        "SCIPION_BRIDGE_CPUS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.450",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "SCIPION_BRIDGE_GPU_FRACTION": "0.5",
    }
