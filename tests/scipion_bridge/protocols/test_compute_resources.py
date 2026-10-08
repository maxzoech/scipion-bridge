"""Declaring compute resources on protocols and assigning them to stages."""

import pytest

import scipion_bridge as B
from scipion_bridge.core.environment.compute import ComputeAssignment
from scipion_bridge.core.streaming.ir import IRMap
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import MapElementOp, MapOp, lineage
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


@pytest.mark.parametrize("kwargs", [{"gpus": -1}, {"cpus": -0.5}])
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
