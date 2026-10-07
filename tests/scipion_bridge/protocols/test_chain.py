import pytest

import scipion_bridge as B
from scipion_bridge import ChainedProtocol, Protocol
from scipion_bridge.core.protocol.chain_utils import ChainError
from scipion_bridge.core.streaming import LoweringContext, Source
from scipion_bridge.single_particle.particle import Particle


class Stage1(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()
    scale: B.Field[float] = B.Field(default=1.0)

    def outputs(self):
        return {"particles": B.Set[Particle], "other": B.Set[Particle]}

    def steps(self):
        return self.particles.map(lambda p: {"particles": p, "other": p})


class Stage2(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()
    mask: B.Input[B.Set[Particle]] = B.Input()
    scale: B.Field[float] = B.Field(default=1.0)

    def outputs(self):
        return {"result": B.Set[Particle]}

    def steps(self):
        op = self.particles.map(lambda p: {"result": p})
        self.mask.op(op)
        return op


class SingleOut(Protocol):
    micrographs: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"picked": B.Set[Particle]}

    def steps(self):
        return self.micrographs.map(lambda p: {"picked": p})


class SingleIn(Protocol):
    coordinates: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"result": B.Set[Particle]}

    def steps(self):
        return self.coordinates.map(lambda p: {"result": p})


class ConflictingParameter(Protocol):
    scale: B.Field[float] = B.Field(default=2.0)
    micrographs: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"result": B.Set[Particle]}

    def steps(self):
        return self.micrographs.map(lambda p: {"result": p})


def _lowered_source_names(op):
    ctx = LoweringContext()
    ctx.lower_node(op)
    return sorted(ctx.sources)


def _source_names(sink_op):
    seen, stack, names = set(), [sink_op], []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if isinstance(node, Source):
            names.append(node.name)
        stack.extend(node.upstream)
    return names


def test_or_returns_protocol():
    chain = Stage1() | Stage2()
    assert isinstance(chain, ChainedProtocol)
    assert isinstance(chain, Protocol)


def test_or_with_non_protocol_is_not_implemented():
    with pytest.raises(TypeError):
        Stage1() | 3  # type: ignore[operator]


def test_configuration_merge():
    chain = Stage1() | Stage2()
    config = chain.configuration

    assert list(config.inputs) == ["particles", "mask"]
    assert list(config.parameters) == ["scale"]
    assert chain.outputs() == Stage2().outputs()


def test_wiring_removes_input_of_second():
    chain = Stage1().pipe(Stage2(), mapping={"other": "mask"})
    assert list(chain.configuration.inputs) == ["particles"]


def test_single_output_wired_by_type():
    chain = SingleOut() | SingleIn()
    assert list(chain.configuration.inputs) == ["micrographs"]


def test_ambiguous_single_output_raises():
    with pytest.raises(ChainError):
        SingleOut() | Stage2()


def test_no_match_raises():
    with pytest.raises(ChainError):
        Stage1() | SingleIn()


def test_unknown_mapping_raises():
    with pytest.raises(ChainError):
        Stage1().pipe(Stage2(), mapping={"missing": "mask"})

    with pytest.raises(ChainError):
        Stage1().pipe(Stage2(), mapping={"other": "missing"})


def test_conflicting_parameter_raises():
    with pytest.raises(ChainError):
        Stage1().pipe(ConflictingParameter(), mapping={"other": "micrographs"})


def test_shared_parameter_is_merged():
    chain = Stage1() | Stage2()
    assert list(chain.configuration.parameters) == ["scale"]


def test_steps_have_one_source_per_name():
    chain = Stage1() | Stage2()
    names = _source_names(chain.steps())

    assert sorted(names) == ["mask", "particles"]


def test_steps_lower_without_duplicate_sources():
    chain = Stage1() | Stage2()

    assert _lowered_source_names(chain.steps()) == ["mask", "particles"]


def test_chain_is_associative():
    a, b, c = Stage1(), Stage2(), SingleIn()
    left = (a | b).pipe(c, mapping={"result": "coordinates"})
    right = a | b.pipe(c, mapping={"result": "coordinates"})

    assert list(left.configuration.inputs) == list(right.configuration.inputs)
    assert left.outputs() == right.outputs()


class RepeatedAccess(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"result": B.Set[Particle]}

    def steps(self):
        op = self.particles.map(lambda p: {"result": p})
        self.particles.op(op)
        return op


def test_repeated_input_access_is_deduplicated():
    chain = SingleOut().pipe(RepeatedAccess(), mapping={"picked": "particles"})
    steps = chain.steps()

    assert _source_names(steps) == ["micrographs"]
