"""Sealing protocols once their pipeline is built, and their fingerprints."""

from typing import Any

import cloudpickle  # type: ignore
from dependency_injector import providers
import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.environment.protocol_config import (
    StaticProtocolConfigurationProvider,
)
from scipion_bridge.core.protocol.fingerprint import class_identity
from scipion_bridge.core.streaming.ir import IRMap, IROp, clone_ir
from scipion_bridge.core.streaming.keyed import share_keys
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.single_particle.particle import Particle


def _identity(item):
    return item


class Recording(B.Protocol):
    """Records its hooks with ``record``, a callable outside of the protocol."""

    particles: B.Input[B.Set[Particle]] = B.Input()
    extra: int = 0

    def __init__(self, record):
        super().__init__()
        self.record = record

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def validate_protocol_configuration(self):
        self.record(("validate", type(self).__name__))

    def setup(self):
        self.record(("setup", type(self).__name__))
        self.prepared = True

    def steps(self):
        self.record(("steps", type(self).__name__))
        self.built = True
        return self.particles.map(lambda p: {"particles": p})


class RecordingSecond(Recording):
    pass


class RecordingThird(Recording):
    pass


class Counting(B.Protocol):
    """Keeps a count in a map, which the seal forbids."""

    particles: B.Input[B.Set[Particle]] = B.Input()
    count: int = 0

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def _count(self, particles):
        self.count = self.count + 1
        return {"particles": particles}

    def steps(self):
        return self.particles.map(self._count)


class Holding(B.Protocol):
    """Holds ``value``, set in ``__init__``."""

    particles: B.Input[B.Set[Particle]] = B.Input()
    scale: B.Field[float] = B.Field(default=1.0)

    def __init__(self, value: Any = None):
        super().__init__()
        self.value = value

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def _apply(self, particles):
        return {"particles": particles}

    def steps(self):
        return self.particles.map(self._apply)


class HoldingSubclass(Holding):
    pass


class TwoChunks(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    def outputs(self):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return self.particles.chunk(4).chunk(8).map(lambda p: {"particles": p})


def _maps_of(protocol):
    (sink,) = lower([protocol.get_pipeline().sink(print)])
    return [node for node in _ir_nodes(sink) if isinstance(node, IRMap)]


def _ir_nodes(exit_: IROp):
    seen = []
    stack = [exit_]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.append(node)
            stack.extend(node.upstream)
    return seen


def test_get_pipeline_runs_the_hooks_once_before_steps():
    calls = []
    Recording(calls.append).get_pipeline()

    assert calls == [
        ("validate", "Recording"),
        ("setup", "Recording"),
        ("steps", "Recording"),
    ]


def test_a_chain_runs_the_hooks_of_every_protocol_once():
    calls = []
    chain = (
        Recording(calls.append)
        | RecordingSecond(calls.append)
        | RecordingThird(calls.append)
    )

    chain.get_pipeline()

    names = ["Recording", "RecordingSecond", "RecordingThird"]
    hooks = [(hook, name) for hook, name in calls if hook != "steps"]
    assert hooks == [
        *(("validate", name) for name in names),
        *(("setup", name) for name in names),
    ]
    assert sorted(name for hook, name in calls if hook == "steps") == sorted(names)
    assert calls.index(("setup", "RecordingThird")) < calls.index(
        ("steps", "Recording"),
    )


def test_attributes_can_be_set_until_the_pipeline_is_built():
    protocol = Recording(lambda call: None)
    protocol.extra = 1

    protocol.get_pipeline()

    assert protocol.sealed
    assert (protocol.prepared, protocol.built, protocol.extra) == (True, True, 1)


def test_a_sealed_protocol_rejects_setting_and_deleting_attributes():
    protocol = Recording(lambda call: None)
    protocol.get_pipeline()

    with pytest.raises(B.SealedProtocolError, match="'prepared'"):
        protocol.prepared = False
    with pytest.raises(B.SealedProtocolError, match="'prepared'"):
        del protocol.prepared


def test_a_map_writing_to_the_protocol_raises_on_its_first_item():
    protocol = Counting()
    (count,) = [node for node in _maps_of(protocol) if node.func == protocol._count]

    with pytest.raises(B.SealedProtocolError, match="'count'"):
        count.func("particles")


def test_the_pipeline_cannot_be_built_twice():
    protocol = Recording(lambda call: None)
    protocol.get_pipeline()

    with pytest.raises(B.SealedProtocolError, match="already built"):
        protocol.get_pipeline()


def test_a_protocol_in_a_chain_cannot_build_a_pipeline_of_its_own():
    first = Recording(lambda call: None)
    (first | RecordingSecond(lambda call: None)).get_pipeline()

    with pytest.raises(B.SealedProtocolError, match="already built"):
        first.get_pipeline()


def test_a_chain_seals_itself_and_its_protocols():
    first = Recording(lambda call: None)
    second = RecordingSecond(lambda call: None)
    chain = first | second

    chain.get_pipeline()

    assert (first.sealed, second.sealed, chain.sealed) == (True, True, True)


@pytest.mark.parametrize(
    "value",
    [[1], {"a": 1}, {1}, bytearray(b"a"), (1, [2]), (frozenset({1}), {"a": 1})],
    ids=["list", "dict", "set", "bytearray", "list-in-tuple", "dict-in-tuple"],
)
def test_mutable_containers_are_rejected_when_sealing(value):
    with pytest.raises(B.SealedProtocolError, match="'value"):
        Holding(value).get_pipeline()


@pytest.mark.parametrize(
    "value",
    [(1, "a"), frozenset({1, 2}), (("key", 1.0),), None],
    ids=["tuple", "frozenset", "pairs", "none"],
)
def test_immutable_values_are_accepted(value):
    protocol = Holding(value)
    protocol.get_pipeline()

    assert protocol.value == value


def test_arrays_become_read_only_and_stay_read_only_when_pickled():
    protocol = Holding(np.zeros(3))
    protocol.get_pipeline()

    copy = cloudpickle.loads(cloudpickle.dumps(protocol))

    assert not protocol.value.flags.writeable
    assert not copy.value.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        copy.value[0] = 1.0


def test_pickled_copies_of_a_sealed_protocol_and_its_methods_are_sealed():
    Holding.__module__ = "__main__"
    try:
        protocol = Holding(1)
        protocol.get_pipeline()

        copy = cloudpickle.loads(cloudpickle.dumps(protocol))
        method = cloudpickle.loads(cloudpickle.dumps(protocol._apply))
    finally:
        Holding.__module__ = __name__

    assert copy.sealed
    assert method.__self__.sealed
    with pytest.raises(B.SealedProtocolError):
        copy.value = 2
    with pytest.raises(B.SealedProtocolError):
        method.__self__.value = 2


def test_the_error_tells_where_state_belongs():
    protocol = Recording(lambda call: None)
    protocol.get_pipeline()

    with pytest.raises(B.SealedProtocolError) as error:
        protocol.state = 1

    message = str(error.value)
    assert "'state'" in message
    assert ".scan(" in message
    assert "B.Resource" in message
    assert "Do not bypass" in message
    assert "Where state belongs" in message


def test_the_error_for_a_container_names_immutable_alternatives():
    with pytest.raises(B.SealedProtocolError) as error:
        Holding([1]).get_pipeline()

    message = str(error.value)
    assert "tuple" in message
    assert "MappingProxyType" in message
    assert ".scan(" in message


def _sealed(protocol):
    protocol.get_pipeline()
    return protocol


def test_the_fingerprint_requires_a_sealed_protocol():
    with pytest.raises(RuntimeError, match="only defined once"):
        Holding().fingerprint()


def test_identically_configured_protocols_have_equal_fingerprints():
    assert _sealed(Holding((1, 2))).fingerprint() == (
        _sealed(Holding((1, 2))).fingerprint()
    )


def test_the_fingerprint_changes_with_the_state():
    assert _sealed(Holding(1)).fingerprint() != _sealed(Holding(2)).fingerprint()


def test_the_fingerprint_changes_with_the_class():
    assert _sealed(Holding()).fingerprint() != (
        _sealed(HoldingSubclass()).fingerprint()
    )


def test_the_fingerprint_changes_with_a_parameter_value(setup_test_container):
    default = _sealed(Holding()).fingerprint()

    with setup_test_container.protocol_config_provider.override(
        providers.Singleton(StaticProtocolConfigurationProvider, values={"scale": 2.0}),
    ):
        scaled = _sealed(Holding()).fingerprint()

    assert default != scaled


def test_the_fingerprint_is_stable_when_pickled():
    protocol = _sealed(Holding(np.arange(4)))

    copy = cloudpickle.loads(cloudpickle.dumps(protocol))

    assert copy.fingerprint() == protocol.fingerprint()


def test_classes_without_source_are_identified_by_their_pickle():
    dynamic = type("Dynamic", (), {"x": 1})

    assert class_identity(dynamic) == class_identity(dynamic)


def test_every_stage_of_a_protocol_has_a_fingerprint():
    (sink,) = lower([TwoChunks().get_pipeline().sink(print)])

    stages = [node for node in _ir_nodes(sink) if node is not sink]
    fingerprints = [node.stage_fp for node in stages]

    assert None not in fingerprints
    assert len(set(fingerprints)) == len(stages)
    assert sink.stage_fp is None


def test_stages_outside_of_protocols_have_no_fingerprint():
    (sink,) = lower([Source("items").map(_identity).chunk(2).sink(print)])

    assert {node.stage_fp for node in _ir_nodes(sink)} == {None}


def test_stages_of_a_chain_keep_the_fingerprints_of_their_protocols():
    first = Recording(lambda call: None)
    second = RecordingSecond(lambda call: None)
    (sink,) = lower([(first | second).get_pipeline().sink(print)])

    fingerprints = {node.stage_fp for node in _ir_nodes(sink) if node is not sink}

    assert None not in fingerprints
    assert len(fingerprints) > 1


def test_copies_of_the_ir_keep_the_fingerprints():
    (sink,) = lower([TwoChunks().get_pipeline().sink(print)])
    exit_ = sink.upstream[0]

    (copy,) = clone_ir([exit_])
    shared = share_keys(exit_)

    original = [node.stage_fp for node in _ir_nodes(exit_)]
    assert [node.stage_fp for node in _ir_nodes(copy)] == original
    assert [node.stage_fp for node in _ir_nodes(shared)] == original
