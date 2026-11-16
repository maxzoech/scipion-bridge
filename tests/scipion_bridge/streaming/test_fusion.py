"""The operator-fusion pass (in-process), and one Ray pipeline it compiles.

No operator of the library is fusable yet (``astype`` will be), so the tests
build IR with ``IRMap(fusable=True)`` directly.
"""

import pickle

import pytest
import ray

from scipion_bridge.backend.ray import RayBackend
from scipion_bridge.core.environment import ComputeAssignment, ComputeResources
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    FusionPolicy,
    IRAccumulate,
    IRDemux,
    IRMap,
    IRSink,
    IRSource,
    Keyed,
    NoFusion,
    Op,
    Pipeline,
    Source,
    lower,
    report_of,
    share_keys,
    topological_order,
)
from scipion_bridge.core.streaming.fusion import fuse
from scipion_bridge.core.streaming.ir import describe

DEFAULT = FusionPolicy()


def _double(x):
    return x * 2


def _inc(x):
    return x + 1


def _square(x):
    return x * x


def _cheap(func=_inc, name=None):
    return IRMap(func=func, name=name or f"cheap({func.__name__})", fusable=True)


def _map(func=_double):
    return IRMap(func=func, name=f"map({func.__name__})")


def _sink():
    return IRSink(writer=CallbackSinkWriter(lambda item: None))


def _chain(*nodes):
    for upstream, downstream in zip(nodes, nodes[1:]):
        upstream.add_downstream(downstream)
    return nodes


def _pairs(n):
    """An accumulator emitting pairs of items as their sum; flush emits the rest."""

    def accumulate(state, item):
        state = [*state, item]
        return ([], [sum(state)]) if len(state) == n else (state, [])

    def flush(state):
        return [], [sum(state)] if state else []

    return IRAccumulate(
        accumulate_fn=accumulate,
        initial_state_fn=list,
        flush_fn=flush,
        name=f"pairs({n})",
    )


def _run(accumulate, items):
    """The emissions of an accumulator for ``items``, then on flush."""
    state = accumulate.initial_state_fn()
    emitted = []
    for item in items:
        state, out = accumulate.accumulate_fn(state, item)
        emitted.extend(out)
    _, out = accumulate.flush_fn(state)
    return emitted + out


def _stages(exits):
    return topological_order(exits)


# -- Into the upstream stage ---------------------------------------------------
def test_a_fusable_map_runs_inside_the_map_upstream_of_it():
    compute = ComputeAssignment(ComputeResources(cpus=1), group="p", name="P")
    source, host, cheap, sink = _chain(
        IRSource(name="items"),
        IRMap(func=_double, name="map(_double)", compute=compute, stage_fp="fp"),
        _cheap(),
        _sink(),
    )

    (exit_,) = fuse([sink], DEFAULT)

    assert exit_ is sink
    assert _stages([sink]) == [source, host, sink]
    assert host.func(3) == 7
    assert host.name == "map(_double) → cheap(_inc)"
    assert host.fused == ("map(_double)", "cheap(_inc)")
    assert (host.stage_fp, host.compute) == ("fp", compute)
    assert sink.upstream == [host] and host.downstream == [sink]
    assert cheap.upstream == cheap.downstream == []


def test_a_fusable_map_maps_the_emissions_of_the_accumulator_upstream_of_it():
    source, pairs, cheap, sink = _chain(
        IRSource(name="items"),
        _pairs(2),
        _cheap(_square),
        _sink(),
    )

    fuse([sink], DEFAULT)

    assert _stages([sink]) == [source, pairs, sink]
    # Pairs 1+2 and 3+4 are emitted with items, the incomplete 5 on flush.
    assert _run(pairs, [1, 2, 3, 4, 5]) == [9, 49, 25]
    assert pairs.name == "pairs(2) → cheap(_square)"


def test_an_accumulator_without_a_flush_keeps_none():
    source, accumulate, cheap, sink = _chain(
        IRSource(name="items"),
        IRAccumulate(name="pass"),
        _cheap(),
        _sink(),
    )

    fuse([sink], DEFAULT)

    assert accumulate.flush_fn is None
    assert accumulate.accumulate_fn(None, 1) == (None, [2])


def test_the_host_takes_over_the_downstream_stages_on_their_ports():
    source, host, cheap = _chain(IRSource(name="items"), _map(), _cheap())
    other = IRSource(name="other")
    merge, sink_a = IRMap(func=_double, name="merge"), _sink()
    other.add_downstream(merge)
    cheap.add_downstream(merge)
    cheap.add_downstream(sink_a)

    fuse([merge, sink_a], DEFAULT)

    assert host.downstream == [merge, sink_a]
    assert merge.upstream == [other, host]
    assert sink_a.upstream == [host]


# -- Into the downstream map ---------------------------------------------------
def test_a_fusable_map_after_a_source_runs_inside_the_map_downstream_of_it():
    compute = ComputeAssignment(ComputeResources(cpus=1), group="p", name="P")
    source, cheap, host, sink = _chain(
        IRSource(name="items"),
        _cheap(),
        IRMap(func=_double, name="map(_double)", compute=compute),
        _sink(),
    )

    fuse([sink], DEFAULT)

    assert _stages([sink]) == [source, host, sink]
    assert host.func(3) == 8
    assert host.name == "cheap(_inc) → map(_double)"
    assert host.compute == compute
    assert source.downstream == [host] and host.upstream == [source]


def test_chains_of_fusable_maps_collapse_into_one_stage():
    source, first, second, host, sink = _chain(
        IRSource(name="items"),
        _cheap(_inc, "a"),
        _cheap(_square, "b"),
        _map(),
        _sink(),
    )

    fuse([sink], DEFAULT)

    assert _stages([sink]) == [source, host, sink]
    assert host.func(2) == 18
    assert host.fused == ("a", "b", "map(_double)")


# -- Not fused -----------------------------------------------------------------
def _alone(*nodes):
    """Fuse the graph of ``nodes`` (its last one the exit) and assert the
    fusable maps among them stay stages of their own."""
    before = {node: (list(node.upstream), list(node.downstream)) for node in nodes}
    fuse([nodes[-1]], DEFAULT)
    assert {n: (n.upstream, n.downstream) for n in nodes} == before
    assert all(not node.fused for node in nodes)


def test_a_fusable_map_between_a_source_and_a_sink_stays_alone():
    _alone(*_chain(IRSource(name="items"), _cheap(), _sink()))


def test_a_fusable_map_behind_a_fan_out_stays_alone():
    source, host, cheap, sink = _chain(
        IRSource(name="items"), _map(), _cheap(), _sink()
    )
    host.add_downstream(other := _sink())

    fuse([sink, other], DEFAULT)

    assert cheap.upstream == [host] and host.downstream == [cheap, other]


def test_a_fusable_map_is_not_fused_into_a_merging_map():
    source, cheap = _chain(IRSource(name="items"), _cheap())
    merge, sink = IRMap(func=_double, name="merge"), _sink()
    cheap.add_downstream(merge)
    IRSource(name="other").add_downstream(merge)
    merge.add_downstream(sink)

    fuse([sink], DEFAULT)

    assert cheap.downstream == [merge] and not merge.fused


def test_a_fusable_map_is_not_fused_into_a_group_by():
    template_source = IRSource(name="group")
    template_exit = _map()
    template_source.add_downstream(template_exit)
    demux = IRDemux(template=template_exit, source_name="group")
    _alone(*_chain(IRSource(name="items"), _cheap(), demux, _cheap(), _sink()))


def test_the_policy_decides_on_every_legal_move():
    class Recording(FusionPolicy):
        def __init__(self):
            self.asked = []

        def fuse(self, op, into):
            self.asked.append((describe(op), describe(into)))
            return False

    policy = Recording()
    source, cheap, host, sink = _chain(
        IRSource(name="items"), _cheap(), _map(), _sink()
    )

    fuse([sink], policy)

    assert policy.asked == [("cheap(_inc)", "map(_double)")]
    assert _stages([sink]) == [source, cheap, host, sink]


def test_no_fusion_keeps_every_stage():
    source, host, cheap, sink = _chain(
        IRSource(name="items"), _map(), _cheap(), _sink()
    )

    fuse([sink], NoFusion())

    assert _stages([sink]) == [source, host, cheap, sink]


def test_a_fusable_map_cannot_require_resources():
    with pytest.raises(ValueError, match="requires resources"):
        IRMap(
            func=_inc,
            fusable=True,
            compute=ComputeAssignment(ComputeResources(cpus=1), group="p"),
        )


# -- group_by templates --------------------------------------------------------
def test_the_maps_of_a_group_by_template_fuse_among_themselves():
    template_source, host, cheap = _chain(IRSource(name="group"), _map(), _cheap())
    demux = IRDemux(template=cheap, source_name="group")
    _chain(IRSource(name="items"), demux, sink := _sink())

    fuse([sink], DEFAULT)

    # The exit of the template was fused away: the template ends in its host.
    assert demux.template is host
    assert host.func(3) == 7


def test_shared_keyed_pipelines_wrap_the_fused_functions():
    template_source, pairs, cheap = _chain(IRSource(name="group"), _pairs(2), _cheap())
    (exit_,) = fuse([cheap], DEFAULT)

    shared = share_keys(exit_)

    assert isinstance(shared, IRAccumulate)
    assert shared.name == "pairs(2) → cheap(_inc)"
    state = shared.initial_state_fn()
    state, out = shared.accumulate_fn(state, Keyed("a", 1))
    state, out = shared.accumulate_fn(state, Keyed("a", 2))
    assert out == [Keyed("a", 4)]


# -- Lowering and reports ------------------------------------------------------
class _Cheap(Op):
    """An operator lowering to a fusable map, as ``astype`` will."""

    def __init__(self):
        super().__init__(upstream=None)

    def lower(self, ctx):
        return _cheap()


def test_lowering_fuses_and_the_report_shows_the_fused_stage():
    sink_node = Source("items").chunk(2).op(_Cheap()).sink(lambda item: None)

    report = report_of(lower([sink_node]))

    assert [n.kind for n in report.nodes] == ["source", "accumulate", "sink"]
    fused = report.nodes[1]
    assert fused.title == "chunk(2) → cheap(_inc)"
    assert fused.fused == ("chunk(2)", "cheap(_inc)")


def test_lowering_without_fusion_keeps_the_fusable_map():
    sink_node = Source("items").chunk(2).op(_Cheap()).sink(lambda item: None)

    report = report_of(lower([sink_node], fusion=NoFusion()))

    assert [n.kind for n in report.nodes] == ["source", "accumulate", "map", "sink"]
    assert all(node.fused == () for node in report.nodes)


def test_fused_functions_pickle():
    host, cheap = _chain(_map(), _cheap())

    fuse([cheap], DEFAULT)

    assert pickle.loads(pickle.dumps(host.func))(3) == 7


# -- Ray -----------------------------------------------------------------------
@ray.remote
class _Collector:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def get(self):
        return sorted(self.items)


def _results(policy):
    """Run source -> cheap(_inc) -> cheap(_double) -> map(_square) -> sink."""
    collector = _Collector.remote()
    writer = CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))
    *_, sink = _chain(
        IRSource(name="items"),
        _cheap(_inc),
        _cheap(_double),
        _map(_square),
        IRSink(writer=writer),
    )
    exits = fuse([sink], policy)

    with Pipeline.from_ir(exits, backend=RayBackend(init_ray=False)) as pipeline:
        for item in range(5):
            pipeline.send(items=item)
    return ray.get(collector.get.remote()), list(pipeline.stats())


@pytest.mark.usefixtures("ray_cluster")
def test_a_fused_pipeline_computes_what_the_unfused_one_does():
    fused, fused_stages = _results(DEFAULT)
    unfused, unfused_stages = _results(NoFusion())

    assert fused == unfused == sorted(((x + 1) * 2) ** 2 for x in range(5))
    assert fused_stages[1:] == [
        "1:cheap(_inc) → cheap(_double) → map(_square)",
        "2:sink(CallbackSinkWriter)",
    ]
    assert len(unfused_stages) == 5
