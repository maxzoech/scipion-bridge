"""Running pipelines in phases: the helpers of the Ray backend, in-process."""

from scipion_bridge.backend.ray.backend import (
    _Placement,
    _boundary_targets,
    _host_of,
    _is_hosted,
    _pool_hosts,
    _stats_of,
)
from scipion_bridge.backend.ray.placement import all_maps, demands_of, plan_placement
from scipion_bridge.core.environment.compute import (
    ComputeAssignment,
    ComputeResources,
    TaskType,
)
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    IRDemux,
    IRMap,
    IRSink,
    IRSource,
    StageStats,
    report_of,
    stage_labels,
    topological_order,
)

LONG = TaskType.LONG_RUNNING


def _f(x):
    return x


def _map(group, gpus=1.0):
    return IRMap(
        func=_f,
        name=f"m_{group}",
        compute=ComputeAssignment(ComputeResources(gpus=gpus, task=LONG), group),
    )


def _claims(nodes):
    """The Ray options of the calls of every map, as the backend resolves them."""
    return {
        compute: {
            "num_gpus": 0 if compute is None else compute.resources.gpus,
            "num_cpus": 0,
        }
        for node in all_maps(nodes)
        for compute in [node.compute]
    }


def _pipeline():
    """source -> a -> b (both a GPU each) -> sink, on one GPU: two phases."""
    source = IRSource(name="x")
    a, b = _map("a"), _map("b")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(a)
    a.add_downstream(b)
    b.add_downstream(sink)
    nodes = topological_order([sink])
    return nodes, stage_labels(nodes), a, b


def _placement(nodes, gpus):
    graph = report_of([nodes[-1]])
    demands = demands_of(all_maps(nodes), lambda resources: resources.gpus)
    return _Placement(graph, plan_placement(graph, demands, [gpus], 4.0), demands)


def test_the_first_map_of_a_later_phase_gets_an_actor_of_its_own():
    nodes, labels, a, b = _pipeline()

    targets = _boundary_targets(nodes, labels, _placement(nodes, gpus=1))

    assert targets == {b}
    assert not _is_hosted(b, targets)
    assert _host_of(b, targets) is b
    assert _is_hosted(a, targets)


def test_one_phase_has_no_boundaries():
    nodes, labels, _, _ = _pipeline()

    assert _boundary_targets(nodes, labels, _placement(nodes, gpus=2)) == frozenset()
    assert _boundary_targets(nodes, labels, None) == frozenset()


def test_the_hosts_of_a_pool_are_the_actors_running_its_maps():
    nodes, labels, a, b = _pipeline()
    targets = frozenset({b})

    hosts = _pool_hosts(nodes, labels, _claims(nodes), targets)

    assert hosts == {"a": {"0:source(x)"}, "b": {labels[b]}}


def test_a_group_by_hosts_the_maps_of_its_template():
    # "a" runs before the group_by and in its template, as an inference
    # protocol that partitions its items.
    source = IRSource(name="x")
    before = _map("a")
    template_source = IRSource(name="group")
    inner = _map("a")
    template_source.add_downstream(inner)
    demux = IRDemux(template=inner, source_name="group")
    after = IRMap(func=_f, name="cpu")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(before)
    before.add_downstream(demux)
    demux.add_downstream(after)
    after.add_downstream(sink)
    nodes = topological_order([sink])
    labels = stage_labels(nodes)

    hosts = _pool_hosts(nodes, labels, _claims(nodes), frozenset())

    assert hosts == {"a": {"0:source(x)", labels[demux]}}


def test_maps_without_a_pool_have_no_hosts():
    source = IRSource(name="x")
    cpu = IRMap(func=_f, name="cpu")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(cpu)
    cpu.add_downstream(sink)
    nodes = topological_order([sink])

    assert _pool_hosts(nodes, stage_labels(nodes), _claims(nodes), frozenset()) == {}


def test_a_map_on_its_own_actor_reports_the_buffering_of_the_actor():
    calls = StageStats(items_in=5, process_s=2.0)
    actor = StageStats(
        items_in=5,
        buffered_peak=9,
        spilled=4,
        spill_s=0.5,
        spill_bytes=100,
        reread_s=0.25,
    )

    stats = _stats_of(calls, actor, own_actor=True)

    assert (stats.process_s, stats.buffered_peak, stats.spilled) == (2.0, 9, 4)
    assert (stats.spill_bytes, stats.reread_s) == (100, 0.25)
    assert _stats_of(calls, actor, own_actor=False) == calls
    assert _stats_of(None, actor, own_actor=False) == actor
