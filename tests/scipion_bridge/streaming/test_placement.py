"""The placement planner, in-process: pure functions of a graph and a cluster."""

import random

import pytest

from scipion_bridge.backend.ray.placement import (
    Bundle,
    ClusterSpec,
    all_maps,
    blockers_of,
    bundles_of,
    demands_of,
    explain_placement,
    placement_layer,
    plan_placement,
)
from scipion_bridge.core.environment.compute import (
    ComputeAssignment,
    ComputeResources,
    TaskType,
)
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    GpuMapPanel,
    IRDemux,
    IRMap,
    IROp,
    IRSink,
    IRSource,
    report_of,
)

LONG = TaskType.LONG_RUNNING


def _f(x):
    return x


def _map(index, resources):
    return IRMap(
        func=_f,
        name=f"m{index}",
        compute=(
            None
            if resources is None
            else ComputeAssignment(resources, f"g{index}", name=f"P{index}")
        ),
    )


def _chain(*resources):
    """source -> one map per entry (None: without resources) -> sink."""
    last: IROp = IRSource(name="x")
    for index, entry in enumerate(resources):
        node = _map(index, entry)
        last.add_downstream(node)
        last = node
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    last.add_downstream(sink)
    return [sink]


def _plan(sinks, node_gpus, cpus=8.0, weights=None):
    graph = report_of(sinks)
    demands = demands_of(all_maps(sinks), lambda resources: resources.gpus)
    return plan_placement(graph, demands, node_gpus, cpus, weights)


def _gpus(plan, group):
    return [str(slot) for slot in plan.groups[group].slots]


# -- Anti-affinity and packing -----------------------------------------------
def test_long_running_groups_get_gpus_of_their_own():
    plan = _plan(
        _chain(
            ComputeResources(gpus=0.5, task=LONG),
            ComputeResources(gpus=0.5, task=LONG),
        ),
        [2],
    )

    assert _gpus(plan, "g0") == ["node 0 GPU 0"]
    assert _gpus(plan, "g1") == ["node 0 GPU 1"]
    assert plan.warnings == ()


def test_long_running_groups_share_a_gpu_with_a_warning_if_they_must():
    plan = _plan(
        _chain(
            ComputeResources(gpus=0.5, task=LONG),
            ComputeResources(gpus=0.5, task=LONG),
        ),
        [1],
    )

    assert _gpus(plan, "g0") == _gpus(plan, "g1") == ["node 0 GPU 0"]
    assert [g.phase for g in plan.groups.values()] == [0, 0]
    assert any("time-slice" in warning for warning in plan.warnings)


def test_replicas_of_a_group_fill_a_gpu_before_the_next():
    plan = _plan(_chain(ComputeResources(gpus=0.5, task=LONG, replicas=3)), [2])

    assert plan.groups["g0"].replicas == 3
    assert [[o.share for o in slot.occupants] for slot in plan.gpu_maps[0]] == [
        [1.0],
        [0.5],
    ]


def test_a_multi_gpu_claim_needs_one_node_with_enough_gpus():
    with pytest.raises(ValueError, match="no node"):
        _plan(_chain(ComputeResources(gpus=2, task=LONG)), [1, 1])

    plan = _plan(_chain(ComputeResources(gpus=2, task=LONG)), [1, 2])
    assert _gpus(plan, "g0") == ["node 1 GPU 0,1"]


def test_cpus_are_counted():
    sinks = _chain(
        ComputeResources(cpus=2, task=LONG),
        ComputeResources(cpus=2, task=LONG),
    )

    assert len(_plan(sinks, [0], cpus=4).phases) == 1
    assert len(_plan(sinks, [0], cpus=3).phases) == 2


# -- Phases ------------------------------------------------------------------
def test_groups_that_do_not_fit_at_once_run_in_phases():
    # MotionCorr -> picking (CPU) -> 2D classification on one GPU.
    plan = _plan(
        _chain(
            ComputeResources(gpus=1, task=LONG),
            None,
            ComputeResources(gpus=1, task=LONG),
        ),
        [1],
    )

    assert plan.phases == (("g0",), ("g2",))
    assert plan.node_phases["2:m1"] == 0
    assert plan.boundaries == {("2:m1", "3:m2")}


def test_an_ephemeral_group_without_a_gpu_left_moves_to_the_next_phase():
    plan = _plan(
        _chain(
            ComputeResources(gpus=1, task=LONG, replicas=2),
            ComputeResources(gpus=1),
        ),
        [2],
    )

    assert plan.groups["g1"].phase == 1
    assert plan.groups["g1"].replicas == 2


def test_a_long_running_group_does_not_take_the_gpus_of_one_upstream():
    # The long-running group would hold both GPUs while the ephemeral group
    # feeding it waits for one: it runs in the next phase instead.
    plan = _plan(
        _chain(
            ComputeResources(gpus=1),
            ComputeResources(gpus=1, task=LONG, replicas=2),
        ),
        [2],
    )

    assert [plan.groups[g].phase for g in ("g0", "g1")] == [0, 1]
    assert plan.groups["g0"].replicas == 2


def test_ephemeral_groups_compete_without_phases():
    # Ephemeral groups do not hold their GPUs: they share them over time.
    plan = _plan(
        _chain(
            ComputeResources(gpus=1, replicas=2),
            ComputeResources(gpus=1, replicas=2),
        ),
        [2],
    )

    assert len(plan.phases) == 1
    assert [plan.groups[g].replicas for g in ("g0", "g1")] == [2, 2]


def test_parallel_branches_get_phases_in_graph_order():
    source = IRSource(name="x")
    first = _map(0, ComputeResources(gpus=1, task=LONG))
    second = _map(1, ComputeResources(gpus=1, task=LONG))
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    for branch in (first, second):
        source.add_downstream(branch)
        branch.add_downstream(sink)

    plan = _plan([sink], [1])

    assert [plan.groups[g].phase for g in ("g0", "g1")] == [0, 1]
    assert plan.node_phases[sink_id(plan)] == 1


def sink_id(plan):
    return next(node for node in plan.node_phases if "sink" in node)


def _random_pipeline(seed):
    """A random DAG of 8 maps with random claims, and a random cluster."""
    rng = random.Random(seed)
    source = IRSource(name="x")
    nodes: list = [source]
    for index in range(8):
        node = _map(
            index,
            ComputeResources(
                gpus=rng.choice([0, 0.25, 0.5, 1]),
                task=rng.choice([LONG, TaskType.EPHEMERAL]),
                replicas=rng.choice([None, None, "auto"]),
            ),
        )
        for upstream in rng.sample(nodes, k=min(len(nodes), rng.randint(1, 2))):
            upstream.add_downstream(node)
        nodes.append(node)
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    nodes[-1].add_downstream(sink)
    sinks = [sink, *(n for n in nodes if not n.downstream)]
    return sinks, [rng.randint(1, 3) for _ in range(rng.randint(1, 2))]


@pytest.mark.parametrize("seed", range(20))
def test_no_group_runs_in_an_earlier_phase_than_one_upstream_of_it(seed):
    sinks, node_gpus = _random_pipeline(seed)

    plan = _plan(sinks, node_gpus)

    graph = report_of(sinks)
    for edge in graph.edges:
        assert plan.node_phases[edge.source] <= plan.node_phases[edge.target]
    for node in graph.nodes:
        match node.group:
            case str(group) if group in plan.groups:
                assert plan.groups[group].phase == plan.node_phases[node.id]
            case _:
                pass


def test_groups_of_a_group_by_template_run_in_one_phase():
    source = IRSource(name="x")
    template_source = IRSource(name="group")
    inner = _map(1, ComputeResources(gpus=1, task=LONG))
    template_source.add_downstream(inner)
    outer = _map(0, ComputeResources(gpus=1, task=LONG))
    demux = IRDemux(template=inner, source_name="group")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(outer)
    outer.add_downstream(demux)
    demux.add_downstream(sink)

    plan = _plan([sink], [1])

    assert [plan.groups[g].phase for g in ("g0", "g1")] == [0, 1]
    assert plan.node_phases["2:group_by/1:m1"] == 1
    assert plan.boundaries == {("1:m0", "2:group_by")}


# -- Unsatisfiable claims ----------------------------------------------------
@pytest.mark.parametrize(
    ("resources", "node_gpus", "message"),
    [
        (ComputeResources(gpus=3, task=LONG), [2], "no node"),
        (ComputeResources(gpus=1.5, task=LONG), [2], "whole number"),
        (ComputeResources(gpus=1, task=LONG, replicas=3), [2], "at once"),
        (ComputeResources(cpus=16, task=LONG), [1], "reserves 16 CPUs"),
    ],
)
def test_claims_no_phase_satisfies_raise(resources, node_gpus, message):
    with pytest.raises(ValueError, match=message):
        _plan(_chain(resources), node_gpus)


# -- Elastic groups ----------------------------------------------------------
def test_ephemeral_groups_split_the_free_gpus_equally():
    plan = _plan(
        _chain(ComputeResources(gpus=1), ComputeResources(gpus=0.5)),
        [2],
    )

    assert _gpus(plan, "g0") == ["node 0 GPU 0"]
    assert _gpus(plan, "g1") == ["node 0 GPU 1"]
    assert [plan.groups[g].replicas for g in ("g0", "g1")] == [1, 2]


def test_ephemeral_groups_split_the_free_gpus_by_weight():
    plan = _plan(
        _chain(ComputeResources(gpus=1), ComputeResources(gpus=1)),
        [4],
        weights={"g0": 3.0, "g1": 1.0},
    )

    assert [plan.groups[g].replicas for g in ("g0", "g1")] == [3, 1]


@pytest.mark.parametrize(
    "resources, replicas",
    [
        # One ephemeral pool on the 2 GPUs.
        ([ComputeResources(gpus=1)], [2]),
        ([ComputeResources(gpus=0.5)], [4]),
        # A long-running pool holds one GPU.
        ([ComputeResources(gpus=1, task=LONG), ComputeResources(gpus=1)], [1, 1]),
        # Ephemeral pools share in equal parts.
        ([ComputeResources(gpus=1), ComputeResources(gpus=1)], [1, 1]),
        ([ComputeResources(gpus=0.5)] * 3, [1, 1, 1]),
        ([ComputeResources(gpus=1, replicas=3), ComputeResources(gpus=1)], [3, 2]),
        ([ComputeResources(gpus=1, task=LONG)], [1]),
    ],
)
def test_the_replicas_of_the_pools(resources, replicas):
    plan = _plan(_chain(*resources), [2])

    assert [plan.groups[f"g{i}"].replicas for i in range(len(resources))] == replicas


def test_an_ephemeral_group_gets_one_executor_at_least():
    plan = _plan(
        _chain(
            ComputeResources(gpus=1),
            ComputeResources(gpus=1),
            ComputeResources(gpus=1),
        ),
        [2],
    )

    assert all(pool.replicas >= 1 for pool in plan.groups.values())
    assert any("shared" in warning for warning in plan.warnings)


def test_auto_replicas_fill_the_gpus_the_others_leave():
    plan = _plan(
        _chain(
            ComputeResources(gpus=1, task=LONG),
            ComputeResources(gpus=0.5, task=LONG, replicas="auto"),
        ),
        [3],
    )

    assert plan.groups["g0"].replicas == 1
    assert plan.groups["g1"].replicas == 4


def test_declared_replicas_never_change():
    plan = _plan(
        _chain(
            ComputeResources(gpus=1, task=LONG, replicas=2),
            ComputeResources(gpus=1, replicas=3),
        ),
        [4],
    )

    assert [plan.groups[g].replicas for g in ("g0", "g1")] == [2, 3]


def test_the_plan_is_deterministic():
    def plan():
        return _plan(
            _chain(
                ComputeResources(gpus=0.5, task=LONG),
                ComputeResources(gpus=1),
                ComputeResources(gpus=0.25, task=LONG, replicas="auto"),
            ),
            [2, 2],
        )

    assert plan() == plan()


def test_auto_replicas_take_whole_executors_and_leave_room_for_ephemeral_groups():
    # The long-running group's share of the free GPUs is 2/3: it keeps one
    # whole GPU, and the ephemeral groups split the other.
    sinks = _chain(
        ComputeResources(gpus=1),
        ComputeResources(gpus=1, task=LONG, replicas="auto"),
        ComputeResources(gpus=1),
    )
    plan = _plan(sinks, [2])
    demands = demands_of(all_maps(sinks), lambda resources: resources.gpus)

    assert plan.groups["g1"].replicas == 1
    assert bundles_of(plan.groups["g1"], demands["g1"]) == (Bundle(0, {"GPU": 1.0}, 1),)
    held = {gpu for slot in plan.groups["g1"].slots for gpu in slot.gpus}
    for group in ("g0", "g2"):
        assert not held & {g for s in plan.groups[group].slots for g in s.gpus}


# -- Bundles and blockers ----------------------------------------------------
def _demands(sinks):
    return demands_of(all_maps(sinks), lambda resources: resources.gpus)


@pytest.mark.parametrize(
    "resources, node_gpus, bundles",
    [
        # Executors of a fraction fill one bundle per GPU, which reserves
        # the GPU whole: Ray would pack fractions of other pools beside them.
        (
            ComputeResources(gpus=0.5, task=LONG, replicas=3),
            [2],
            (Bundle(0, {"GPU": 1.0}, 2), Bundle(0, {"GPU": 1.0}, 1)),
        ),
        (
            ComputeResources(gpus=1, task=LONG, replicas=2, cpus=2),
            [2],
            (
                Bundle(0, {"GPU": 1.0, "CPU": 2}, 1),
                Bundle(0, {"GPU": 1.0, "CPU": 2}, 1),
            ),
        ),
        # A claim of several GPUs is one bundle, on a node with enough.
        (
            ComputeResources(gpus=2, task=LONG),
            [1, 2],
            (Bundle(1, {"GPU": 2.0}, 1),),
        ),
        # Ephemeral pools and pools without GPUs reserve nothing.
        (ComputeResources(gpus=1), [2], ()),
        (ComputeResources(cpus=2, task=LONG), [2], ()),
    ],
)
def test_the_bundles_of_a_pool(resources, node_gpus, bundles):
    sinks = _chain(resources)
    plan = _plan(sinks, node_gpus)

    assert bundles_of(plan.groups["g0"], _demands(sinks)["g0"]) == bundles


def test_pools_sharing_a_gpu_reserve_what_their_executors_claim():
    sinks = _chain(
        ComputeResources(gpus=0.5, task=LONG),
        ComputeResources(gpus=0.5, task=LONG),
    )
    plan = _plan(sinks, [1])

    for group in ("g0", "g1"):
        assert bundles_of(plan.groups[group], _demands(sinks)[group]) == (
            Bundle(0, {"GPU": 0.5}, 1),
        )


def test_a_later_pool_waits_only_for_the_pools_on_its_gpus():
    # On one GPU, P2 waits for P0. On two GPUs, P0 and P1 run at once, and
    # P2 waits for the one on its GPU.
    sinks = _chain(
        ComputeResources(gpus=1, task=LONG),
        ComputeResources(gpus=1, task=LONG),
        ComputeResources(gpus=1, task=LONG),
    )
    one = _plan(sinks, [1])
    two = _plan(sinks, [2])

    assert blockers_of(one, _demands(sinks)) == {
        "g0": frozenset(),
        "g1": {"g0"},
        "g2": {"g0", "g1"},
    }
    assert [two.groups[g].phase for g in ("g0", "g1", "g2")] == [0, 0, 1]
    assert blockers_of(two, _demands(sinks))["g2"] == {"g0"}


def test_pools_reserving_cpus_wait_for_each_other():
    sinks = _chain(
        ComputeResources(cpus=2, task=LONG),
        ComputeResources(cpus=2, task=LONG),
    )

    assert blockers_of(_plan(sinks, [0], cpus=3), _demands(sinks))["g1"] == {"g0"}


@pytest.mark.parametrize("seed", range(30))
def test_reservations_fit_and_leave_room_for_ephemeral_pools(seed):
    sinks, node_gpus = _random_pipeline(seed)
    demands = _demands(sinks)

    plan = _plan(sinks, node_gpus)

    for phase, groups in enumerate(plan.phases):
        held: dict = {}
        for group in groups:
            demand, pool = demands[group], plan.groups[group]
            bundles = bundles_of(pool, demand)
            if bundles:
                assert sum(b.executors for b in bundles) == pool.replicas
            for slot, bundle in zip(
                [s for s in pool.slots if s.executors > 0], bundles
            ):
                for gpu in slot.gpus:
                    use = bundle.resources["GPU"] / len(slot.gpus)
                    held[(slot.node, gpu)] = held.get((slot.node, gpu), 0) + use
        assert all(use <= 1 + 1e-9 for use in held.values()), (phase, held)
        for group in groups:
            demand = demands[group]
            if demand.holds or demand.gpus == 0:
                continue
            free = [
                1 - held.get((node, gpu), 0)
                for node, count in enumerate(node_gpus)
                for gpu in range(count)
            ]
            assert max(free) + 1e-9 >= min(demand.gpus, 1.0), (group, held)


def test_replicas_auto_is_a_valid_declaration():
    assert ComputeResources(replicas="auto").replicas == "auto"
    for invalid in ["max", 0]:
        with pytest.raises(ValueError, match="replica"):
            ComputeResources(replicas=invalid)  # type: ignore[arg-type]


# -- The report layer --------------------------------------------------------
def test_the_layer_shows_phases_boundaries_and_gpus():
    sinks = _chain(
        ComputeResources(gpus=1, task=LONG),
        None,
        ComputeResources(gpus=1, task=LONG),
    )
    graph = report_of(sinks)
    demands = demands_of(all_maps(sinks), lambda resources: resources.gpus)
    plan = plan_placement(graph, demands, [1], 8.0)

    layer = placement_layer(graph, plan, demands)
    report = graph.with_layer(layer)

    assert layer.nodes["3:m2"]["phase"] == 1
    assert layer.nodes["2:m1"]["runs_as"] == "Ray tasks"
    assert layer.edges[("2:m1", "3:m2")] == {"boundary": "spills"}
    assert [panel.title for panel in layer.panels] == [
        "Pools",
        "GPUs in phase 0: P0",
        "GPUs in phase 1: P2",
    ]
    assert layer.nodes["3:m2"]["waits_for"] == "the FLUSH of phase 0"
    assert any("wait on disk" in warning for warning in report.warnings)


def test_after_a_run_the_gpus_are_named_by_the_devices_they_ran_on():
    sinks = _chain(
        ComputeResources(gpus=1, task=LONG),
        None,
        ComputeResources(gpus=1, task=LONG),
    )
    graph = report_of(sinks)
    demands = demands_of(all_maps(sinks), lambda resources: resources.gpus)
    plan = plan_placement(graph, demands, [1], 8.0)
    # With CUDA_VISIBLE_DEVICES=3, the one GPU of the plan is device 3.
    devices = {group: {0: ("3",)} for group in plan.groups}

    planned = placement_layer(graph, plan, demands)
    ran = placement_layer(graph, plan, demands, devices=devices)

    gpu_maps = [p for p in ran.panels if isinstance(p, GpuMapPanel)]
    assert [slot.name for p in gpu_maps for slot in p.gpus] == ["GPU 3", "GPU 3"]
    planned_maps = [p for p in planned.panels if isinstance(p, GpuMapPanel)]
    assert [slot.name for p in planned_maps for slot in p.gpus] == ["GPU 0", "GPU 0"]


def test_explain_plans_without_ray():
    report = explain_placement(
        _chain(
            ComputeResources(gpus=0.5, task=LONG),
            ComputeResources(gpus=0.5, task=LONG),
        ),
        ClusterSpec(node_gpus=(2,), cpus=8),
    )

    groups = report.layer("placement").groups
    assert groups["g0"]["gpus"] == ("node 0 GPU 0",)
    assert groups["g1"]["gpus"] == ("node 0 GPU 1",)


def test_explain_resolves_memory_claims_for_the_smallest_gpu():
    report = explain_placement(
        _chain(ComputeResources(min_vram=8, task=LONG, replicas=2)),
        ClusterSpec(node_gpus=(1,), cpus=8, gpu_memory=(16.0,)),
    )

    assert report.layer("placement").groups["g0"]["gpus_per_executor"] == 0.5
