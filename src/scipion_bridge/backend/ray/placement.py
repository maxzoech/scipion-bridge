"""Placement of the executor pools of a pipeline on the GPUs of a cluster.

Pure functions, without Ray: ``plan_placement`` turns the compute groups of
a pipeline (``GroupDemand``) and the GPUs of a cluster into a ``Plan``. It
decides how many executors every pool gets, on which GPUs, and in which
*phase*.

Long-running groups hold their GPUs until the stream is flushed, and the
executors of a group run at once. Groups that do not fit next to each other
run in phases: a later phase starts when the earlier ones have flushed, and
the items for it wait in front of it. Phases follow the graph, so a later
phase never feeds an earlier one.

Ephemeral groups start executors on demand and release them when idle, so
they do not hold GPUs; they need room for one executor beside the
long-running groups of their phase. Ephemeral groups without declared
replicas and long-running ones with ``replicas="auto"`` are *elastic*: they
split the GPUs that the long-running groups of their phase leave.

Rules, in order:

1. Claims no circumstance satisfies raise: an executor larger than every
   node, a claim above one GPU that is not a whole number, declared replicas
   of a long-running group that do not fit the cluster even alone.
2. Long-running groups are packed in the order of the graph, first fit: on
   GPUs of their own if possible (anti-affinity: processes of different
   groups on one GPU time-slice between their CUDA contexts), else sharing
   GPUs with a warning, else in the next phase.
3. Elastic groups split the free GPUs of their phase by their weights (equal
   by default); without room for an executor, they move to the next phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from scipion_bridge.core.environment.compute import (
    ComputeAssignment,
    ComputeResources,
    Replicas,
    TaskType,
    gpu_claim,
)
from scipion_bridge.core.streaming.ir import IRDemux, IRMap, IROp
from scipion_bridge.core.streaming.report.model import (
    GpuMapPanel,
    GpuOccupant,
    GpuSlot,
    Layer,
    PipelineReport,
    TablePanel,
    Value,
    report_of,
)

# Tolerance of sums of fractional claims.
_EPS = 1e-9


@dataclass(frozen=True)
class GroupDemand:
    """What a compute group that runs on executors needs.

    Attributes:
        group: The id of the group.
        name: Its readable name.
        gpus: The GPU claim of one executor, resolved from ``gpus`` and
            ``min_vram``.
        cpus: The CPUs reserved by one executor.
        task: Whether the executors hold their GPUs (long-running).
        replicas: The declared replicas.
    """

    group: str
    name: str
    gpus: float
    cpus: float
    task: TaskType
    replicas: Replicas

    @property
    def holds(self) -> bool:
        """Whether the executors hold their GPUs until the stream is flushed."""
        return self.task == TaskType.LONG_RUNNING

    @property
    def elastic(self) -> bool:
        """Whether the planner decides the number of executors."""
        match (self.task, self.replicas):
            case (_, "auto") | (TaskType.EPHEMERAL, None):
                return True
            case _:
                return False

    @property
    def fixed(self) -> bool:
        """Whether the group holds a number of executors it declared."""
        return self.holds and not self.elastic


@dataclass(frozen=True)
class ClusterSpec:
    """The resources of a cluster, for a plan made without Ray.

    Attributes:
        node_gpus: The GPUs of every node.
        cpus: The CPUs of the cluster.
        gpu_memory: The memory in GiB of every GPU, for claims of
            ``min_vram``; claims are sized for the smallest.
    """

    node_gpus: Tuple[int, ...]
    cpus: float
    gpu_memory: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Slot:
    """GPUs of one node that the executors of a group run on."""

    node: int
    gpus: Tuple[int, ...]

    def __str__(self) -> str:
        return f"node {self.node} GPU {','.join(map(str, self.gpus))}"


@dataclass(frozen=True)
class GroupPlan:
    """The pool of a group.

    Attributes:
        replicas: The executors of the pool; for ephemeral groups, the most
            that run at once.
        phase: The phase the group runs in.
        slots: The GPUs its executors run on.
    """

    replicas: int
    phase: int
    slots: Tuple[Slot, ...]


@dataclass(frozen=True)
class Plan:
    """The placement of the pools of a pipeline.

    Attributes:
        groups: The pool of every group, by its id.
        phases: The groups of every phase.
        node_phases: The phase of every stage, by the id of its node in the
            report: the latest phase at or upstream of it.
        boundaries: The edges from one phase into a later one, by the ids
            of their nodes; the items for the later phase wait there.
        gpu_maps: The executors on every GPU, per phase.
        warnings: Findings of the planner, e.g. groups sharing a GPU.
    """

    groups: Mapping[str, GroupPlan]
    phases: Tuple[Tuple[str, ...], ...]
    node_phases: Mapping[str, int]
    boundaries: FrozenSet[Tuple[str, str]]
    gpu_maps: Tuple[Tuple[GpuSlot, ...], ...]
    warnings: Tuple[str, ...]


# -- Demands -----------------------------------------------------------------
def pooled(resources: ComputeResources, num_gpus: float) -> bool:
    """Whether the calls of a map with ``resources`` run on executors.

    Long-running protocols hold their executors; calls claiming GPUs need
    processes that keep their GPU. The ``cpu_only`` maps of a protocol share
    its group id, but claim no GPUs and run as tasks.
    """
    return resources.task == TaskType.LONG_RUNNING or num_gpus > 0


def all_maps(nodes: Sequence[IROp]) -> List[IRMap]:
    """The maps among and upstream of ``nodes``, and in their ``group_by`` templates."""
    maps: List[IRMap] = []
    seen: Set[IROp] = set()
    stack: List[IROp] = list(nodes)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        match node:
            case IRMap():
                maps.append(node)
            case IRDemux(template=IROp() as template):
                stack.append(template)
            case _:
                pass
        stack.extend(node.upstream)
    return maps


def demands_of(
    maps: Sequence[IRMap],
    gpus_of: Callable[[ComputeResources], float],
) -> Dict[str, GroupDemand]:
    """The demands of the groups of ``maps`` that run on executors.

    Args:
        gpus_of: Resolves the GPU claim of resources, e.g. of ``min_vram``.
    """
    return {
        compute.group: GroupDemand(
            group=compute.group,
            name=compute.name or compute.group,
            gpus=num_gpus,
            cpus=0 if resources.cpus is None else resources.cpus,
            task=resources.task,
            replicas=resources.replicas,
        )
        for node in maps
        if isinstance(compute := node.compute, ComputeAssignment)
        for resources in [compute.resources]
        for num_gpus in [gpus_of(resources)]
        if pooled(resources, num_gpus)
    }


# -- Planning ----------------------------------------------------------------
@dataclass
class _Phase:
    """The GPUs and CPUs of a phase while it is planned."""

    # The executors on every GPU of the cluster, by its index: their group
    # and share of the GPU.
    gpus: List[List[Tuple[str, float]]]
    cpus: float = 0.0
    groups: List[str] = field(default_factory=list)

    def free(self, gpu: int) -> float:
        return 1.0 - sum(share for _, share in self.gpus[gpu])

    def copy(self) -> _Phase:
        return _Phase([list(gpu) for gpu in self.gpus], self.cpus, list(self.groups))


@dataclass(frozen=True)
class _Cluster:
    node_gpus: Tuple[int, ...]
    cpus: float

    @property
    def gpus(self) -> List[Tuple[int, int]]:
        """The (node, GPU on the node) of every GPU, by its index."""
        return [(n, g) for n, count in enumerate(self.node_gpus) for g in range(count)]

    def empty_phase(self) -> _Phase:
        return _Phase(gpus=[[] for _ in self.gpus])


@dataclass(frozen=True)
class _Unit:
    """Groups that run in one phase: a group, or the groups of a ``group_by``
    template with the groups of their other maps.

    Attributes:
        anchors: The top-level nodes where the unit's maps run: the maps, or
            the ``group_by`` whose template holds them; in graph order.
    """

    groups: Tuple[str, ...]
    anchors: Tuple[str, ...]


def plan_placement(
    graph: PipelineReport,
    demands: Mapping[str, GroupDemand],
    node_gpus: Sequence[int],
    cluster_cpus: float,
    weights: Optional[Mapping[str, float]] = None,
) -> Plan:
    """Plan the pools of the groups of ``graph``.

    Args:
        graph: The pipeline (``report_of``), for the order of its groups.
        demands: The groups that run on executors, by id.
        node_gpus: The GPUs of every node of the cluster.
        cluster_cpus: The CPUs of the cluster.
        weights: The share of the free GPUs of every elastic group, relative
            to the others; equal by default.

    Raises:
        ValueError: If a claim cannot be satisfied in any phase.
    """
    cluster = _Cluster(tuple(int(n) for n in node_gpus), cluster_cpus)
    for demand in demands.values():
        _check_satisfiable(demand, cluster)

    units = _units(graph, demands)
    phases, unit_phase, warnings = _assign_phases(units, graph, demands, cluster)
    _place_flexible(phases, demands, weights or {}, warnings)
    return _plan(graph, units, unit_phase, phases, demands, cluster, warnings)


def _check_satisfiable(demand: GroupDemand, cluster: _Cluster) -> None:
    largest = max(cluster.node_gpus, default=0)
    if demand.gpus > 1 and not float(demand.gpus).is_integer():
        raise ValueError(
            f"{demand.name} claims {demand.gpus:g} GPUs per process; a claim "
            "above one GPU must be a whole number of GPUs.",
        )
    if demand.gpus > largest:
        raise ValueError(
            f"A process of {demand.name} needs {demand.gpus:g} GPUs, but no "
            f"node of the cluster has as many ({largest} at most).",
        )
    if demand.cpus > cluster.cpus:
        raise ValueError(
            f"A process of {demand.name} reserves {demand.cpus:g} CPUs, but "
            f"the cluster has {cluster.cpus:g}.",
        )
    match (demand.fixed, demand.replicas):
        case (True, int(replicas)) if not _fits_alone(demand, replicas, cluster):
            raise ValueError(
                f"The {replicas} processes of {demand.name} need "
                f"{replicas * demand.gpus:g} GPUs and "
                f"{replicas * demand.cpus:g} CPUs at once, since they hold "
                "them until the stream is flushed, but the cluster has "
                f"{sum(cluster.node_gpus)} GPUs and {cluster.cpus:g} CPUs. "
                "Reduce the replicas or the claim.",
            )
        case _:
            pass


def _fits_alone(demand: GroupDemand, replicas: int, cluster: _Cluster) -> bool:
    return (
        replicas * demand.cpus <= cluster.cpus + _EPS
        and _pack(cluster.empty_phase(), demand, replicas, cluster, exclusive=False)
        is not None
    )


def _units(graph: PipelineReport, demands: Mapping[str, GroupDemand]) -> List[_Unit]:
    """The units of the groups of ``demands``, in the order of the graph.

    Groups with maps in one ``group_by`` template run in one phase: its
    children are compiled at runtime, so the template has no boundaries.
    """
    parents = {node.id: node.parent for node in graph.nodes}
    order = {node.id: i for i, node in enumerate(graph.nodes)}

    def anchor(node_id: str) -> str:
        match parents[node_id]:
            case None:
                return node_id
            case parent:
                return anchor(parent)

    anchors: Dict[str, Set[str]] = {}
    for node in graph.nodes:
        match node.group:
            case str(group) if group in demands:
                anchors.setdefault(group, set()).add(anchor(node.id))
            case _:
                pass

    # Groups that share a group_by anchor join one unit.
    templates = {node.id for node in graph.nodes if node.kind == "group_by"}
    units: List[Tuple[Set[str], Set[str]]] = []
    for group, group_anchors in anchors.items():
        joined = [unit for unit in units if unit[1] & group_anchors & templates]
        merged = (
            {group}.union(*(u[0] for u in joined)),
            set(group_anchors).union(*(u[1] for u in joined)),
        )
        units = [unit for unit in units if unit not in joined] + [merged]

    return sorted(
        (
            _Unit(
                groups=tuple(g for g in anchors if g in groups),
                anchors=tuple(sorted(unit_anchors, key=order.__getitem__)),
            )
            for groups, unit_anchors in units
        ),
        key=lambda unit: order[unit.anchors[0]],
    )


def _upstream(graph: PipelineReport) -> Mapping[str, Set[str]]:
    """The top-level nodes upstream of every top-level node, transitively."""
    inputs: Dict[str, List[str]] = {}
    for edge in graph.edges:
        inputs.setdefault(edge.target, []).append(edge.source)
    upstream: Dict[str, Set[str]] = {}
    for node in graph.nodes:
        if node.parent is None:
            direct = inputs.get(node.id, [])
            upstream[node.id] = set(direct).union(*(upstream[u] for u in direct))
    return upstream


def _assign_phases(
    units: List[_Unit],
    graph: PipelineReport,
    demands: Mapping[str, GroupDemand],
    cluster: _Cluster,
) -> Tuple[List[_Phase], List[int], List[str]]:
    """The phase of every unit, with its groups packed (see ``_place_unit``).

    Raises:
        ValueError: If a unit runs both before and after a unit of a later
            phase, or the long-running groups of a unit do not fit together.
    """
    upstream = _upstream(graph)
    anchor_phase: Dict[str, int] = {}
    phases: List[_Phase] = []
    unit_phase: List[int] = []
    warnings: List[str] = []

    for unit in units:
        phase = max(
            (anchor_phase.get(u, 0) for u in upstream[unit.anchors[0]]),
            default=0,
        )
        while True:
            while len(phases) <= phase:
                phases.append(cluster.empty_phase())
            if _place_unit(phases[phase], unit, demands, cluster, warnings):
                break
            if not phases[phase].groups:
                raise ValueError(
                    "The long-running protocols "
                    f"{', '.join(demands[g].name for g in unit.groups)} run in "
                    "one group_by and hold their GPUs at once, but do not fit "
                    "the cluster together. Reduce their replicas or claims.",
                )
            phase += 1
        unit_phase.append(phase)
        phases[phase].groups.extend(unit.groups)
        anchor_phase.update({anchor: phase for anchor in unit.anchors})

    for unit, phase in zip(units, unit_phase):
        late = [
            node
            for anchor in unit.anchors[1:]
            for node in sorted(upstream[anchor])
            if anchor_phase.get(node, 0) > phase
        ]
        if late:
            raise ValueError(
                f"{', '.join(demands[g].name for g in unit.groups)} runs both "
                f"before and after {late[0]}, which runs in a later phase: it "
                "would hold its GPUs while it waits for them. The protocols "
                "do not fit the cluster at once.",
            )
    return phases, unit_phase, warnings


def _place_unit(
    phase: _Phase,
    unit: _Unit,
    demands: Mapping[str, GroupDemand],
    cluster: _Cluster,
    warnings: List[str],
) -> bool:
    """Pack the groups of ``unit`` into ``phase``, if they fit.

    Fixed long-running groups take their executors. Flexible groups take one
    executor, a reservation that keeps the groups packed after them from
    taking all GPUs; ``_place_flexible`` sizes them once all are packed.
    Ephemeral groups do not hold their GPUs, so they pack beside other
    ephemeral groups as if those were not there.
    """
    trial = phase.copy()
    shared: List[str] = []
    for demand in (demands[g] for g in unit.groups):
        replicas = (
            int(demand.replicas)
            if demand.fixed and isinstance(demand.replicas, int)
            else 1
        )
        if demand.holds and trial.cpus + replicas * demand.cpus > cluster.cpus + _EPS:
            return False
        view = _holding_view(trial, demands) if not demand.holds else trial
        placement = _pack(view, demand, replicas, cluster, exclusive=True)
        if placement is None:
            placement = _pack(view, demand, replicas, cluster, exclusive=False)
            if demand.fixed:
                shared.append(demand.name)
        if placement is None:
            return False
        for gpus in placement:
            for gpu in gpus:
                trial.gpus[gpu].append((demand.group, min(demand.gpus, 1.0)))
        if demand.holds:
            trial.cpus += replicas * demand.cpus
    phase.gpus, phase.cpus = trial.gpus, trial.cpus
    warnings.extend(
        f"{name} shares GPUs with other long-running protocols; their "
        "processes time-slice the GPUs."
        for name in shared
    )
    return True


def _holding_view(phase: _Phase, demands: Mapping[str, GroupDemand]) -> _Phase:
    """``phase`` with only the executors of groups that hold their GPUs."""
    return _Phase(
        [[(g, s) for g, s in gpu if demands[g].holds] for gpu in phase.gpus],
        phase.cpus,
        list(phase.groups),
    )


def _pack(
    phase: _Phase,
    demand: GroupDemand,
    replicas: int,
    cluster: _Cluster,
    exclusive: bool,
) -> Optional[List[Tuple[int, ...]]]:
    """The GPUs of every executor of ``demand`` in ``phase`` (first fit), or None.

    ``exclusive`` places executors only on GPUs without other groups.
    """
    trial = phase.copy()
    nodes = cluster.gpus
    placement: List[Tuple[int, ...]] = []

    def usable(gpu: int, need: float) -> bool:
        others = any(group != demand.group for group, _ in trial.gpus[gpu])
        return trial.free(gpu) + _EPS >= need and not (exclusive and others)

    def own(gpu: int) -> bool:
        return any(group == demand.group for group, _ in trial.gpus[gpu])

    for _ in range(replicas):
        match demand.gpus:
            case 0:
                chosen: Tuple[int, ...] = ()
            case claim if claim < 1:
                # The GPUs of the group first, then the others in order.
                candidates = [g for g in range(len(nodes)) if usable(g, claim)]
                candidates.sort(key=lambda g: (not own(g), g))
                if not candidates:
                    return None
                chosen = (candidates[0],)
            case claim:
                fitting = [
                    free[: int(claim)]
                    for node in range(len(cluster.node_gpus))
                    for free in [
                        [
                            g
                            for g, (n, _) in enumerate(nodes)
                            if n == node and usable(g, 1.0)
                        ]
                    ]
                    if len(free) >= claim
                ]
                if not fitting:
                    return None
                chosen = tuple(fitting[0])
        placement.append(chosen)
        for gpu in chosen:
            trial.gpus[gpu].append((demand.group, min(demand.gpus, 1.0)))
    return placement


def _place_flexible(
    phases: List[_Phase],
    demands: Mapping[str, GroupDemand],
    weights: Mapping[str, float],
    warnings: List[str],
) -> None:
    """Give the flexible groups of every phase their GPUs, in place of their
    reservations: the GPUs the fixed groups leave (see ``_split``)."""
    for phase in phases:
        flexible = [
            demands[g]
            for g in phase.groups
            if not demands[g].fixed and demands[g].gpus > 0
        ]
        phase.gpus = [
            [(g, s) for g, s in gpu if demands[g].fixed] for gpu in phase.gpus
        ]
        _split(phase, flexible, weights)
        _warn_shared(phase, demands, warnings)


def _free_gpus(phase: _Phase) -> List[int]:
    return [g for g, occupants in enumerate(phase.gpus) if not occupants]


def _split(
    phase: _Phase, flexible: List[GroupDemand], weights: Mapping[str, float]
) -> None:
    """Give the flexible groups of ``phase`` their GPUs.

    Elastic groups split the free GPUs by weight, each on consecutive GPUs;
    ephemeral groups with declared replicas may use all free GPUs. Without
    a free GPU, a group runs one executor beside the fixed groups.
    """
    free = _free_gpus(phase)
    elastic = [d for d in flexible if d.elastic]
    total = sum(weights.get(d.group, 1.0) for d in elastic)
    start = 0.0
    placed: List[Tuple[str, List[int], float]] = []
    for demand in flexible:
        match (demand.elastic, free):
            case (_, []):
                room = [
                    g
                    for g in range(len(phase.gpus))
                    if phase.free(g) + _EPS >= demand.gpus
                ]
                placed.append((demand.group, room[:1], demand.gpus))
            case (True, _):
                share = len(free) * weights.get(demand.group, 1.0) / total
                first = math.floor(start + _EPS)
                last = max(first, math.ceil(start + share - _EPS) - 1)
                placed.append((demand.group, free[first : last + 1], share))
                start += share
            case (False, _):
                replicas = int(demand.replicas or 1)
                placed.append(
                    (demand.group, free, min(len(free), replicas * demand.gpus))
                )
    for group, gpus, share in placed:
        for gpu in gpus:
            phase.gpus[gpu].append((group, share / len(gpus)))


def _warn_shared(
    phase: _Phase,
    demands: Mapping[str, GroupDemand],
    warnings: List[str],
) -> None:
    for gpu, occupants in enumerate(phase.gpus):
        groups = list(dict.fromkeys(demands[group].name for group, _ in occupants))
        message = f"GPU {gpu} is shared by {', '.join(groups)}."
        if len(groups) > 1 and message not in warnings:
            warnings.append(message)


def _plan(
    graph: PipelineReport,
    units: List[_Unit],
    unit_phase: List[int],
    phases: List[_Phase],
    demands: Mapping[str, GroupDemand],
    cluster: _Cluster,
    warnings: List[str],
) -> Plan:
    """The plan of the planned phases."""
    nodes = cluster.gpus
    phase_of = {g: phase for unit, phase in zip(units, unit_phase) for g in unit.groups}
    node_phases = _node_phases(graph, phase_of)
    names = {group: demand.name for group, demand in demands.items()}
    return Plan(
        groups={
            group: GroupPlan(
                replicas=_replicas(demands[group], phases[phase]),
                phase=phase,
                slots=_slots(group, phases[phase], nodes, demands[group].gpus),
            )
            for group, phase in phase_of.items()
        },
        phases=tuple(tuple(phase.groups) for phase in phases),
        node_phases=node_phases,
        boundaries=frozenset(
            (edge.source, edge.target)
            for edge in graph.edges
            if node_phases[edge.source] < node_phases[edge.target]
        ),
        gpu_maps=tuple(
            tuple(
                GpuSlot(
                    nodes[gpu][0], nodes[gpu][1], _occupants(phase.gpus[gpu], names)
                )
                for gpu in range(len(nodes))
            )
            for phase in phases
        ),
        warnings=tuple(warnings),
    )


def _replicas(demand: GroupDemand, phase: _Phase) -> int:
    """The executors of ``demand``: declared, or what its share of the GPUs holds."""
    match (demand.elastic, demand.replicas):
        case (False, int(replicas)):
            return replicas
        case (False, _):
            return 1
        case (True, _) if demand.gpus == 0:
            return 1
        case (True, _):
            share = sum(s for gpu in phase.gpus for g, s in gpu if g == demand.group)
            return max(1, math.floor(share / demand.gpus + _EPS))


def _slots(
    group: str,
    phase: _Phase,
    nodes: List[Tuple[int, int]],
    claim: float,
) -> Tuple[Slot, ...]:
    """The GPUs of ``group`` in ``phase``: one slot per GPU, or per group of
    ``claim`` GPUs on a node for claims above one GPU."""
    by_node: Dict[int, List[int]] = {}
    for gpu, occupants in enumerate(phase.gpus):
        if any(g == group for g, _ in occupants):
            by_node.setdefault(nodes[gpu][0], []).append(nodes[gpu][1])
    size = max(1, int(claim))
    return tuple(
        Slot(node, tuple(on_node[i : i + size]))
        for node, on_node in by_node.items()
        for i in range(0, len(on_node) - size + 1, size)
    )


def _occupants(
    occupants: List[Tuple[str, float]],
    names: Mapping[str, str],
) -> Tuple[GpuOccupant, ...]:
    shares: Dict[str, float] = {}
    for group, share in occupants:
        shares[group] = shares.get(group, 0.0) + share
    return tuple(
        GpuOccupant(names[group], round(min(share, 1.0), 6))
        for group, share in shares.items()
    )


def _node_phases(graph: PipelineReport, phase_of: Mapping[str, int]) -> Dict[str, int]:
    """The phase of every node: the latest at or upstream of it.

    The nodes of a ``group_by`` template run in the phase of the
    ``group_by``, which is the latest of the groups in it.
    """
    inputs: Dict[str, List[str]] = {}
    for edge in graph.edges:
        inputs.setdefault(edge.target, []).append(edge.source)
    parents = {node.id: node.parent for node in graph.nodes}

    def top(node_id: str) -> str:
        match parents[node_id]:
            case None:
                return node_id
            case parent:
                return top(parent)

    def own(group: Optional[str]) -> List[int]:
        """The phase of ``group``, if it is planned."""
        match group:
            case str() if group in phase_of:
                return [phase_of[group]]
            case _:
                return []

    in_template: Dict[str, int] = {}
    for node in graph.nodes:
        if node.parent is not None:
            anchor = top(node.id)
            in_template[anchor] = max([in_template.get(anchor, 0), *own(node.group)])

    phases: Dict[str, int] = {}
    for node in graph.nodes:
        match node.parent:
            case None:
                phases[node.id] = max(
                    [
                        0,
                        *own(node.group),
                        *([in_template[node.id]] if node.id in in_template else []),
                        *(phases[u] for u in inputs.get(node.id, [])),
                    ],
                )
            case _:
                phases[node.id] = phases[top(node.id)]
    return phases


# -- The report layer --------------------------------------------------------
def placement_layer(
    graph: PipelineReport,
    plan: Plan,
    demands: Mapping[str, GroupDemand],
) -> Layer:
    """The plan as a layer of the report ``graph``."""
    nodes: Dict[str, Dict[str, Value]] = {
        node.id: {"phase": plan.node_phases[node.id]} for node in graph.nodes
    }
    for node in graph.nodes:
        match (node.kind, node.group):
            case ("map", str(group)) if group in plan.groups:
                nodes[node.id]["executors"] = plan.groups[group].replicas
                nodes[node.id]["gpus_per_executor"] = demands[group].gpus
            case ("map", _):
                nodes[node.id]["runs_as"] = "Ray tasks"
            case _:
                pass

    for source, target in plan.boundaries:
        nodes[target]["waits_for"] = f"the FLUSH of phase {plan.node_phases[source]}"
    waits = tuple(
        f"{target} starts once the stages of phase {plan.node_phases[source]} "
        "have flushed, at the end of the stream; its items wait on disk until "
        "then."
        for source, target in sorted(plan.boundaries)
    )
    return Layer(
        name="placement",
        title="Placement",
        nodes=nodes,
        edges={edge: {"boundary": "spills"} for edge in plan.boundaries},
        groups={
            group: {
                "task": demands[group].task.value,
                "phase": pool.phase,
                "executors": pool.replicas,
                "declared_replicas": str(demands[group].replicas),
                "gpus_per_executor": demands[group].gpus,
                "cpus_per_executor": demands[group].cpus,
                "gpus": tuple(str(slot) for slot in pool.slots),
            }
            for group, pool in plan.groups.items()
        },
        metric="phase",
        panels=(
            TablePanel(
                "Pools",
                ("protocol", "task", "phase", "executors", "GPUs each", "on"),
                tuple(
                    (
                        demands[group].name,
                        demands[group].task.value,
                        pool.phase,
                        pool.replicas,
                        demands[group].gpus,
                        ", ".join(str(slot) for slot in pool.slots),
                    )
                    for group, pool in plan.groups.items()
                ),
            ),
            *(
                GpuMapPanel(
                    f"GPUs in phase {index}: "
                    + (", ".join(demands[g].name for g in groups) or "no pools"),
                    gpu_map,
                )
                for index, (groups, gpu_map) in enumerate(
                    zip(plan.phases, plan.gpu_maps)
                )
            ),
        ),
        warnings=(*plan.warnings, *waits),
    )


def explain_placement(ir_sinks: Sequence[IROp], cluster: ClusterSpec) -> PipelineReport:
    """The report of the pipeline ending in ``ir_sinks``, with its placement
    on ``cluster``, planned without Ray.

    Raises:
        ValueError: If a claim cannot be satisfied on ``cluster``.
    """
    graph = report_of(ir_sinks)
    smallest = min(cluster.gpu_memory, default=None)
    demands = demands_of(
        all_maps(ir_sinks),
        lambda resources: gpu_claim(resources, smallest),
    )
    plan = plan_placement(graph, demands, cluster.node_gpus, cluster.cpus)
    return graph.with_layer(placement_layer(graph, plan, demands))
