"""The compiler of the Ray streaming backend, and the handle of its pipelines.

``RayBackend.compile`` plans a pipeline: the stages, the claims of its maps
and the placement of its pools on the cluster. ``RayCompiledPipeline``
starts the actors of the plan (``actors.py``) when it is started or first
used.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import asyncio
import dataclasses
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import ray  # type: ignore
from ray.util.placement_group import (  # type: ignore
    get_placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import (  # type: ignore
    NodeAffinitySchedulingStrategy,
)

from scipion_bridge.core.environment.compute import (
    CPUS_ENV_VAR,
    ComputeAssignment,
    ComputeResources,
    gpu_claim,
    configure_numpy_hugepages,
    gpu_memory_env,
    numpy_hugepage_env,
)
from scipion_bridge.core.streaming.backend import (
    CompiledPipeline,
    StageStats,
    StreamingBackendProvider,
)
from scipion_bridge.core.streaming.ir import (
    DEFAULT_GROUP_BY_WORKERS,
    GroupByWorkers,
    IROp,
    IRSource,
    IRMap,
    IRAccumulate,
    IRSink,
    IRDemux,
    WorkersFrom,
    stage_labels,
    topological_order,
)
from scipion_bridge.core.streaming.node import FLUSH
from scipion_bridge.core.streaming.provenance import StageSamples
from scipion_bridge.core.streaming.report.model import (
    Layer,
    PipelineReport,
    report_of,
)
from scipion_bridge.core.streaming.spill import SpillStoreFactory, pickle_spill_store
from scipion_bridge.core.struct import references

from .actors import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_EPHEMERAL_IDLE_S,
    DEFAULT_SPILL_THRESHOLD,
    _DEFAULT_MAX_IN_FLIGHT,
    _EXECUTOR_CALLS_IN_FLIGHT,
    RayAccumulatorActor,
    RayComputeGroup,
    RayDemuxActor,
    RayMergeActor,
    RaySinkActor,
    RaySourceActor,
    _Apply,
    _Push,
    _Route,
    _SpillConfig,
    _StageConfig,
    _reservation_of,
    _transport,
)
from .function_table import FunctionTable
from .mailbox import Ref
from .placement import (
    GroupDemand,
    Plan,
    all_maps,
    blockers_of,
    bundles_of,
    demands_of,
    placement_layer,
    plan_placement,
    pooled,
)
from .profiling import ProfileConfig, Recorder, make_recorder, start_profile

logger = logging.getLogger(__name__)


@cache
def _cluster_cpus() -> int:
    return int(ray.cluster_resources().get("CPU", 1))


def _ray_options(resources: ComputeResources, num_gpus: float) -> Dict[str, Any]:
    """Ray options of a task or actor holding ``resources``.

    ``num_gpus`` is the GPU claim resolved from ``gpus`` and ``min_vram``. With
    a fraction of a GPU, the environment limits the memory that frameworks
    preallocate (see ``gpu_memory_env``).

    Without reserved CPUs, the process may use all cores of the machine and
    the OS schedules them. Ray would set ``OMP_NUM_THREADS`` to the number of
    reserved cores, making NumPy, BLAS and PyTorch single-threaded, so it is
    set to all cores instead. With reserved cores, user code learns their
    number from ``CPUS_ENV_VAR`` (e.g. to size the pools of map_element).

    NumPy's transparent hugepage hint is turned off, as in every process of
    the backend (see ``numpy_hugepage_env``).
    """
    if resources.cpus is None:
        num_cpus: float = 0
        cpu_env = {"OMP_NUM_THREADS": str(_cluster_cpus())}
    else:
        num_cpus = resources.cpus
        cpu_env = {CPUS_ENV_VAR: str(resources.cpus)}

    return {
        "num_gpus": num_gpus,
        "num_cpus": num_cpus,
        "runtime_env": {
            "env_vars": {
                **numpy_hugepage_env(),
                **cpu_env,
                **gpu_memory_env(num_gpus),
            },
        },
    }


def read_gpu_memory() -> List[float]:
    """Memory in GiB of every GPU of this node, or none without nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    # nvidia-smi reports MiB.
    return [float(mib) / 1024 for mib in output.split()]


@ray.remote(num_cpus=0, num_gpus=0)
def _node_gpu_memory() -> List[float]:
    """Probe of the GPU memory of the node the task runs on.

    A task without GPUs does not know which GPUs Ray manages on its node, so
    it reports all of them; a GPU hidden from Ray only makes claims larger.
    """
    return read_gpu_memory()


@cache
def _probe_cluster_gpu_memory() -> Tuple[float, ...]:
    """Memory in GiB of every GPU of the cluster, probed once per process."""
    nodes = [
        node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    probes = [
        _node_gpu_memory.options(  # type: ignore
            scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False),
            runtime_env={"env_vars": numpy_hugepage_env()},
        ).remote()
        for node in nodes
    ]
    return tuple(memory for node in ray.get(probes) for memory in node)


@cache
def _resolve_gpus(resources: ComputeResources, gpu_memory: Tuple[float, ...]) -> float:
    """The GPU claim of ``resources``, logged once per process.

    Ray treats the GPUs of a node as interchangeable and a call requests a
    single number of GPUs, so memory claims are sized for the smallest GPU.
    """
    num_gpus = gpu_claim(resources, min(gpu_memory, default=None))
    if resources.min_vram is not None and gpu_memory:
        logger.info(
            "min_vram=%g GiB -> %g GPU, sized for the smallest GPU (%g GiB)%s.",
            resources.min_vram,
            num_gpus,
            min(gpu_memory),
            _unused_share(resources.min_vram, num_gpus, gpu_memory),
        )
    return num_gpus


def _unused_share(
    min_vram: float,
    num_gpus: float,
    gpu_memory: Tuple[float, ...],
) -> str:
    """Note on the memory of larger GPUs that claims sized for the smallest leave."""
    calls_per_gpu = math.floor(1 / min(num_gpus, 1.0))
    return "".join(
        f"; {1 - calls_per_gpu * min_vram / size:.0%} of the {size:g} GiB GPUs "
        "stays unused"
        for size in sorted(set(gpu_memory))
        if size > min(gpu_memory)
    )


# Seconds ``close`` waits for the stages to remove their spilled items.
_CLOSE_SPILL_TIMEOUT_S = 10.0


# Placement of the pools of executors: "managed" reserves the GPUs the
# planner gives every long-running pool in a placement group, "ray" leaves
# placement to Ray (for clusters shared with other jobs).
PlacementMode = Literal["managed", "ray"]


@dataclass(frozen=True)
class _PoolPlan:
    """The pools of executors a pipeline owns, as planned when it is compiled."""

    placement: _Placement
    # The pools of earlier phases that every pool waits for.
    waits: Mapping[str, FrozenSet[str]]
    # The maps of every group, each running one call at a time on an executor.
    members: Mapping[str, int]
    # The Ray options of the executors of every group.
    options: Mapping[str, Dict[str, Any]]
    # The Ray id of every node of the cluster, by its index in the plan.
    node_ids: Tuple[str, ...]


@dataclass(frozen=True)
class _PipelinePlan:
    """A compiled pipeline: its stages and pools, before any actor starts."""

    # Identifies the actors and placement groups of the pipeline in Ray.
    pipeline_id: str
    # Prefix of the stage labels, for nested pipelines.
    name: str
    abort_targets: Tuple[Any, ...]
    # All nodes in topological order (upstream first), with their labels.
    nodes: Tuple[IROp, ...]
    labels: Mapping[IROp, str]
    # All maps, those in the templates of group_by included.
    maps: Tuple[IRMap, ...]
    # The Ray options of the calls of the maps, by compute assignment.
    claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]]
    # The pools the pipeline owns; None for the child of a group_by, which
    # runs its maps on the pools of its parent.
    pools: Optional[_PoolPlan]
    # The stages behind a phase boundary, which have an actor of their own.
    boundaries: FrozenSet[IROp]
    spill: Optional[_SpillConfig]
    boundary_spill: Optional[_SpillConfig]
    # Temporary directory of the spilled items, removed on close.
    spill_root: Optional[Path]

    @property
    def placement(self) -> Optional[_Placement]:
        return None if self.pools is None else self.pools.placement

    @property
    def sources(self) -> Dict[str, IRSource]:
        """The sources of the pipeline, by name."""
        return {node.name: node for node in self.nodes if isinstance(node, IRSource)}

    @property
    def stage_labels(self) -> List[str]:
        """The labels of the stages, in topological order."""
        return [self.labels[node] for node in self.nodes]

    def host_of(self, node: IROp) -> IROp:
        """The stage whose actor runs ``node``."""
        return _host_of(node, self.boundaries)

    @property
    def own_actor_maps(self) -> FrozenSet[str]:
        """The labels of the maps that run on an actor of their own (merging
        maps, maps behind a phase boundary), whose stats include the
        buffering of the actor."""
        return frozenset(
            self.labels[node]
            for node in self.nodes
            if isinstance(node, IRMap) and self.host_of(node) is node
        )


@dataclass
class _Runtime:
    """The actors of a started pipeline."""

    # Source actors keyed by input name.
    sources: Dict[str, Any]
    # All stages of the pipeline keyed by label, in topological order, each
    # with the node and actor hosting it: its own actor, or for a map the
    # actor submitting its calls.
    stages: Dict[str, Tuple[IROp, Any]]
    # The group_by routers among the stages, whose children report stats.
    demuxes: List[Any]
    # Coordinators of the pools the pipeline owns: it releases them after
    # every flush, and their executors terminate with them on close.
    groups: List[Any]
    profile: Optional[ProfileConfig]
    # Whether the pipeline completes the trace on close; a nested pipeline
    # writes into the trace of its parent.
    owns_profile: bool
    # The function table of the maps, kept in the object store while the
    # pipeline runs.
    functions: Optional[ray.ObjectRef]
    recorder: Recorder

    @property
    def hosts(self) -> Dict[IROp, Any]:
        """The actor of every stage that has one."""
        return {node: actor for node, actor in self.stages.values()}


class RayCompiledPipeline(CompiledPipeline):
    """
    Executable compiled streaming pipeline running on Ray.

    Compiling plans the pipeline without starting any actor, so its report
    and stats are available at no cost. The actors start with ``start``, or
    with the first ``send``, ``flush``, ``source_handle`` or ``profile``.
    """

    def __init__(self, backend: RayBackend, plan: _PipelinePlan):
        """
        Args:
            backend: The backend that compiled the pipeline, which starts it.
            plan: The stages and pools of the pipeline.
        """
        self._backend = backend
        self._plan = plan
        self._runtime: Optional[_Runtime] = None
        self._closed = False

    def start(self) -> None:
        """Start the actors of the pipeline and wait until they are connected.

        Long-running pools of the first phase start preparing their
        executors. Does nothing if the pipeline has started.

        Raises:
            RuntimeError: If the pipeline was closed.
        """
        if self._runtime is None:
            self._check_open()
            runtime, connected = self._backend._launch(self._plan)
            self._runtime = runtime
            ray.get(connected)

    async def start_async(self) -> None:
        """Start like ``start``, without blocking the event loop.

        Starting the actors of a pipeline takes seconds; an async actor (the
        router of a ``group_by``) awaits it while it keeps working, and
        starts several pipelines at once.
        """
        if self._runtime is None:
            self._check_open()
            runtime, connected = await asyncio.to_thread(
                self._backend._launch,
                self._plan,
            )
            self._runtime = runtime
            await asyncio.gather(*connected)

    def layers(self, report: PipelineReport) -> Tuple[Layer, ...]:
        """The placement of the pools, if the pipeline owns them."""
        placement = self._plan.placement
        return () if placement is None else (placement.layer(),)

    @property
    def profile(self) -> Optional[ProfileConfig]:
        """Profiling of the pipeline, for processes that record into its trace.

        Starts the pipeline, which starts its profile.
        """
        return self._started().profile

    def source_handle(self, source_name: str) -> Any:
        """Return the actor handle of a named input source.

        Lets an async actor push into the pipeline without blocking its event
        loop: ``await pipeline.source_handle(name).push.remote(item)``.
        """
        if source_name not in self._plan.sources:
            raise KeyError(
                f"Input source '{source_name}' is not registered in this pipeline. Available: {list(self._plan.sources)}",
            )
        return self._started().sources[source_name]

    def send(self, source_name: str, value: Any) -> None:
        """Push an item into a named input source.

        Returns once the source has buffered the item. While the buffer of
        the source is full, because the stages downstream of it are, it waits
        for room.
        """
        handle = self.source_handle(source_name)
        with self._started().recorder.span("send", "driver", source=source_name):
            ray.get(handle.push.remote(_transport(value)))

    def flush(self) -> None:
        """Flush all sources in parallel and wait for pipeline completion.

        Then release the pools of executors: FLUSH has passed every stage,
        the children of ``group_by`` included, so no calls are left.
        """
        runtime = self._started()
        with runtime.recorder.span("flush", "driver", level=logging.INFO):
            ray.get([src.push.remote(FLUSH) for src in runtime.sources.values()])
        ray.get([group.release_all.remote() for group in runtime.groups])
        runtime.recorder.flush_sync()

    def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of every process.

        A stage sends its events at every FLUSH; this sends the events since,
        e.g. of a failed pipeline. A pipeline that has not started recorded
        nothing.
        """
        if self._runtime is not None:
            runtime = self._runtime
            ray.get(
                [
                    actor.flush_profile.remote()
                    for actor in [*runtime.hosts.values(), *runtime.groups]
                ],
            )
            runtime.recorder.flush_sync()

    def stats(self) -> Dict[str, StageStats]:
        """Return execution metrics per stage, in topological order.

        The stages of ``group_by`` children follow, labelled with their key.
        Before the pipeline has started, every stage has empty stats.
        """
        if self._runtime is None:
            return {label: StageStats() for label in self._plan.stage_labels}
        else:
            return self._runtime_stats(self._runtime)

    def provenance(self) -> Dict[str, StageSamples]:
        """The provenance samples of every stage that started, by label.

        The stages of ``group_by`` children follow, labelled with their key.
        """
        if self._runtime is None:
            return {}
        else:
            runtime = self._runtime
            hosts = ray.get(
                [actor.provenance.remote() for actor in runtime.hosts.values()]
            )
            children = ray.get(
                [demux.children_provenance.remote() for demux in runtime.demuxes],
            )
            return {
                label: samples
                for stage in [*hosts, *children]
                for label, samples in stage.items()
            }

    def _runtime_stats(self, runtime: _Runtime) -> Dict[str, StageStats]:
        hosts = runtime.hosts
        stage_stats = dict(
            zip(hosts, ray.get([actor.stats.remote() for actor in hosts.values()])),
        )
        map_stats = {
            label: stage
            for maps in ray.get([actor.map_stats.remote() for actor in hosts.values()])
            for label, stage in maps.items()
        }
        children = ray.get(
            [demux.children_stats.remote() for demux in runtime.demuxes],
        )
        own_actor_maps = self._plan.own_actor_maps
        return {
            **{
                label: _stats_of(
                    map_stats.get(label),
                    stage_stats[host],
                    own_actor=label in own_actor_maps,
                )
                for label, (host, _) in runtime.stages.items()
            },
            **{label: stage for child in children for label, stage in child.items()},
        }

    def close(self) -> None:
        """Terminate all actors allocated for this pipeline.

        The children of ``group_by`` routers terminate with their router, and
        the executors of compute groups with their coordinator. A profiled
        pipeline completes its trace first, and every stage removes its
        spilled items from the disk of its node. A pipeline that has not
        started holds nothing to terminate.
        """
        self._closed = True
        if self._runtime is not None:
            self._terminate(self._runtime)

    def _terminate(self, runtime: _Runtime) -> None:
        if runtime.owns_profile:
            self._close_profile(runtime)

        try:
            ray.get(self.close_spill(), timeout=_CLOSE_SPILL_TIMEOUT_S)
        except Exception:
            # A failed stage may be gone; its node removes its temporary files.
            pass

        for actor in [*runtime.hosts.values(), *runtime.groups]:
            try:
                ray.kill(actor)
            except Exception:
                pass
        # Ray keeps a placement group when the actor that created it dies.
        if self._plan.pools is not None:
            _remove_placement_groups(f"{self._plan.pipeline_id}/")
        if self._plan.spill_root is not None:
            # ``close`` may run more than once.
            shutil.rmtree(self._plan.spill_root, ignore_errors=True)

        # Without a reference, Ray frees the function table.
        runtime.functions = None

    def close_spill(self) -> List[ray.ObjectRef]:
        """Make every stage remove its spilled items; the calls to await."""
        if self._runtime is None:
            return []
        else:
            actors = set(self._runtime.hosts.values())
            return [actor.close_spill.remote() for actor in actors]

    def _close_profile(self, runtime: _Runtime) -> None:
        """Write the remaining events and complete the trace file."""
        assert runtime.profile is not None, "The pipeline owns no profile."
        self.flush_profile()
        ray.get(runtime.profile.collector.close.remote())
        ray.kill(runtime.profile.collector)
        # ``close`` may run more than once, after the actors were killed.
        runtime.owns_profile = False

    def _started(self) -> _Runtime:
        """The actors of the pipeline, started if necessary."""
        self.start()
        assert self._runtime is not None
        return self._runtime

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "The pipeline is closed. Compile a new one to process more items.",
            )


def _remove_placement_groups(prefix: str) -> None:
    """Remove the placement groups whose names start with ``prefix``."""
    for entry in placement_group_table().values():
        if entry["name"].startswith(prefix) and entry["state"] != "REMOVED":
            remove_placement_group(get_placement_group(entry["name"]))


class RayBackend(StreamingBackendProvider):
    """
    Streaming backend that compiles IR DAGs into persistent Ray actors with direct P2P messaging.
    """

    def __init__(
        self,
        init_ray: bool = True,
        queue_size: int = 2,
        parameters: Optional[Mapping[str, Any]] = None,
        gpu_memory: Optional[Sequence[float]] = None,
        max_in_flight: Optional[int] = None,
        buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE,
        spill_threshold: Optional[int] = DEFAULT_SPILL_THRESHOLD,
        spill_store: Optional[SpillStoreFactory] = None,
        profile: Union[None, bool, str, Path, ProfileConfig] = None,
        profile_log_level: int = logging.INFO,
        group_by_workers: Optional[int] = DEFAULT_GROUP_BY_WORKERS,
        ephemeral_idle_s: float = DEFAULT_EPHEMERAL_IDLE_S,
        pools: Optional[Mapping[str, "_Pool"]] = None,
        placement: PlacementMode = "managed",
    ):
        """
        Args:
            init_ray: Initialize a local Ray instance if none is running.
            parameters: Values of the protocol parameters, served to
                ``Field.value`` inside the worker stages.
            gpu_memory: Memory in GiB of the cluster's GPUs, for claims of
                ``min_vram``. Probed on the cluster when a map needs it if not
                given; ``group_by`` children receive it from their parent.
            queue_size: Number of items each stage reads ahead of its compute
                task, and buffers in its outbox. The default of 2
                double-buffers every stage.
            max_in_flight: Number of items on the routes out of a stage at
                once, each through all the maps up to the next stage.
                Defaults to ``_DEFAULT_MAX_IN_FLIGHT``.
            buffer_size: Most items waiting for a stage (default
                ``DEFAULT_BUFFER_SIZE``); beyond it, upstream stages wait, as
                far as the driver's ``send``. The bound counts items, whatever
                their size. ``None`` lets items wait without limit, so a slow
                stage never stalls the stages upstream of it; set a spill
                threshold with it, or the backlog fills the object store.
            spill_threshold: Number of waiting items a stage keeps in the
                object store (default ``DEFAULT_SPILL_THRESHOLD``); further
                items are written to the spill store. Spilling only happens
                when it is below ``buffer_size``. ``None`` never spills.
            spill_store: Creates the spill store of a stage from its label.
                Defaults to pickling the items into a temporary directory of
                the compiled pipeline.
            profile: Profile the compiled pipelines: every worker logs what it
                does, and the events are written to a trace file at this path
                while the pipeline runs (see ``profiling``). ``True`` writes
                to ``ray_profile_<pipeline id>.json`` in the working
                directory. A ``ProfileConfig`` writes into an existing trace
                (``group_by`` children receive the one of their parent).
            profile_log_level: Level of the profiling log; the events of
                every item are logged at DEBUG, the others at INFO.
            group_by_workers: Child pipelines a ``group_by`` shares between
                its keys, unless the ``group_by`` sets ``workers`` itself.
                Every child starts the stages (processes) of the pipeline;
                ``None`` starts a child per key.
            ephemeral_idle_s: Seconds an executor of an ephemeral protocol
                stays alive without calls before it releases its GPUs
                (default ``DEFAULT_EPHEMERAL_IDLE_S``).
            pools: The pools of executors of the parent pipeline, by compute
                group; ``group_by`` children run their maps on them instead of
                pools of their own.
            placement: ``"managed"`` (the default) reserves the GPUs that the
                placement planner gives every long-running pool, in a Ray
                placement group, while the pool holds them. ``"ray"`` lets
                Ray place the executors, for clusters shared with other jobs.
        """
        if queue_size <= 0:
            raise ValueError(f"Queue size must be positive, got {queue_size}.")
        if placement not in ("managed", "ray"):
            raise ValueError(
                f"placement must be 'managed' or 'ray', got {placement!r}.",
            )
        for option, value in {
            "max_in_flight": max_in_flight,
            "buffer_size": buffer_size,
            "spill_threshold": spill_threshold,
            "group_by_workers": group_by_workers,
            "ephemeral_idle_s": ephemeral_idle_s,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{option} must be positive, got {value}.")
        if spill_store is not None and spill_threshold is None:
            raise ValueError("A spill_store requires a spill_threshold.")

        self.queue_size = queue_size
        self.max_in_flight = (
            _DEFAULT_MAX_IN_FLIGHT if max_in_flight is None else max_in_flight
        )
        self.buffer_size = buffer_size
        self.spill_threshold = spill_threshold
        self.spill_store = spill_store
        self.parameters = dict(parameters or {})
        self._gpu_memory = None if gpu_memory is None else tuple(gpu_memory)
        self.profile = Path(profile) if isinstance(profile, str) else profile
        self.profile_log_level = profile_log_level
        self.group_by_workers = group_by_workers
        self.ephemeral_idle_s = ephemeral_idle_s
        self.pools = pools
        self.placement: PlacementMode = placement
        # The driver imported NumPy before; the workers get the environment.
        configure_numpy_hugepages()

        if init_ray and not ray.is_initialized():
            extra_paths = [os.getcwd(), os.path.abspath("src")]
            tests_dir = os.path.abspath("tests")
            if os.path.exists(tests_dir):
                for root, _, _ in os.walk(tests_dir):
                    extra_paths.append(root)

            all_paths = list(
                dict.fromkeys(extra_paths + [os.path.abspath(p) for p in sys.path if p])
            )
            python_path = ":".join(all_paths)

            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": python_path,
                        **numpy_hugepage_env(),
                        **references.reference_env(),
                    },
                },
            )

    def compile(
        self,
        ir_sinks: List[IROp],
        *,
        name: str = "",
        abort_targets: Sequence[Any] = (),
    ) -> RayCompiledPipeline:
        """Compile a list of IR sink nodes into an executable RayCompiledPipeline.

        Plans the stages of the pipeline and the placement of its pools on
        the cluster, without starting any actor: the pipeline starts them on
        ``start``, or when it is first used.

        Args:
            ir_sinks: Terminal IR nodes of the pipeline.
            name: Prefix of the stage labels and actor names, identifying a
                nested pipeline (e.g. a ``group_by`` child) in stats and the
                Ray dashboard.
            abort_targets: Further actor handles aborted when a stage of this
                pipeline fails, in addition to its own sources. A nested
                pipeline passes the sources of its parent, so that a failure
                stops the parent even if it never sends to the failed stage
                again.

        Raises:
            ValueError: If there are no sinks, or the claims of the maps
                cannot be satisfied by the cluster.
        """
        plan = self._plan(ir_sinks, name=name, abort_targets=abort_targets)
        return RayCompiledPipeline(self, plan)

    def _plan(
        self,
        ir_sinks: List[IROp],
        *,
        name: str,
        abort_targets: Sequence[Any],
    ) -> _PipelinePlan:
        """Plan the stages of a pipeline and its pools, without starting them."""
        if not ir_sinks:
            raise ValueError("RayBackend.compile() requires at least one IRSink.")

        nodes = topological_order(ir_sinks)
        labels = stage_labels(nodes, name)
        # The maps of the children of group_by share the pools of this
        # pipeline, so the claims cover them too.
        maps = all_maps(nodes)
        claims = {node.compute: self._claim(node.compute) for node in maps}
        # The children of a group_by run on the pools of their parent.
        pools = (
            None if self.pools is not None else self._plan_pools(ir_sinks, maps, claims)
        )
        boundaries = _boundary_targets(
            nodes,
            labels,
            None if pools is None else pools.placement,
        )
        # The stages behind a phase boundary have an actor of their own, whose
        # mailbox takes the backlog of the earlier phases without bound: the
        # stages before it never wait for it, so FLUSH reaches their pools.
        spill, spill_root = self._spill_config()
        if boundaries and spill is None:
            logger.warning(
                "Spilling is off, but the stages %s wait for an earlier "
                "phase: they spill their backlog anyway.",
                sorted(labels[node] for node in boundaries),
            )
            boundary_spill, spill_root = _default_spill(DEFAULT_SPILL_THRESHOLD)
        else:
            boundary_spill = spill

        return _PipelinePlan(
            # Actors are named so that stages can be identified in the Ray
            # dashboard; the pipeline id keeps the names unique within the
            # Ray namespace.
            pipeline_id=uuid.uuid4().hex[:8],
            name=name,
            abort_targets=tuple(abort_targets),
            nodes=tuple(nodes),
            labels=labels,
            maps=tuple(maps),
            claims=claims,
            pools=pools,
            boundaries=boundaries,
            spill=spill,
            boundary_spill=boundary_spill,
            spill_root=spill_root,
        )

    def _launch(self, plan: _PipelinePlan) -> Tuple[_Runtime, List[ray.ObjectRef]]:
        """Create the actors of a planned pipeline and start connecting them.

        Every IR node but the maps gets an actor; maps run on the routes out
        of the stage upstream of them, as Ray tasks or on the executors of a
        pool.

        Returns:
            The actors, and the calls connecting its stages; it is ready once
            they are done.
        """
        labels = plan.labels
        profile, owns_profile = self._profile_config(plan.pipeline_id)
        # Map calls and the warm-up of executors find their functions in the
        # function table of the pipeline, put in the object store once.
        functions = ray.put(
            FunctionTable(
                functions={
                    labels[node]: node.func
                    for node in plan.nodes
                    if isinstance(node, IRMap)
                },
                parameters=self.parameters,
                profile=profile,
                preparers={
                    compute.group: compute.prepare
                    for node in plan.maps
                    if isinstance(compute := node.compute, ComputeAssignment)
                    and compute.prepare is not None
                },
            ),
        )
        if plan.pools is None:
            assert self.pools is not None, "A pipeline without pools of its own."
            pools = dict(self.pools)
        else:
            pools = self._start_pools(
                plan.pools,
                Ref(functions),
                profile,
                plan.pipeline_id,
            )
        # Every executor of a pool takes calls of a map at once; the items on
        # the routes of a stage must cover the replicas a protocol declared.
        groups = {
            compute.group
            for node in plan.nodes
            if isinstance(node, IRMap)
            and isinstance(compute := node.compute, ComputeAssignment)
        }
        pooled_calls = max(
            (
                pool.replicas * _EXECUTOR_CALLS_IN_FLIGHT
                for group, pool in pools.items()
                if group in groups and pool.declared and pool.replicas > 1
            ),
            default=0,
        )
        config = _StageConfig(
            queue_size=self.queue_size,
            max_in_flight=max(self.max_in_flight, pooled_calls),
            buffer_size=self.buffer_size,
            spill=plan.spill,
            profile=profile,
        )
        boundary_config = dataclasses.replace(
            config,
            buffer_size=None,
            spill=plan.boundary_spill,
        )
        actor_map: Dict[IROp, Any] = {
            node: actor_cls.options(
                name=f"{plan.pipeline_id}:{labels[node]}",
                runtime_env={
                    "env_vars": {**numpy_hugepage_env(), **references.reference_env()},
                },
            ).remote(**kwargs)
            for node in plan.nodes
            if not _is_hosted(node, plan.boundaries)
            for actor_cls, kwargs in [
                self._stage_actor(
                    node,
                    labels[node],
                    boundary_config if node in plan.boundaries else config,
                    pools,
                ),
            ]
        }
        sources = {
            source_name: actor_map[node] for source_name, node in plan.sources.items()
        }

        # Wire the routes and source actor handles and start the stage loops.
        # Every stage feeds a downstream stage on the port given by its
        # position among the upstream stages of the downstream stage; maps in
        # between are applied on the way. A failing stage aborts the sources
        # and the further abort targets.
        builder = _RouteBuilder(
            actor_map,
            labels,
            pools,
            plan.claims,
            Ref(functions),
            plan.boundaries,
        )
        abort_handles = [*sources.values(), *plan.abort_targets]
        # The stages running maps of a pool tell it when FLUSH has passed
        # them; the children of a group_by count as their router.
        hosts = (
            {}
            if plan.pools is None
            else _pool_hosts(plan.nodes, labels, plan.claims, plan.boundaries)
        )
        connected = [
            *(
                actor.connect.remote(
                    builder.stage_routes(node),
                    abort_handles,
                    max(1, len(node.upstream)),
                    [
                        pools[group].coordinator
                        for group, stages in hosts.items()
                        if labels[node] in stages
                    ],
                )
                for node, actor in actor_map.items()
            ),
            *(
                pools[group].coordinator.set_hosts.remote(sorted(stages))
                for group, stages in hosts.items()
            ),
        ]

        runtime = _Runtime(
            sources=sources,
            stages={
                labels[node]: (plan.host_of(node), actor_map[plan.host_of(node)])
                for node in plan.nodes
            },
            demuxes=[
                actor_map[node] for node in plan.nodes if isinstance(node, IRDemux)
            ],
            # The pools of children of a group_by belong to their parent.
            groups=(
                []
                if plan.pools is None
                else [pool.coordinator for pool in pools.values()]
            ),
            profile=profile,
            owns_profile=owns_profile,
            functions=functions,
            recorder=make_recorder(profile, "driver"),
        )
        return runtime, connected

    def _profile_config(
        self,
        pipeline_id: str,
    ) -> Tuple[Optional[ProfileConfig], bool]:
        """Profiling of a started pipeline, and whether the pipeline owns it."""
        match self.profile:
            case None | False:
                return None, False
            case ProfileConfig() as shared:
                return shared, False
            case True:
                path = Path(f"ray_profile_{pipeline_id}.json")
                return start_profile(path, self.profile_log_level), True
            case Path() as path:
                return start_profile(path, self.profile_log_level), True

    def _spill_config(self) -> Tuple[Optional[_SpillConfig], Optional[Path]]:
        """Spilling of a compiled pipeline, with the temporary directory it owns."""
        if self.spill_threshold is None:
            return None, None
        if self.spill_store is None:
            return _default_spill(self.spill_threshold)
        else:
            return _SpillConfig(self.spill_threshold, self.spill_store), None

    def _stage_actor(
        self,
        node: IROp,
        label: str,
        config: _StageConfig,
        pools: Mapping[str, _Pool],
    ) -> Tuple[Any, Dict[str, Any]]:
        """The actor class of a stage, with its constructor arguments.

        The router of a ``group_by`` gets the pools of the pipeline, which its
        children share.
        """
        spill = None if config.spill is None else config.spill.policy(label)
        match node:
            case IRMap():
                return RayMergeActor, {
                    "config": config,
                    "spill": spill,
                    "label": label,
                }
            case IRSource(name=source_name):
                return RaySourceActor, {
                    "name": source_name,
                    "config": config,
                    "spill": spill,
                    "label": label,
                }
            case IRAccumulate(
                accumulate_fn=accumulate_fn,
                initial_state_fn=initial_state_fn,
                flush_fn=flush_fn,
                tag_inputs=tag_inputs,
            ):
                return RayAccumulatorActor, {
                    "accumulate_fn": accumulate_fn,
                    "initial_state_fn": initial_state_fn,
                    "config": config,
                    "spill": spill,
                    "label": label,
                    "parameters": self.parameters,
                    "flush_fn": flush_fn,
                    "tag_inputs": tag_inputs,
                }
            case IRSink(writer=writer):
                return RaySinkActor, {
                    "writer": writer,
                    "config": config,
                    "spill": spill,
                    "label": label,
                }
            case IRDemux(
                key_fn=key_fn,
                template=template,
                source_name=template_source,
                max_keys=max_keys,
                workers=workers,
                outer_keyed=outer_keyed,
            ):
                assert template is not None, "IRDemux has no template."
                return RayDemuxActor, {
                    "key_fn": key_fn,
                    "template": template,
                    "source_name": template_source,
                    "max_keys": max_keys,
                    "workers": self._group_by_workers(workers),
                    "outer_keyed": outer_keyed,
                    "backend": self._child_backend(config, pools, template),
                    "config": config,
                    "spill": spill,
                    "label": label,
                    "parameters": self.parameters,
                }
            case _:
                raise NotImplementedError(
                    f"Unsupported IR op type for Ray backend: {type(node).__name__}",
                )

    def _child_backend(
        self,
        config: _StageConfig,
        pools: Mapping[str, _Pool],
        template: IROp,
    ) -> RayBackend:
        """The backend of the children of a ``group_by`` with ``template``.

        The children buffer, spill and profile like the router, and run
        their maps on the pools of its pipeline.
        """
        spill = config.spill
        return RayBackend(
            init_ray=False,
            queue_size=config.queue_size,
            max_in_flight=config.max_in_flight,
            buffer_size=config.buffer_size,
            spill_threshold=None if spill is None else spill.threshold,
            spill_store=None if spill is None else spill.store,
            parameters=self.parameters,
            gpu_memory=(
                self._cluster_gpu_memory()
                if _needs_gpu_memory(template)
                else self._gpu_memory
            ),
            profile=config.profile,
            group_by_workers=self.group_by_workers,
            ephemeral_idle_s=self.ephemeral_idle_s,
            pools=pools,
            placement=self.placement,
        )

    def _group_by_workers(self, workers: GroupByWorkers) -> Optional[int]:
        """Children of a ``group_by`` setting ``workers``."""
        match workers:
            case WorkersFrom.BACKEND:
                return self.group_by_workers
            case _:
                return workers

    def _claim(self, compute: Optional[ComputeAssignment]) -> Dict[str, Any]:
        """Ray options of the calls of a map, with the GPU claim resolved.

        Raises:
            ValueError: If a map requires more resources than the cluster
                has; Ray would wait for them forever.
        """
        if compute is None:
            return _ray_options(ComputeResources(), 0)
        else:
            num_gpus = self._num_gpus(compute.resources)
            _check_cluster_fits(compute, num_gpus)
            return _ray_options(compute.resources, num_gpus)

    def _num_gpus(self, resources: ComputeResources) -> float:
        if resources.min_vram is None:
            return resources.gpus
        else:
            return _resolve_gpus(resources, self._cluster_gpu_memory())

    def _cluster_gpu_memory(self) -> Tuple[float, ...]:
        if self._gpu_memory is None:
            self._gpu_memory = _probe_cluster_gpu_memory()
        return self._gpu_memory

    def _plan_pools(
        self,
        ir_sinks: Sequence[IROp],
        maps: Sequence[IRMap],
        claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]],
    ) -> _PoolPlan:
        """Plan the pool of every group of ``maps`` that runs on executors.

        The placement planner (``placement.py``) decides the executors of
        every pool.

        Raises:
            ValueError: If the claims cannot be satisfied, or the
                long-running pools do not fit the cluster at once.
        """
        # The maps of a group, once each: their copies in the children of a
        # group_by keep the fingerprint.
        stages: Dict[str, Set[Optional[str]]] = {}
        options: Dict[str, Dict[str, Any]] = {}
        for node in maps:
            group = _pooled_group(node, claims)
            if group is not None:
                stages.setdefault(group, set()).add(node.stage_fp)
                options[group] = claims[node.compute]

        graph = report_of(ir_sinks)
        demands = demands_of(maps, self._num_gpus)
        cluster = _cluster_nodes()
        plan = plan_placement(
            graph,
            demands,
            [gpus for _, gpus in cluster],
            ray.cluster_resources().get("CPU", 0),
        )
        return _PoolPlan(
            placement=_Placement(graph, plan, demands, self.placement == "managed"),
            waits=blockers_of(plan, demands),
            members={
                group: len(fingerprints) for group, fingerprints in stages.items()
            },
            options=options,
            node_ids=tuple(node_id for node_id, _ in cluster),
        )

    def _start_pools(
        self,
        pools: _PoolPlan,
        functions: Ref,
        profile: Optional[ProfileConfig],
        pipeline_id: str,
    ) -> Dict[str, _Pool]:
        """Create the coordinators of the planned pools.

        Long-running pools of the first phase prepare their executors while
        the rest of the pipeline is created; ephemeral pools start executors
        on demand.
        """
        placement = pools.placement
        plan, demands = placement.plan, placement.demands
        # The pools of a phase start executors once the pools of the earlier
        # phases have released their GPUs: with managed placement, the pools
        # on the same GPUs; else all of them, since Ray picks the GPUs.
        started: Dict[str, _Pool] = {}
        for phase in plan.phases:
            earlier = dict(started)
            started.update(
                {
                    group: _Pool(
                        coordinator=RayComputeGroup.options(
                            runtime_env={"env_vars": numpy_hugepage_env()},
                        ).remote(
                            pools.options[group],
                            pools.members[group],
                            plan.groups[group].replicas,
                            functions,
                            self.parameters,
                            group,
                            profile,
                            idle_s=(
                                None if demands[group].holds else self.ephemeral_idle_s
                            ),
                            blockers=[
                                pool.coordinator
                                for other, pool in earlier.items()
                                if other in pools.waits[group] or not placement.managed
                            ],
                            reservation=(
                                _reservation_of(
                                    f"{pipeline_id}/{group}",
                                    bundles_of(plan.groups[group], demands[group]),
                                    pools.node_ids,
                                )
                                if placement.managed
                                else None
                            ),
                        ),
                        replicas=plan.groups[group].replicas,
                        ephemeral=not demands[group].holds,
                        declared=demands[group].replicas is not None,
                    )
                    for group in phase
                },
            )
        # Long-running pools of the first phase prepare their executors while
        # the pipeline is created; those of later phases on their first call.
        # A failed warm-up surfaces on the first lease, which prepares again.
        for group in plan.phases[0] if plan.phases else ():
            if demands[group].holds:
                started[group].coordinator.start.remote()
        return started


@dataclass(frozen=True)
class _Placement:
    """The plan a pipeline runs with, for its report."""

    graph: PipelineReport
    plan: Plan
    demands: Mapping[str, GroupDemand]
    managed: bool = True

    def layer(self) -> Layer:
        return placement_layer(self.graph, self.plan, self.demands, self.managed)


def _cluster_nodes() -> List[Tuple[str, int]]:
    """The id and the GPUs of every live node of the Ray cluster."""
    return [
        (node["NodeID"], int(node["Resources"].get("GPU", 0)))
        for node in ray.nodes()
        if node["Alive"]
    ]


@dataclass(frozen=True)
class _Pool:
    """The pool of executors of a compute group."""

    coordinator: Any
    replicas: int
    # Whether the pool releases idle executors (ephemeral protocols).
    ephemeral: bool
    # Whether the protocol declared its replicas (a number or "auto").
    declared: bool


def _pooled(
    compute: Optional[ComputeAssignment],
    options: Mapping[str, Any],
) -> bool:
    """Whether the calls of a map run on the executors of a pool."""
    return compute is not None and pooled(compute.resources, options["num_gpus"])


def _without_prepare(
    compute: Optional[ComputeAssignment],
) -> Optional[ComputeAssignment]:
    """The claim without its warm-up, which holds the protocol.

    Routes are sent to every stage; the warm-up reaches the executors through
    the function table instead.
    """
    return None if compute is None else dataclasses.replace(compute, prepare=None)


def _stage_of(node: IRMap, label: str) -> str:
    """The fingerprint of a map of a protocol, or its label outside of protocols.

    Only maps of protocols run on executors, which lock by it.
    """
    return label if node.stage_fp is None else node.stage_fp


def _pool_of(
    pools: Mapping[str, _Pool],
    compute: Optional[ComputeAssignment],
    options: Mapping[str, Any],
) -> Optional[_Pool]:
    """The pool running the calls of a map, if they run on executors."""
    if compute is not None and _pooled(compute, options):
        return pools[compute.group]
    else:
        return None


@dataclass
class _RouteBuilder:
    """Builds the routes of the stages with actors, through the hosted maps."""

    actor_map: Mapping[IROp, Any]
    labels: Mapping[IROp, str]
    pools: Mapping[str, _Pool]
    claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]]
    functions: Ref
    # The maps with an actor of their own, though they have a single input.
    own_actors: FrozenSet[IROp] = frozenset()

    def stage_routes(self, node: IROp) -> List[_Route]:
        """Routes out of the actor of ``node``.

        A merging map runs on the route out of its own actor.
        """
        match node:
            case IRMap():
                return [self._apply(node)]
            case _:
                return self._routes(node)

    def _apply(self, node: IRMap) -> _Apply:
        options = self.claims[node.compute]
        pool = _pool_of(self.pools, node.compute, options)
        if pool is None:
            group, replicas, returns_idle_leases = None, 1, False
        else:
            group, replicas = pool.coordinator, pool.replicas
            returns_idle_leases = pool.ephemeral
        return _Apply(
            label=self.labels[node],
            functions=self.functions,
            compute=_without_prepare(node.compute),
            options=options,
            group=group,
            replicas=replicas,
            returns_idle_leases=returns_idle_leases,
            stage=_stage_of(node, self.labels[node]),
            routes=self._routes(node),
        )

    def _routes(self, node: IROp) -> List[_Route]:
        return [self._route(node, downstream) for downstream in node.downstream]

    def _route(self, node: IROp, downstream: IROp) -> _Route:
        # Every stage feeds a downstream stage on the port given by its
        # position among the upstream stages of the downstream stage.
        match downstream:
            case IRMap() if _is_hosted(downstream, self.own_actors):
                return self._apply(downstream)
            case _:
                return _Push(
                    self.actor_map[downstream],
                    downstream.upstream.index(node),
                )


def _stats_of(
    map_stats: Optional[StageStats],
    stage_stats: StageStats,
    own_actor: bool,
) -> StageStats:
    """The stats of a stage: of its actor, or of the calls of a map.

    A map on an actor of its own also reports how items waited for the
    actor (buffering and spilling), which its calls do not see: its items
    wait in the mailbox of the actor, then for an executor.
    """
    if map_stats is None:
        return stage_stats
    elif own_actor:
        return dataclasses.replace(
            map_stats,
            buffered_peak=stage_stats.buffered_peak,
            spilled=stage_stats.spilled,
            spill_s=stage_stats.spill_s,
            spill_bytes=stage_stats.spill_bytes,
            reread_s=stage_stats.reread_s,
            fetch_s=stage_stats.fetch_s,
            wait_s=map_stats.wait_s + stage_stats.wait_s,
            wait_max_s=max(map_stats.wait_max_s, stage_stats.wait_max_s),
        )
    else:
        return map_stats


def _is_hosted(node: IROp, own_actors: FrozenSet[IROp]) -> bool:
    """Whether ``node`` runs on the actor of another stage.

    Maps with a single input do, unless they have an actor of their own
    (``own_actors``, the stages behind a phase boundary).
    """
    match node:
        case IRMap(upstream=[_]) if node not in own_actors:
            return True
        case _:
            return False


def _host_of(node: IROp, own_actors: FrozenSet[IROp]) -> IROp:
    """The stage whose actor runs ``node``.

    A map with a single input runs on the actor of the nearest upstream
    stage; a map merging several inputs, or in ``own_actors``, has an actor
    of its own.
    """
    match node:
        case IRMap(upstream=[upstream]) if node not in own_actors:
            return _host_of(upstream, own_actors)
        case _:
            return node


def _default_spill(threshold: int) -> Tuple[_SpillConfig, Path]:
    """Spilling into the temporary directory of the node of every stage, with
    the directory of the driver's node, removed by the pipeline.

    Every stage removes its directory when it closes; most pipelines never
    spill, and the directories are created by the first spilled item.
    """
    root = Path(f"scipion_bridge_spill_{uuid.uuid4().hex}")
    return (
        _SpillConfig(threshold, pickle_spill_store(root)),
        Path(tempfile.gettempdir()) / root,
    )


def _boundary_targets(
    nodes: Sequence[IROp],
    labels: Mapping[IROp, str],
    placement: Optional[_Placement],
) -> FrozenSet[IROp]:
    """The stages that the plan puts behind a phase boundary."""
    if placement is None:
        return frozenset()
    else:
        targets = {target for _, target in placement.plan.boundaries}
        return frozenset(node for node in nodes if labels[node] in targets)


def _pooled_group(
    node: IRMap,
    claims: Mapping[Optional[ComputeAssignment], Mapping[str, Any]],
) -> Optional[str]:
    """The group of a map whose calls run on a pool, else None."""
    compute = node.compute
    if compute is not None and _pooled(compute, claims[compute]):
        return compute.group
    else:
        return None


def _pool_hosts(
    nodes: Sequence[IROp],
    labels: Mapping[IROp, str],
    claims: Mapping[Optional[ComputeAssignment], Mapping[str, Any]],
    own_actors: FrozenSet[IROp],
) -> Dict[str, Set[str]]:
    """The labels of the stage actors running maps of every pool.

    A map runs on its host (``_host_of``); the maps in the template of a
    ``group_by`` run in its children, which flush before their router.
    """
    hosts: Dict[str, Set[str]] = {}
    for node in nodes:
        match node:
            case IRMap():
                stages = [(node, labels[_host_of(node, own_actors)])]
            case IRDemux(template=IROp() as template):
                stages = [(inner, labels[node]) for inner in all_maps([template])]
            case _:
                stages = []
        for inner, stage in stages:
            group = _pooled_group(inner, claims)
            if group is not None:
                hosts.setdefault(group, set()).add(stage)
    return hosts


def _needs_gpu_memory(node: IROp) -> bool:
    """Whether a map upstream of ``node``, or in a ``group_by`` there, sets min_vram."""
    match node:
        case IRMap(
            compute=ComputeAssignment(
                resources=ComputeResources(min_vram=float() | int())
            )
        ):
            return True
        case IRDemux(template=IROp() as template) if _needs_gpu_memory(template):
            return True
        case _:
            return any(_needs_gpu_memory(upstream) for upstream in node.upstream)


def _check_cluster_fits(assignment: ComputeAssignment, num_gpus: float) -> None:
    resources = assignment.resources
    cluster = ray.cluster_resources()
    gpus, cpus = cluster.get("GPU", 0), cluster.get("CPU", 0)
    reserved_cpus = 0 if resources.cpus is None else resources.cpus
    if num_gpus > gpus or reserved_cpus > cpus:
        raise ValueError(
            f"The stages of protocol {assignment.group} require {resources} "
            f"({num_gpus:g} GPUs), "
            f"but the Ray cluster has {gpus} GPUs and {cpus} CPUs.",
        )
