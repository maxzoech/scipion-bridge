from __future__ import annotations
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import abc
import asyncio
import contextlib
import dataclasses
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import ray  # type: ignore
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # type: ignore

from scipion_bridge.core.environment.compute import (
    CPUS_ENV_VAR,
    ComputeAssignment,
    ComputeResources,
    TaskType,
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
    Keyed,
    Tagged,
    WorkersFrom,
    clone_ir,
)
from scipion_bridge.core.streaming.keyed import share_keys
from scipion_bridge.core.streaming.node import FlushSignal, FLUSH
from scipion_bridge.core.streaming.sink_writer import SinkWriter
from scipion_bridge.core.streaming.spill import SpillStoreFactory, pickle_spill_store

from .function_table import FunctionTable, table_of
from .mailbox import Mailbox, Ref, SpillPolicy
from .profiling import ProfileConfig, Recorder, make_recorder, start_profile

logger = logging.getLogger(__name__)

_Barrier = asyncio.Future[None]
_InboxEntry = Tuple[Any, int, Optional[_Barrier]]
_OutboxEntry = Tuple[Any, List[_Barrier]]
# An item ready to be pushed to a downstream stage: (actor, port, item).
_Delivery = Tuple[Any, int, Any]
# Items being routed in order, or a FLUSH (None) with its barriers.
_InFlightEntry = Tuple[Optional["asyncio.Task[List[_Delivery]]"], List[_Barrier]]


@dataclass(frozen=True)
class _Failure:
    """An error sent to other stages to abort them.

    Ray treats a ``RayTaskError`` passed directly as an argument as a failed
    input and fails the call instead of delivering the error, so it is
    wrapped.
    """

    error: BaseException


@dataclass
class _Push:
    """Route of items to a downstream stage, on the port it receives them."""

    handle: Any
    port: int


@dataclass
class _Apply:
    """Route of items through a map, run as a Ray task, then on to ``routes``.

    Maps are stateless, so they need no actor of their own: the stage
    upstream of a map submits its calls. Chains and branches of maps nest.
    """

    label: str
    # The function table of the pipeline, holding the function of the map.
    functions: Ref
    compute: Optional[ComputeAssignment]
    # Ray options of the task of every call (resources, runtime env).
    options: Dict[str, Any]
    # Coordinator of the pool of a long-running compute group.
    group: Optional[Any]
    # Executors in the pool of the group.
    replicas: int
    # Whether the map returns its leases when it has no calls in flight, so
    # that the pool can release idle executors (ephemeral protocols).
    returns_idle_leases: bool
    # Fingerprint of the map, which runs one call at a time on an executor.
    stage: str
    routes: List[_Route]


_Route = Union[_Push, _Apply]


# Items on the routes out of a stage at once, by default. An item holds its
# slot through every map on the route until it is delivered, so the slots
# must cover a slow map (e.g. CPU post-processing) behind a fast one (e.g. a
# GPU model); the maps themselves limit how many of their calls run.
_DEFAULT_MAX_IN_FLIGHT = 16

# Items waiting for a stage at most, by default. Beyond it, the stages
# upstream wait, as far as the driver: waiting items are held in the object
# store, so an unbounded backlog fills it and stalls every stage (Ray then
# spills and restores objects at disk speed). The bound counts items, not
# bytes; a stage holds a few more in its inbox, outbox and in flight.
DEFAULT_BUFFER_SIZE = 4

# Waiting items a stage keeps in the object store, by default; further items
# go to the spill store. With the default buffer size nothing spills, but a
# larger or unbounded buffer keeps its backlog on disk.
DEFAULT_SPILL_THRESHOLD = 4


@dataclass(frozen=True)
class _SpillConfig:
    """Spill items once ``threshold`` items wait for a stage, into ``store``."""

    threshold: int
    store: SpillStoreFactory

    def policy(self, label: str) -> SpillPolicy:
        """The spill policy of the stage labelled ``label``."""
        return SpillPolicy(threshold=self.threshold, store=self.store(label))


@dataclass(frozen=True)
class _StageConfig:
    """Buffering of the stages of a pipeline (see ``RayBackend``)."""

    queue_size: int
    max_in_flight: int
    buffer_size: Optional[int]
    spill: Optional[_SpillConfig]
    profile: Optional[ProfileConfig]


def _fail(barriers: List[_Barrier], error: BaseException) -> None:
    """Fail pending FLUSH barriers."""
    for barrier in barriers:
        barrier.set_exception(error)


def _release(barriers: List[_Barrier]) -> None:
    """Release FLUSH barriers whose flush has propagated."""
    for barrier in barriers:
        barrier.set_result(None)


def _transport(item: Any) -> Any:
    """The form in which ``item`` is sent to another stage.

    Results of maps are already in the object store; they travel as a
    ``Ref``, which the receiving stage fetches only when it processes the
    item. Other items travel inline in the call, which is cheapest for the
    many small items of e.g. a ``flatten``; Ray moves large ones through the
    object store by itself.
    """
    match item:
        case ray.ObjectRef():
            return Ref(item)
        case _:
            return item


async def _deliver(handle: Any, item: Any, port: int) -> None:
    """Push ``item`` into the stage of ``handle`` on ``port``.

    Returns once the item is buffered by the stage, or for FLUSH once the
    flush has propagated through the stage and its downstream.
    """
    await handle.push.remote(_transport(item), port)


class _PipelinedStage(abc.ABC):
    """Pipelined stage of a compiled Ray pipeline.

    Every stage decouples receiving, computing and forwarding items so that
    consecutive stages run concurrently:

    - ``push`` buffers an item in the mailbox and returns, so a busy stage
      does not stall the stages upstream of it while it has room. Once
      ``buffer_size`` items wait, ``push`` waits for room, which propagates
      backpressure upstream to the driver. Results of maps travel between
      stages as references into the object store, other items inline; beyond
      the spill threshold, waiting items are spilled to a ``SpillStore``.
    - A fetch task reads the waiting items in order into a bounded inbox, so
      up to ``queue_size`` items are read ahead of the compute task.
    - A single compute task processes items one at a time in arrival order, so
      stage logic (e.g. a GPU model or accumulator state) never runs
      concurrently.
    - A single emitter task routes results downstream, so the compute task
      is never stalled by serialization or downstream stages. Results pass
      the maps on their routes (Ray tasks) with up to ``max_in_flight`` items
      in flight; a forwarder task pushes them on in order.

    ``FLUSH`` acts as a barrier: ``push(FLUSH)`` returns only after all
    preceding items and the flush itself have propagated through every
    downstream stage. A stage with several upstream stages receives one FLUSH
    from each; it finalizes and forwards a single FLUSH once all have arrived.

    Each item is pushed with the ``port`` it arrives on: the position of the
    sending stage among the upstream stages of the receiver. Stages with
    several inputs use it to tell their inputs apart.

    A failure in a stage or downstream is stored and raised by every
    subsequent ``push``, so it propagates upstream to the driver. The failing
    stage also aborts the sources of the pipeline, so the driver's next
    ``send`` fails immediately instead of only at the final FLUSH (a failed
    stage may never receive another item, e.g. after a ``collect``).
    """

    def __init__(
        self,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
    ) -> None:
        """
        Args:
            config: Buffering of the stage.
            spill: Where the items overflowing the mailbox are spilled, or
                ``None`` to keep every item in memory.
            label: Label of the stage in stats and the profile.
        """
        self._stats = StageStats()
        self._recorder = make_recorder(config.profile, label)
        self._mailbox = Mailbox(
            self._stats,
            config.buffer_size,
            spill,
            self._recorder,
        )
        self._inbox: asyncio.Queue[_InboxEntry] = asyncio.Queue(
            maxsize=config.queue_size,
        )
        self._outbox: asyncio.Queue[_OutboxEntry] = asyncio.Queue(
            maxsize=config.queue_size,
        )
        # Items being routed, in order; the slots bound how many of them run
        # their maps at once.
        self._in_flight: asyncio.Queue[_InFlightEntry] = asyncio.Queue()
        self._in_flight_slots = asyncio.Semaphore(config.max_in_flight)
        self._routes: List[_Route] = []
        self._maps: Dict[str, _MapRunner] = {}
        self._abort_targets: List[Any] = []
        self._num_inputs = 1
        self._pending_flushes: List[_Barrier] = []
        self._error: Optional[BaseException] = None
        self._tasks: List[asyncio.Task[None]] = []

    async def connect(
        self,
        routes: List[_Route],
        abort_targets: List[Any],
        num_inputs: int,
    ) -> None:
        """Configure the stage and start its loops.

        Args:
            routes: Routes of the results to the downstream stages, through
                the maps between them.
            abort_targets: Actor handles aborted when this stage fails: the
                sources of the pipeline and, for a nested pipeline, the
                sources of its parent.
            num_inputs: Number of upstream stages, each sending one FLUSH.
        """
        self._routes = routes
        completions = _Completions()
        self._maps = {
            route.label: _MapRunner(route, completions, self._recorder)
            for route in _walk_routes(routes)
            if isinstance(route, _Apply)
        }
        self._abort_targets = abort_targets
        self._num_inputs = num_inputs
        self._tasks = [
            asyncio.create_task(self._run_fetch()),
            asyncio.create_task(self._run_compute()),
            asyncio.create_task(self._run_emit()),
            asyncio.create_task(self._run_forward()),
        ]
        self._recorder.instant(
            "connect",
            "stage",
            inputs=num_inputs,
            routes=len(routes),
        )

    async def stats(self) -> StageStats:
        """Return the execution metrics of this stage."""
        return self._stats

    async def map_stats(self) -> Dict[str, StageStats]:
        """Return the execution metrics of the maps on the routes, by label."""
        return {label: runner.stats for label, runner in self._maps.items()}

    async def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of this stage."""
        await self._recorder.flush()

    async def abort(self, failure: _Failure) -> None:
        """Fail this stage with an error raised elsewhere in the pipeline."""
        if self._error is None:
            self._store_error(failure.error)

    def _set_error(self, error: BaseException) -> None:
        """Fail this stage and abort the abort targets on the first error."""
        if self._error is not None:
            return

        self._store_error(error)
        self._recorder.instant("error", "stage", level=logging.ERROR, error=error)
        for target in self._abort_targets:
            target.abort.remote(_Failure(error))

    def _store_error(self, error: BaseException) -> None:
        self._error = error
        _fail(self._pending_flushes, error)
        self._pending_flushes = []

    async def push(self, item: Any, port: int = 0) -> None:
        """Buffer an item; for FLUSH, wait until the pipeline has drained."""
        if self._error is not None:
            raise self._error

        match item:
            case FlushSignal():
                done: _Barrier = asyncio.get_running_loop().create_future()
                await self._mailbox.put(item, port, done)
                await done
            case _:
                await self._mailbox.put(item, port, None)

    @abc.abstractmethod
    async def process(self, item: Any, port: int) -> List[Any]:
        """Process an item arriving on ``port`` and return the items to emit."""
        ...

    async def on_flush(self) -> List[Any]:
        """Finalize the stage on FLUSH and return the items to emit downstream."""
        return []

    async def _run_fetch(self) -> None:
        while True:
            entry, port, done = await self._mailbox.get()
            try:
                item = await self._mailbox.resolve(entry)
            except Exception as error:
                self._set_error(error)
                _fail([] if done is None else [done], error)
                continue

            await self._inbox.put((item, port, done))

    async def _run_compute(self) -> None:
        while True:
            t_wait = time.perf_counter()
            item, port, done = await self._inbox.get()
            t_start = time.perf_counter()
            self._stats.idle_s += t_start - t_wait

            barriers = [] if done is None else [done]
            if self._error is not None:
                _fail(barriers, self._error)
                continue

            match item:
                case FlushSignal():
                    # Wait for the FLUSH of every upstream stage; the barriers of
                    # all of them are released by the single forwarded FLUSH.
                    self._pending_flushes.extend(barriers)
                    if len(self._pending_flushes) < self._num_inputs:
                        continue

                    barriers, self._pending_flushes = self._pending_flushes, []
                    span = self._recorder.span(
                        "on_flush", "compute", level=logging.INFO
                    )

                case _:
                    self._stats.items_in += 1
                    span = self._recorder.span(
                        "process",
                        "compute",
                        seq=self._stats.items_in,
                        port=port,
                    )

            try:
                with span:
                    emissions = await self._compute(item, port)
            except Exception as error:
                self._set_error(error)
                _fail(barriers, error)
                continue

            t_processed = time.perf_counter()
            self._stats.process_s += t_processed - t_start

            with self._recorder.span("blocked", "compute", items=len(emissions)):
                for out_item in emissions:
                    await self._outbox.put((out_item, []))
                if barriers:
                    await self._outbox.put((item, barriers))
            self._stats.blocked_s += time.perf_counter() - t_processed

    async def _compute(self, item: Any, port: int) -> List[Any]:
        match item:
            case FlushSignal():
                return await self.on_flush()

            case _:
                return await self.process(item, port)

    async def _run_emit(self) -> None:
        while True:
            item, barriers = await self._outbox.get()
            if self._error is not None:
                _fail(barriers, self._error)
                continue

            match barriers:
                case []:
                    # The maps of the item start once a slot is free; the slot
                    # is released when the item has been forwarded.
                    await self._in_flight_slots.acquire()
                    routing = asyncio.create_task(self._route(self._routes, item))
                    self._in_flight.put_nowait((routing, []))
                case _:
                    self._in_flight.put_nowait((None, barriers))

    async def _run_forward(self) -> None:
        while True:
            routing, barriers = await self._in_flight.get()
            try:
                await self._forward(routing, barriers)
            finally:
                match routing:
                    case None:
                        pass
                    case _:
                        self._in_flight_slots.release()

    async def _forward(
        self,
        routing: Optional[asyncio.Task[List[_Delivery]]],
        barriers: List[_Barrier],
    ) -> None:
        """Deliver a routed item, or a FLUSH, to the downstream stages."""
        if self._error is not None:
            _fail(barriers, self._error)
            _cancel(routing)
            return

        t_start = time.perf_counter()
        try:
            with self._forward_span(routing):
                deliveries = await self._finish(routing)
                await asyncio.gather(
                    *(_deliver(handle, item, port) for handle, port, item in deliveries)
                )
        except Exception as error:
            self._set_error(error)
            _fail(barriers, error)
            return

        match barriers:
            case []:
                self._stats.emit_s += time.perf_counter() - t_start
                self._stats.items_out += 1
            case _:
                # The events up to the FLUSH are in the trace when it returns.
                await self._recorder.flush()
                _release(barriers)

    def _forward_span(
        self,
        routing: Optional[asyncio.Task[List[_Delivery]]],
    ) -> AbstractContextManager[Dict[str, Any]]:
        """Span of forwarding an item, from its routing until its delivery."""
        match routing:
            case None:
                return self._recorder.span("flush", "forward", level=logging.INFO)
            case _:
                return self._recorder.span(
                    "forward",
                    "forward",
                    seq=self._stats.items_out + 1,
                )

    async def _finish(
        self,
        routing: Optional[asyncio.Task[List[_Delivery]]],
    ) -> List[_Delivery]:
        """Wait for the maps of an item, or prepare a FLUSH for every route.

        All earlier items have been delivered when a FLUSH is forwarded, so
        the long-running maps release their resources first.
        """
        match routing:
            case None:
                await asyncio.gather(*(m.release() for m in self._maps.values()))
                return [
                    (route.handle, route.port, FLUSH)
                    for route in _walk_routes(self._routes)
                    if isinstance(route, _Push)
                ]
            case _:
                return await routing

    async def _route(self, routes: List[_Route], item: Any) -> List[_Delivery]:
        deliveries = await asyncio.gather(
            *(self._route_one(route, item) for route in routes),
        )
        return [delivery for nested in deliveries for delivery in nested]

    async def _route_one(self, route: _Route, item: Any) -> List[_Delivery]:
        match route:
            case _Push(handle=handle, port=port):
                return [(handle, port, item)]
            case _Apply(label=label, routes=routes):
                # The result stays in the object store; the next map or the
                # receiving stage resolves it.
                result = await self._maps[label].run(item)
                return await self._route(routes, result)


def _cancel(task: Optional[asyncio.Task[Any]]) -> None:
    match task:
        case None:
            pass
        case _:
            task.cancel()


def _walk_routes(routes: List[_Route]) -> List[_Route]:
    """All routes of a route tree."""
    return [
        nested
        for route in routes
        for nested in [
            route,
            *(_walk_routes(route.routes) if isinstance(route, _Apply) else []),
        ]
    ]


@ray.remote
class RaySourceActor(_PipelinedStage):
    """Input source of the streaming pipeline, forwarding items unchanged."""

    def __init__(
        self,
        name: str,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
    ):
        super().__init__(config, spill, label)
        self.name = name

    async def process(self, item: Any, port: int) -> List[Any]:
        return [item]


@ray.remote
class RayMergeActor(_PipelinedStage):
    """Joins the inputs of a map with several upstream stages.

    The items pass unchanged; the map is applied on the route of the merged
    stream, after the FLUSH of every input arrived.
    """

    def __init__(
        self,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
    ):
        super().__init__(config, spill, label)

    async def process(self, item: Any, port: int) -> List[Any]:
        return [item]


@ray.remote
class RayAccumulatorActor(_PipelinedStage):
    """Stateful accumulation stage (IRAccumulate) of the streaming pipeline.

    With ``tag_inputs``, every item is passed to ``accumulate_fn`` as
    ``Tagged(port, item)`` so that it can tell its inputs apart.
    """

    def __init__(
        self,
        accumulate_fn: Callable[[Any, Any], tuple[Any, List[Any]]],
        initial_state_fn: Callable[[], Any],
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        parameters: Mapping[str, Any],
        flush_fn: Optional[Callable[[Any], tuple[Any, List[Any]]]] = None,
        tag_inputs: bool = False,
    ):
        from .container import configure_ray_env

        configure_ray_env(parameters=parameters)
        super().__init__(config, spill, label)
        self.accumulate_fn = accumulate_fn
        self.flush_fn = flush_fn
        self.tag_inputs = tag_inputs
        self.state = initial_state_fn()

    async def process(self, item: Any, port: int) -> List[Any]:
        value = Tagged(port=port, item=item) if self.tag_inputs else item
        self.state, emissions = await asyncio.to_thread(
            self.accumulate_fn,
            self.state,
            value,
        )
        return emissions

    async def on_flush(self) -> List[Any]:
        if self.flush_fn is None:
            return []

        self.state, emissions = await asyncio.to_thread(self.flush_fn, self.state)
        return emissions


# Parameters served to Field.value in this process, by the last map call.
_served_parameters: Optional[Dict[str, Any]] = None


def _serve_parameters(parameters: Mapping[str, Any]) -> None:
    """Serve ``parameters`` in this process, rewiring only when they change.

    Ray reuses its worker processes for many CPU map calls.
    """
    from .container import configure_ray_env

    global _served_parameters
    match _served_parameters == dict(parameters):
        case True:
            pass
        case False:
            configure_ray_env(parameters=parameters)
            _served_parameters = dict(parameters)


def _run_map_call(functions: Ref, label: str, item: Any) -> Any:
    """Run one call of a map ``label``, holding the resources of the task.

    The function comes from the pipeline's function table, unpacked once per
    process. Ray may reuse the worker process for other calls or end it, so
    the profiling events of the call are sent before it returns.
    """
    table = table_of(functions)
    _serve_parameters(table.parameters)
    recorder = make_recorder(table.profile, "task worker")
    try:
        with recorder.span("execute", f"map:{label}", gpus=ray.get_gpu_ids()):
            return table.functions[label](item)
    finally:
        recorder.flush_sync()


# Calls without GPUs; Ray reuses their worker processes. Calls claiming GPUs
# run on the executors of a pool instead: CUDA (and JAX) bind a process to the
# GPU it first used, so a worker reused by tasks would keep running on the GPU
# of its first call, while an executor keeps its GPU for its lifetime.
_run_in_task = ray.remote(_run_map_call)


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
    match resources.cpus:
        case None:
            num_cpus: float = 0
            cpu_env = {"OMP_NUM_THREADS": str(_cluster_cpus())}
        case cpus:
            num_cpus = cpus
            cpu_env = {CPUS_ENV_VAR: str(cpus)}

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


def _read_gpu_memory() -> List[float]:
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
    return _read_gpu_memory()


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
    match (resources.min_vram, gpu_memory):
        case (None, _) | (_, ()):
            pass
        case (min_vram, _):
            logger.info(
                "min_vram=%g GiB -> %g GPU, sized for the smallest GPU (%g GiB)%s.",
                min_vram,
                num_gpus,
                min(gpu_memory),
                _unused_share(min_vram, num_gpus, gpu_memory),
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


# Seconds an executor of an ephemeral protocol stays alive without calls,
# unless the backend sets it otherwise.
DEFAULT_EPHEMERAL_IDLE_S = 10.0

# Time a map of an ephemeral protocol keeps its leases without calls in
# flight, so that a short gap between items does not release the executors.
_LEASE_LINGER_S = 1.0

# Calls of a long-running map in flight at once: one computes while the
# argument of the next is transferred and deserialized, and the result of the
# previous one serialized.
_EXECUTOR_CALLS_IN_FLIGHT = 2


@ray.remote
class RayComputeExecutor:
    """Process holding the resources of a long-running group of maps.

    Every map of the group runs its calls here, so the maps share the
    resources and the process-scope resources of the protocol (e.g. a model
    on the GPU) are built once. A threaded actor: the maps call concurrently.

    Every map has ``_EXECUTOR_CALLS_IN_FLIGHT`` calls in flight, but its
    function runs one call at a time. Ray reads the argument of a call before
    it starts and writes its result after it returns, so these overlap with
    the computation of the other call (double buffering).
    """

    def __init__(
        self,
        parameters: Mapping[str, Any],
        group: str,
        profile: Optional[ProfileConfig],
    ) -> None:
        _serve_parameters(parameters)
        self._group = group
        self._locks: Dict[str, threading.Lock] = {}
        self._recorder = make_recorder(profile, f"executor({group})")
        self._recorder.instant("start", "executor", gpus=ray.get_gpu_ids())

    def call(self, functions: Ref, label: str, stage: str, item: Any) -> Any:
        """Run the function of the map ``label`` from ``functions`` on ``item``.

        The calls of a map of the protocol (``stage``, its fingerprint) run
        one at a time; the copies of a map in the children of a ``group_by``
        are the same map. So an executor runs as many calls at once as the
        protocol has maps, however many children share it.
        """
        func = table_of(functions).functions[label]
        lane = f"map:{label}"
        lock = self._locks.setdefault(stage, threading.Lock())
        with self._recorder.span("lock_wait", lane, overlapping=True):
            lock.acquire()
        try:
            with self._recorder.span("execute", lane):
                return func(item)
        finally:
            lock.release()

    def prepare(self, functions: Ref) -> None:
        """Build the process-scope resources of the group before the first call.

        Returns once the executor holds its resources, runs, and e.g. has
        loaded the model of its protocol.
        """
        with self._recorder.span("prepare", "executor", level=logging.INFO):
            table_of(functions).preparers[self._group]()

    def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of the executor."""
        self._recorder.flush_sync()


@ray.remote
class RayComputeGroup:
    """Coordinator of the pool of executors of a group of maps.

    The pool holds up to ``replicas`` executors, each building the
    process-scope resources of the protocol before it takes calls. The maps
    of the group lease executors to send their calls to: a ready executor
    nobody leases is preferred, else the ready one with the fewest leases
    that the map does not hold yet. While the pool is not full, a lease that
    finds no unleased executor starts a new one in the background.

    No lock is held while an executor starts: it may wait for GPUs that
    another pool holds, which that pool releases only if its maps can return
    their leases meanwhile. ``lease`` waits only if no executor is ready for
    the map. ``start`` creates the whole pool when the pipeline is compiled.

    The pool is shared by every pipeline running maps of the group, the
    children of ``group_by`` included. The pipeline owning the pool releases
    it once its FLUSH has passed every stage (``release_all``), killing the
    executors; executors still starting then are killed once they are ready.
    Items arriving after a flush create new ones. With
    ``idle_s`` (ephemeral protocols), an executor nobody leased for
    ``idle_s`` seconds is killed as well, so that other protocols can take
    its GPUs while the stream pauses. Maps return their leases when they have
    no calls in flight for a moment (``return_leases``).
    """

    def __init__(
        self,
        options: Dict[str, Any],
        members: int,
        replicas: int,
        functions: Ref,
        parameters: Mapping[str, Any],
        group: str,
        profile: Optional[ProfileConfig],
        idle_s: Optional[float] = None,
    ) -> None:
        self._options = options
        self._members = members
        self._replicas = replicas
        self._functions = functions
        self._idle_s = idle_s
        self._parameters = parameters
        self._group = group
        self._profile = profile
        self._recorder = make_recorder(profile, f"group({group})")
        self._executors: Dict[int, Any] = {}
        # Leases of every ready executor, by its id.
        self._leases: Dict[int, int] = {}
        # Since when executors without leases are idle, by id.
        self._idle_since: Dict[int, float] = {}
        # Executors being started, and the background tasks starting them.
        self._starting = 0
        self._tasks: Set[asyncio.Task[None]] = set()
        # Incremented when the pool is released, so that executors started
        # before are killed once ready.
        self._generation = 0
        # Error of the last executor that failed to start, raised to a lease
        # that has no executor to fall back on.
        self._failure: Optional[Exception] = None
        self._reaper: Optional[asyncio.Task[None]] = None
        self._next_id = 0
        self._changed = asyncio.Condition()

    async def lease(self, held: Sequence[int]) -> Optional[Tuple[int, Any]]:
        """Lease an executor of the pool that the map does not hold yet.

        An executor nobody leases is preferred. Else, while the pool is not
        full, the lease waits for a new executor rather than sharing a leased
        one: an executor runs one call of a map at a time, so maps sharing it
        while another one starts would queue their calls behind each other.
        The lock is released while waiting, and the map goes on with the
        executors it has. A full pool shares the executor with the fewest
        leases.

        Returns ``None`` if the pool was released at FLUSH while the lease
        waited for an executor.

        Raises:
            Exception: The error of an executor that failed to start.
        """
        async with self._changed:
            generation = self._generation
            while True:
                match generation == self._generation:
                    case True:
                        pass
                    case False:
                        return None
                others = {i: n for i, n in self._leases.items() if i not in held}
                unleased = [i for i, leases in others.items() if leases == 0]
                missing = self._replicas - len(self._executors) - self._starting
                match (unleased, self._failure, missing > 0, self._starting > 0):
                    case ([executor_id, *_], _, _, _):
                        return self._hand_out(executor_id)
                    case ([], Exception() as failure, _, _):
                        self._failure = None
                        raise failure
                    case ([], None, True, _):
                        self._start_one()
                        await self._changed.wait()
                    case ([], None, False, True):
                        await self._changed.wait()
                    case _ if others:
                        return self._hand_out(min(others, key=others.__getitem__))
                    case _:
                        await self._changed.wait()

    def _hand_out(self, executor_id: int) -> Tuple[int, Any]:
        """Lease the executor ``executor_id``; call with the lock held."""
        self._leases[executor_id] += 1
        self._idle_since.pop(executor_id, None)
        return executor_id, self._executors[executor_id]

    async def start(self) -> None:
        """Start the missing executors of the pool and wait until they are ready.

        Raises:
            Exception: The error of an executor that failed to start.
        """
        async with self._changed:
            missing = self._replicas - len(self._executors) - self._starting
            for _ in range(missing):
                self._start_one()
            await self._changed.wait_for(
                lambda: len(self._executors) >= self._replicas
                or self._failure is not None,
            )
            match self._failure:
                case None:
                    pass
                case failure:
                    self._failure = None
                    raise failure

    def _start_one(self) -> None:
        """Start an executor in the background; call with the lock held."""
        self._starting += 1
        task = asyncio.create_task(self._start_executor(self._generation))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _start_executor(self, generation: int) -> None:
        self._recorder.instant("executor_create", "group")
        executor: Any = RayComputeExecutor.options(
            max_concurrency=self._members * _EXECUTOR_CALLS_IN_FLIGHT,
            **self._options,
        ).remote(self._parameters, self._group, self._profile)
        try:
            await executor.prepare.remote(self._functions)
        except Exception as error:
            ray.kill(executor)
            async with self._changed:
                self._starting -= 1
                self._failure = error
                self._changed.notify_all()
        else:
            async with self._changed:
                self._starting -= 1
                match generation == self._generation:
                    case True:
                        self._executors[self._next_id] = executor
                        self._leases[self._next_id] = 0
                        self._next_id += 1
                    case False:
                        # The pool was released while the executor started.
                        ray.kill(executor)
                self._changed.notify_all()

    async def return_leases(self, executor_ids: Sequence[int]) -> None:
        """Return the leases of a map without calls in flight."""
        async with self._changed:
            self._return(executor_ids)

    def _return(self, executor_ids: Sequence[int]) -> None:
        # Leases of executors released meanwhile are gone.
        executor_ids = [i for i in executor_ids if i in self._leases]
        for executor_id in executor_ids:
            self._leases[executor_id] -= 1
        match self._idle_s:
            case None:
                pass
            case idle_s:
                now = time.monotonic()
                for executor_id in executor_ids:
                    if self._leases[executor_id] == 0:
                        self._idle_since[executor_id] = now
                if self._reaper is None:
                    self._reaper = asyncio.create_task(self._reap(idle_s))

    async def _reap(self, idle_s: float) -> None:
        """Kill the executors that have been idle for ``idle_s`` seconds."""
        while True:
            await asyncio.sleep(idle_s / 2)
            async with self._changed:
                now = time.monotonic()
                expired = [
                    executor_id
                    for executor_id, since in self._idle_since.items()
                    if now - since >= idle_s
                ]
                await self._kill(expired)

    async def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of the group."""
        await self._recorder.flush()

    async def describe(self) -> Dict[str, Any]:
        """The state of the pool, read without waiting for its lock."""
        return {
            "group": self._group,
            "locked": self._changed.locked(),
            "replicas": self._replicas,
            "executors": sorted(self._executors),
            "starting": self._starting,
            "leases": dict(self._leases),
            "idle": sorted(self._idle_since),
        }

    async def release_all(self) -> None:
        """Kill the executors of the pool once its pipeline has flushed."""
        async with self._changed:
            self._generation += 1
            self._changed.notify_all()
            await self._kill(list(self._executors))

    async def _kill(self, executor_ids: Sequence[int]) -> None:
        """Kill the executors ``executor_ids``, releasing their resources."""
        executors = [self._executors.pop(executor_id) for executor_id in executor_ids]
        for executor_id in executor_ids:
            del self._leases[executor_id]
            self._idle_since.pop(executor_id, None)
        # Killing an executor discards the events it holds.
        await asyncio.gather(*(e.flush_profile.remote() for e in executors))
        for executor in executors:
            ray.kill(executor)
        match executors:
            case []:
                pass
            case _:
                self._recorder.instant("executor_kill", "group", count=len(executors))
                await self._recorder.flush()


class _LeasedExecutors:
    """The executors a map leased from its pool, and its calls on each.

    A call goes to an idle leased executor, else to the least busy one with
    fewer than ``capacity`` calls in flight, else it waits. When no leased
    executor is idle and the map holds fewer than ``replicas``, it leases
    another in the background, one at a time. The lease may wait for an
    executor to start, which may wait for GPUs another pool holds; meanwhile
    the map keeps using the executors it has, and finishing calls free their
    slots, so that the other pool can go on and eventually release its GPUs.

    ``on_idle`` is called whenever the map becomes idle: after its last call
    in flight, and after a lease that arrives when no call needs it anymore.
    """

    def __init__(
        self,
        lease: Callable[[Sequence[int]], Awaitable[Optional[Tuple[int, Any]]]],
        replicas: int,
        capacity: int,
        on_idle: Callable[[], None],
    ) -> None:
        self._lease = lease
        self._on_idle = on_idle
        self._replicas = replicas
        self._capacity = capacity
        self._executors: Dict[int, Any] = {}
        self._in_flight: Dict[int, int] = {}
        self._leasing: Optional[asyncio.Task[None]] = None
        # Error of a failed lease, raised to a call with no executor to use.
        self._failure: Optional[Exception] = None
        # Calls waiting for a slot.
        self._waiting = 0
        self._changed = asyncio.Condition()

    @property
    def ids(self) -> List[int]:
        """The ids of the leased executors."""
        return list(self._executors)

    @property
    def idle(self) -> bool:
        """Whether the map holds leases but has no call in flight or waiting."""
        return (
            bool(self._executors)
            and not any(self._in_flight.values())
            and self._waiting == 0
        )

    def clear(self) -> None:
        """Forget the leases, which the map returns."""
        self._executors.clear()
        self._in_flight.clear()

    def stop(self) -> None:
        """Forget the leases on FLUSH, and the lease in progress."""
        _cancel(self._leasing)
        self._leasing = None
        self.clear()

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncGenerator[Any]:
        """An executor with room for one more call, for the duration of it."""
        executor_id = await self._acquire()
        try:
            yield self._executors[executor_id]
        finally:
            async with self._changed:
                self._in_flight[executor_id] -= 1
                self._changed.notify_all()
            self._report_idle()

    def _report_idle(self) -> None:
        match self.idle:
            case True:
                self._on_idle()
            case False:
                pass

    async def _acquire(self) -> int:
        """An idle executor, else the least busy with room, else wait."""
        async with self._changed:
            self._waiting += 1
            try:
                while True:
                    idle = [i for i, calls in self._in_flight.items() if calls == 0]
                    free = [
                        i
                        for i, calls in self._in_flight.items()
                        if calls < self._capacity
                    ]
                    can_lease = (
                        len(self._executors) < self._replicas and self._leasing is None
                    )
                    match (idle, can_lease):
                        case ([], True):
                            self._leasing = asyncio.create_task(self._grow())
                        case _:
                            pass
                    match (idle, free, self._failure):
                        case ([executor_id, *_], _, _):
                            self._in_flight[executor_id] += 1
                            return executor_id
                        case ([], [_, *_], _):
                            executor_id = min(free, key=self._in_flight.__getitem__)
                            self._in_flight[executor_id] += 1
                            return executor_id
                        case (_, _, Exception() as failure):
                            self._failure = None
                            raise failure
                        case _:
                            await self._changed.wait()
            finally:
                self._waiting -= 1

    async def _grow(self) -> None:
        """Lease another executor, without holding the lock while waiting."""
        try:
            leased = await self._lease(self.ids)
        except Exception as error:
            async with self._changed:
                self._failure = error
                self._leasing = None
                self._changed.notify_all()
        else:
            async with self._changed:
                match leased:
                    case None:
                        # The pool was released while the lease waited.
                        pass
                    case (executor_id, executor):
                        self._executors[executor_id] = executor
                        self._in_flight.setdefault(executor_id, 0)
                self._leasing = None
                self._changed.notify_all()
            self._report_idle()


class _MapRunner:
    """Runs the calls of a map on a route, with the map's compute resources.

    - Calls without GPUs of ephemeral protocols (and of maps without
      resources) are Ray tasks, so Ray queues them on the resources.
    - Calls of long-running protocols, and calls claiming GPUs, run on the
      pool of executors of the map's group. A call goes to an idle executor
      the map leased, else a newly leased one, else the least busy; in each,
      the map's calls run one at a time, as in a stage of its own, while the
      next call is already being transferred to the executor. A map of an
      ephemeral protocol returns its leases once it had no call in flight for
      ``_LEASE_LINGER_S``, so that the pool can release idle executors.
    """

    def __init__(
        self,
        route: _Apply,
        completions: _Completions,
        recorder: Recorder,
    ) -> None:
        self._label = route.label
        self._functions = route.functions
        self._completions = completions
        self._options = route.options
        self._stage = route.stage
        self._group = route.group
        self._returns_idle_leases = route.returns_idle_leases
        self._linger: Optional[asyncio.Task[None]] = None
        self._recorder = recorder
        self.stats = StageStats()
        # The executors stay alive until this map released them on FLUSH.
        self._executors = _LeasedExecutors(
            self._lease,
            replicas=route.replicas,
            capacity=_EXECUTOR_CALLS_IN_FLIGHT,
            on_idle=self._schedule_linger,
        )

    async def run(self, item: Any) -> ray.ObjectRef:
        """Submit a call and wait until its result is in the object store.

        The span of the call covers its queueing for resources and its
        transfer; the worker running it records its execution.
        """
        t_start = time.perf_counter()
        with self._recorder.span(
            "call",
            f"map:{self._label}",
            overlapping=True,
            seq=self.stats.items_in + 1,
        ):
            result = await self._call(item)

        self.stats.items_in += 1
        self.stats.items_out += 1
        self.stats.process_s += time.perf_counter() - t_start
        return result

    async def _call(self, item: Any) -> ray.ObjectRef:
        match self._group:
            case None:
                return await self._completions.done(self._submit(item))
            case _:
                _cancel(self._linger)
                async with self._executors.slot() as executor:
                    return await self._completions.done(
                        executor.call.remote(
                            self._functions,
                            self._label,
                            self._stage,
                            item,
                        ),
                    )

    def _schedule_linger(self) -> None:
        """Return the leases soon, if the map is idle (ephemeral protocols)."""
        match self._returns_idle_leases:
            case True:
                _cancel(self._linger)
                self._linger = asyncio.create_task(self._return_idle_leases())
            case False:
                pass

    async def _return_idle_leases(self) -> None:
        """Return the leases once the map had no call in flight for a moment."""
        await asyncio.sleep(_LEASE_LINGER_S)
        match self._executors.idle:
            case True:
                leases = self._executors.ids
                self._executors.clear()
                assert self._group is not None
                await self._group.return_leases.remote(leases)
            case False:
                pass

    async def release(self) -> None:
        """Release the resources of the map's pool on FLUSH."""
        _cancel(self._linger)
        match self._group:
            case None:
                pass
            case group:
                leases = self._executors.ids
                self._executors.stop()
                await group.return_leases.remote(leases)

    def _submit(self, item: Any) -> ray.ObjectRef:
        return _run_in_task.options(**self._options).remote(
            self._functions,
            self._label,
            item,
        )

    async def _lease(self, held: Sequence[int]) -> Optional[Tuple[int, Any]]:
        assert self._group is not None, "Long-running map has no compute group."
        return await self._group.lease.remote(held)


# Longest time a newly submitted call waits to join the poll of the calls in
# flight.
_POLL_S = 0.01


class _Completions:
    """Waits for the results of Ray calls without fetching them.

    One poll covers every call in flight of a stage, so their number is not
    limited by the threads of the event loop's default executor.
    """

    def __init__(self) -> None:
        self._pending: Dict[ray.ObjectRef, asyncio.Future[None]] = {}
        self._submitted = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def done(self, result: ray.ObjectRef) -> ray.ObjectRef:
        """Wait until ``result`` is in the object store."""
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending[result] = future
        self._submitted.set()
        try:
            await future
        finally:
            # A cancelled call leaves the poll.
            self._pending.pop(result, None)
        return result

    async def _run(self) -> None:
        while True:
            await self._submitted.wait()
            self._submitted.clear()
            while self._pending:
                await self._poll()

    async def _poll(self) -> None:
        ready, _ = await asyncio.to_thread(
            ray.wait,
            list(self._pending),
            num_returns=1,
            timeout=_POLL_S,
            fetch_local=False,
        )
        waiting = [
            self._pending.pop(result) for result in ready if result in self._pending
        ]
        for future in waiting:
            if not future.cancelled():
                future.set_result(None)


@ray.remote
class RaySinkActor(_PipelinedStage):
    """Terminal stage wrapping a SinkWriter, finalized on FLUSH."""

    def __init__(
        self,
        writer: SinkWriter,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
    ):
        super().__init__(config, spill, label)
        self.writer = writer

    async def process(self, item: Any, port: int) -> List[Any]:
        await self.writer.write(item)
        return []

    async def on_flush(self) -> List[Any]:
        await self.writer.finalize()
        return []


class _DemuxResultWriter:
    """Sink of the ``group_by`` child of a key, handing its results back to the router."""

    def __init__(self, router: Any, key: Any) -> None:
        self.router = router
        self.key = key

    async def write(self, item: Any) -> None:
        await self.router.emit.remote(self.key, item)

    async def finalize(self) -> None:
        pass


class _SharedResultWriter:
    """Sink of a ``group_by`` child shared by several keys (see ``share_keys``).

    Its results arrive as ``Keyed(key, result)``.
    """

    def __init__(self, router: Any) -> None:
        self.router = router

    async def write(self, item: Any) -> None:
        assert isinstance(item, Keyed), "A shared group_by child lost the key."
        await self.router.emit.remote(item.key, item.value)

    async def finalize(self) -> None:
        pass


class _Child:
    """Child pipeline of a ``group_by`` key, fed in order once it has started.

    Starting a child waits for its actors to start their processes. The items
    arriving meanwhile wait in its queue, so that the router keeps routing
    the items of the other keys and the children of new keys start in
    parallel.
    """

    def __init__(
        self,
        start: Coroutine[Any, Any, RayCompiledPipeline],
        source_name: str,
        on_error: Callable[[BaseException], None],
        buffer_size: Optional[int],
    ) -> None:
        """
        Args:
            start: Compiles the child pipeline.
            source_name: Input of the child pipeline fed with the items.
            on_error: Called with the first error of the child's start or
                feed.
            buffer_size: Most items queued for the child; ``send`` waits for
                room beyond it. ``None`` queues without limit.
        """
        self.ready: asyncio.Task[RayCompiledPipeline] = asyncio.create_task(start)
        self._source_name = source_name
        self._on_error = on_error
        self._queue: asyncio.Queue[Tuple[Any, Optional[_Barrier]]] = asyncio.Queue(
            maxsize=0 if buffer_size is None else buffer_size,
        )
        self._feeder = asyncio.create_task(self._feed())

    async def send(self, item: Any) -> None:
        """Queue an item for the child, waiting while its queue is full."""
        await self._queue.put((item, None))

    async def flush(self) -> None:
        """Flush the child after all queued items; returns once it has drained."""
        done: _Barrier = asyncio.get_running_loop().create_future()
        await self._queue.put((FLUSH, done))
        await done

    async def _feed(self) -> None:
        error: Optional[BaseException] = None
        while True:
            item, done = await self._queue.get()
            barriers = [] if done is None else [done]
            if error is not None:
                _fail(barriers, error)
                continue

            try:
                child = await self.ready
                await _deliver(child.source_handle(self._source_name), item, 0)
            except Exception as raised:
                error = raised
                self._on_error(raised)
                _fail(barriers, raised)
                continue

            _release(barriers)


@ray.remote
class RayDemuxActor(_PipelinedStage):
    """Router of a ``group_by`` (IRDemux) into child pipelines.

    With ``workers=None``, every key has a child pipeline of its own, compiled
    from a clone of the template when the first item of the key arrives. With
    ``workers=n``, the keys share up to ``n`` children compiled from
    ``share_keys(template)``: every new key is assigned to the next child in
    turn, a child is started with its first key, and the items carry their
    key through it as ``Keyed(key, item)``.

    Nested in a shared child (``outer_keyed``), the items arrive as
    ``Keyed(outer, item)``; the router keys its children by ``(outer, key)``
    and emits ``Keyed(outer, Keyed(key, result))``.

    The sink of a child hands every result back to ``emit``, which forwards it
    as ``Keyed(key, result)``:

    - The router does not wait for a child to start: the items of the key
      wait in the queue of the child meanwhile, and the items of other keys
      are routed on. The children of the keys of a burst start in parallel.
      The queue holds up to ``buffer_size`` items; beyond it, the router
      waits, which propagates backpressure upstream.
    - The compute task queues items for a child, whose sink waits for room in
      the outbox of the router. The outbox is drained by the emitter task,
      independently of the compute task, so the two never wait on each other.
    - On FLUSH, the router flushes every child, after the items queued for
      it, before it forwards the FLUSH. A child returns from its flush only
      after its sink handed back all results, so they precede the FLUSH
      downstream.
    - The children abort the abort targets of the router, so that a failing
      child stops the enclosing pipeline.

    The children are owned by the router and terminated together with it.
    """

    def __init__(
        self,
        key_fn: Callable[[Any], Any],
        template: IROp,
        source_name: str,
        max_keys: Optional[int],
        workers: Optional[int],
        outer_keyed: bool,
        group_by_workers: Optional[int],
        prefix: str,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        parameters: Mapping[str, Any],
        ephemeral_idle_s: float,
        pools: Mapping[str, "_Pool"],
        gpu_memory: Optional[Tuple[float, ...]] = None,
    ):
        """
        Args:
            workers: Children shared by the keys; ``None`` starts a child per
                key.
            outer_keyed: Whether the items arrive as ``Keyed(outer, item)``.
            group_by_workers: Setting of the backend, passed to the children.
        """
        from .container import configure_ray_env

        configure_ray_env(parameters=parameters)
        super().__init__(config, spill, label)
        self.gpu_memory = gpu_memory
        self.key_fn = key_fn
        self.template = template
        self.source_name = source_name
        self.max_keys = max_keys
        self.workers = workers
        self.outer_keyed = outer_keyed
        self.group_by_workers = group_by_workers
        self.ephemeral_idle_s = ephemeral_idle_s
        self.pools = pools
        self.prefix = prefix
        self.config = config
        self.parameters = parameters
        # Children by key, or by index of a shared child.
        self._children: Dict[Any, _Child] = {}
        # Child of every key.
        self._keys: Dict[Any, _Child] = {}

    async def process(self, item: Any, port: int) -> List[Any]:
        match self.outer_keyed:
            case True:
                outer, value = item
                key = (outer, await asyncio.to_thread(self.key_fn, value))
            case False:
                value = item
                key = await asyncio.to_thread(self.key_fn, value)

        if key not in self._keys:
            self._keys[key] = self._assign(key)

        match self.workers:
            case None:
                await self._keys[key].send(value)
            case _:
                await self._keys[key].send(Keyed(key=key, value=value))
        return []

    async def on_flush(self) -> List[Any]:
        await asyncio.gather(*(child.flush() for child in self._children.values()))
        return []

    async def emit(self, key: Any, item: Any) -> None:
        """Forward a result of the child pipeline of ``key``."""
        match self.outer_keyed:
            case True:
                outer, inner = key
                keyed = Keyed(key=outer, value=Keyed(key=inner, value=item))
            case False:
                keyed = Keyed(key=key, value=item)
        await self._outbox.put((keyed, []))

    async def children_stats(self) -> Dict[str, StageStats]:
        """Return the execution metrics of every stage of the started children."""
        stats = await asyncio.gather(
            *(asyncio.to_thread(child.stats) for child in self._started()),
        )
        return {label: stage for child in stats for label, stage in child.items()}

    async def flush_profile(self) -> None:
        """Wait until the collector wrote the events of the router and its children."""
        await asyncio.gather(
            *(asyncio.to_thread(child.flush_profile) for child in self._started()),
        )
        await super().flush_profile()

    def _started(self) -> List[RayCompiledPipeline]:
        """The child pipelines that have started."""
        return [
            child.ready.result()
            for child in self._children.values()
            if child.ready.done()
        ]

    async def _record_start(
        self,
        child: str,
        start: Coroutine[Any, Any, RayCompiledPipeline],
    ) -> RayCompiledPipeline:
        """Start a child, recording how long it takes."""
        with self._recorder.span(
            "start_child",
            "group_by",
            level=logging.INFO,
            overlapping=True,
            child=child,
        ):
            return await start

    def _assign(self, key: Any) -> _Child:
        """The child of a new key, started if necessary."""
        if self.max_keys is not None and len(self._keys) >= self.max_keys:
            raise ValueError(
                f"group_by received key {key!r} after max_keys={self.max_keys} "
                f"keys: {list(self._keys)}.",
            )

        router = ray.get_runtime_context().current_actor
        match self.workers:
            case None:
                (exit_,) = clone_ir([self.template])
                self._children[key] = self._start(
                    exit_,
                    _DemuxResultWriter(router, key),
                    f"{key}",
                )
                return self._children[key]
            case workers:
                worker = len(self._keys) % workers
                if worker not in self._children:
                    self._children[worker] = self._start(
                        share_keys(self.template),
                        _SharedResultWriter(router),
                        f"worker {worker}",
                    )
                return self._children[worker]

    def _start(self, exit_: IROp, writer: SinkWriter, child: str) -> _Child:
        """Start the child pipeline ending in ``exit_``, labelled ``child``."""
        sink = IRSink(writer=writer)
        exit_.add_downstream(sink)
        spill = self.config.spill
        backend = RayBackend(
            init_ray=False,
            queue_size=self.config.queue_size,
            max_in_flight=self.config.max_in_flight,
            buffer_size=self.config.buffer_size,
            spill_threshold=None if spill is None else spill.threshold,
            spill_store=None if spill is None else spill.store,
            parameters=self.parameters,
            gpu_memory=self.gpu_memory,
            profile=self.config.profile,
            group_by_workers=self.group_by_workers,
            ephemeral_idle_s=self.ephemeral_idle_s,
            pools=self.pools,
        )
        start = backend.compile_async(
            [sink],
            name=f"{self.prefix}group_by[{child}]:",
            abort_targets=self._abort_targets,
        )
        return _Child(
            self._record_start(child, start),
            self.source_name,
            self._set_error,
            self.config.buffer_size,
        )


class RayCompiledPipeline(CompiledPipeline):
    """
    Executable compiled streaming pipeline running on Ray.
    """

    def __init__(
        self,
        sources: Dict[str, Any],
        stages: Dict[str, Tuple[IROp, Any]],
        demuxes: Sequence[Any] = (),
        groups: Sequence[Any] = (),
        spill_root: Optional[Path] = None,
        profile: Optional[ProfileConfig] = None,
        owns_profile: bool = False,
        functions: Optional[ray.ObjectRef] = None,
    ):
        """
        Args:
            sources: Source actors keyed by input name.
            stages: All stages of the pipeline keyed by a readable label, in
                topological order, each with the node and actor hosting it:
                its own actor, or for a map the actor submitting its calls.
            demuxes: The ``group_by`` routers among the stages, whose child
                pipelines are reported by ``stats``.
            groups: Coordinators of the pools the pipeline owns: it releases
                them after every flush, and their executors terminate with
                them on ``close``.
            spill_root: Temporary directory of the spilled items, owned by
                the pipeline and removed on ``close``.
            profile: Profiling of the pipeline, if it is profiled.
            owns_profile: Whether the pipeline completes the trace on
                ``close``; a nested pipeline writes into the trace of its
                parent.
            functions: The function table of the pipeline's maps, kept in
                the object store while the pipeline runs.
        """
        self._sources = sources
        self._stages = stages
        self._demuxes = list(demuxes)
        self._groups = list(groups)
        self._spill_root = spill_root
        self._profile = profile
        self._owns_profile = owns_profile
        self._functions = functions
        self._recorder = make_recorder(profile, "driver")

    @property
    def profile(self) -> Optional[ProfileConfig]:
        """Profiling of the pipeline, for processes that record into its trace."""
        return self._profile

    def source_handle(self, source_name: str) -> Any:
        """Return the actor handle of a named input source.

        Lets an async actor push into the pipeline without blocking its event
        loop: ``await pipeline.source_handle(name).push.remote(item)``.
        """
        if source_name not in self._sources:
            raise KeyError(
                f"Input source '{source_name}' is not registered in this pipeline. Available: {list(self._sources.keys())}",
            )
        return self._sources[source_name]

    def send(self, source_name: str, value: Any) -> None:
        """Push an item into a named input source.

        Returns once the source has buffered the item. While the buffer of
        the source is full, because the stages downstream of it are, it waits
        for room.
        """
        handle = self.source_handle(source_name)
        with self._recorder.span("send", "driver", source=source_name):
            ray.get(handle.push.remote(_transport(value)))

    def flush(self) -> None:
        """Flush all sources in parallel and wait for pipeline completion.

        Then release the pools of executors: FLUSH has passed every stage,
        the children of ``group_by`` included, so no calls are left.
        """
        with self._recorder.span("flush", "driver", level=logging.INFO):
            ray.get([src.push.remote(FLUSH) for src in self._sources.values()])
        ray.get([group.release_all.remote() for group in self._groups])
        self._recorder.flush_sync()

    def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of every process.

        A stage sends its events at every FLUSH; this sends the events since,
        e.g. of a failed pipeline.
        """
        hosts = {node: actor for node, actor in self._stages.values()}
        ray.get(
            [
                actor.flush_profile.remote()
                for actor in [*hosts.values(), *self._groups]
            ],
        )
        self._recorder.flush_sync()

    def stats(self) -> Dict[str, StageStats]:
        """Return execution metrics per stage, in topological order.

        The stages of ``group_by`` children follow, labelled with their key.
        """
        hosts = {node: actor for node, actor in self._stages.values()}
        stage_stats = dict(
            zip(hosts, ray.get([actor.stats.remote() for actor in hosts.values()])),
        )
        map_stats = {
            label: stage
            for maps in ray.get([actor.map_stats.remote() for actor in hosts.values()])
            for label, stage in maps.items()
        }
        children = ray.get([demux.children_stats.remote() for demux in self._demuxes])
        return {
            **{
                label: map_stats[label] if label in map_stats else stage_stats[host]
                for label, (host, _) in self._stages.items()
            },
            **{label: stage for child in children for label, stage in child.items()},
        }

    def close(self) -> None:
        """Terminate all actors allocated for this pipeline.

        The children of ``group_by`` routers terminate with their router, and
        the executors of compute groups with their coordinator. A profiled
        pipeline completes its trace first.
        """
        match (self._profile, self._owns_profile):
            case (ProfileConfig() as profile, True):
                self._close_profile(profile)
            case _:
                pass

        hosts = {node: actor for node, actor in self._stages.values()}
        for actor in [*hosts.values(), *self._groups]:
            try:
                ray.kill(actor)
            except Exception:
                pass

        match self._spill_root:
            case None:
                pass
            case root:
                # ``close`` may run more than once.
                shutil.rmtree(root, ignore_errors=True)

        # Without a reference, Ray frees the function table.
        self._functions = None

    def _close_profile(self, profile: ProfileConfig) -> None:
        """Write the remaining events and complete the trace file."""
        self.flush_profile()
        ray.get(profile.collector.close.remote())
        ray.kill(profile.collector)
        # ``close`` may run more than once, after the actors were killed.
        self._owns_profile = False


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
        """
        if queue_size <= 0:
            raise ValueError(f"Queue size must be positive, got {queue_size}.")
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
        """
        pipeline, connected = self._instantiate(
            ir_sinks,
            name=name,
            abort_targets=abort_targets,
        )
        ray.get(connected)
        return pipeline

    async def compile_async(
        self,
        ir_sinks: List[IROp],
        *,
        name: str = "",
        abort_targets: Sequence[Any] = (),
    ) -> RayCompiledPipeline:
        """Compile like ``compile``, without blocking the event loop.

        Starting the actors of a pipeline takes seconds; an async actor (the
        router of a ``group_by``) awaits it while it keeps working, and
        compiles several pipelines at once.
        """
        pipeline, connected = await asyncio.to_thread(
            self._instantiate,
            ir_sinks,
            name=name,
            abort_targets=abort_targets,
        )
        await asyncio.gather(*connected)
        return pipeline

    def _instantiate(
        self,
        ir_sinks: List[IROp],
        *,
        name: str,
        abort_targets: Sequence[Any],
    ) -> Tuple[RayCompiledPipeline, List[ray.ObjectRef]]:
        """Create the actors of a pipeline and start connecting them.

        Returns:
            The pipeline, and the calls connecting its stages; it is ready
            once they are done.
        """
        if not ir_sinks:
            raise ValueError("RayBackend.compile() requires at least one IRSink.")

        # 1. Discover all reachable IR nodes in topological order (upstream first)
        visited: Set[IROp] = set()
        all_nodes: List[IROp] = []
        sources_map: Dict[str, IRSource] = {}

        def _traverse(node: IROp) -> None:
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, IRSource):
                sources_map[node.name] = node
            for up in node.upstream:
                _traverse(up)
            all_nodes.append(node)

        for sink in ir_sinks:
            _traverse(sink)

        # 2. Instantiate a Ray actor for every IR node but the maps, which run
        # as Ray tasks submitted by the stage upstream of them. Actors are
        # named so that stages can be identified in the Ray dashboard; the
        # pipeline id keeps the names unique within the Ray namespace.
        pipeline_id = uuid.uuid4().hex[:8]
        labels = {
            node: f"{name}{index}:{_describe(node)}"
            for index, node in enumerate(all_nodes)
        }
        # The maps of the children of group_by share the pools of this
        # pipeline, so the claims cover them too.
        maps = _all_maps(all_nodes)
        claims = {node.compute: self._claim(node.compute) for node in maps}
        profile, owns_profile = self._profile_config(pipeline_id)
        # Map calls and the warm-up of executors find their functions in the
        # function table of the pipeline, put in the object store once.
        functions = ray.put(
            FunctionTable(
                functions={
                    labels[node]: node.func
                    for node in all_nodes
                    if isinstance(node, IRMap)
                },
                parameters=self.parameters,
                profile=profile,
                preparers={
                    compute.group: compute.prepare
                    for node in maps
                    if isinstance(compute := node.compute, ComputeAssignment)
                    and compute.prepare is not None
                },
            ),
        )
        owns_pools = self.pools is None
        pools = (
            self._compute_groups(maps, claims, Ref(functions), profile)
            if self.pools is None
            else self.pools
        )
        spill, spill_root = self._spill_config()
        # Every executor of a pool takes calls of a map at once; the items on
        # the routes of a stage must cover the replicas a protocol declared.
        pooled_calls = max(
            (
                replicas * _EXECUTOR_CALLS_IN_FLIGHT
                for node in all_nodes
                if isinstance(node, IRMap)
                and isinstance(compute := node.compute, ComputeAssignment)
                and (replicas := _declared_replicas(compute.resources)) > 1
            ),
            default=0,
        )
        config = _StageConfig(
            queue_size=self.queue_size,
            max_in_flight=max(self.max_in_flight, pooled_calls),
            buffer_size=self.buffer_size,
            spill=spill,
            profile=profile,
        )
        actor_map: Dict[IROp, Any] = {
            node: actor_cls.options(
                name=f"{pipeline_id}:{labels[node]}",
                runtime_env={"env_vars": numpy_hugepage_env()},
            ).remote(**kwargs)
            for node in all_nodes
            if not _is_hosted(node)
            for actor_cls, kwargs in [
                self._stage_actor(node, name, labels[node], config, pools),
            ]
        }

        compiled_sources = {
            source_name: actor_map[src_node]
            for source_name, src_node in sources_map.items()
        }

        # 3. Wire the routes and source actor handles and start the stage
        # loops. Every stage feeds a downstream stage on the port given by its
        # position among the upstream stages of the downstream stage; maps in
        # between are applied on the way. A failing stage aborts the sources
        # and the further abort targets.
        builder = _RouteBuilder(actor_map, labels, pools, claims, Ref(functions))
        abort_handles = [*compiled_sources.values(), *abort_targets]
        connected = [
            actor.connect.remote(
                builder.stage_routes(node),
                abort_handles,
                max(1, len(node.upstream)),
            )
            for node, actor in actor_map.items()
        ]

        pipeline = RayCompiledPipeline(
            sources=compiled_sources,
            stages={
                labels[node]: (_host_of(node), actor_map[_host_of(node)])
                for node in all_nodes
            },
            demuxes=[
                actor_map[node] for node in all_nodes if isinstance(node, IRDemux)
            ],
            # The pools of children of a group_by belong to their parent.
            groups=(
                [pool.coordinator for pool in pools.values()] if owns_pools else []
            ),
            spill_root=spill_root,
            profile=profile,
            owns_profile=owns_profile,
            functions=functions,
        )
        return pipeline, connected

    def _profile_config(
        self,
        pipeline_id: str,
    ) -> Tuple[Optional[ProfileConfig], bool]:
        """Profiling of a compiled pipeline, and whether the pipeline owns it."""
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
        match (self.spill_threshold, self.spill_store):
            case (None, _):
                return None, None
            case (threshold, None):
                # Created by the first spilled item: most pipelines never spill.
                root = Path(tempfile.gettempdir()) / (
                    f"scipion_bridge_spill_{uuid.uuid4().hex}"
                )
                return _SpillConfig(threshold, pickle_spill_store(root)), root
            case (threshold, store):
                return _SpillConfig(threshold, store), None

    def _stage_actor(
        self,
        node: IROp,
        name: str,
        label: str,
        config: _StageConfig,
        pools: Mapping[str, "_Pool"],
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
                    "group_by_workers": self.group_by_workers,
                    "ephemeral_idle_s": self.ephemeral_idle_s,
                    "pools": pools,
                    "prefix": name,
                    "config": config,
                    "spill": spill,
                    "label": label,
                    "parameters": self.parameters,
                    "gpu_memory": (
                        self._cluster_gpu_memory()
                        if _needs_gpu_memory(template)
                        else self._gpu_memory
                    ),
                }
            case _:
                raise NotImplementedError(
                    f"Unsupported IR op type for Ray backend: {type(node).__name__}",
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
        match compute:
            case None:
                return _ray_options(ComputeResources(), 0)
            case ComputeAssignment(resources=resources):
                num_gpus = self._num_gpus(resources)
                _check_cluster_fits(compute, num_gpus)
                return _ray_options(resources, num_gpus)

    def _num_gpus(self, resources: ComputeResources) -> float:
        match resources.min_vram:
            case None:
                return resources.gpus
            case _:
                return _resolve_gpus(resources, self._cluster_gpu_memory())

    def _cluster_gpu_memory(self) -> Tuple[float, ...]:
        if self._gpu_memory is None:
            self._gpu_memory = _probe_cluster_gpu_memory()
        return self._gpu_memory

    def _compute_groups(
        self,
        nodes: Sequence[IRMap],
        claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]],
        functions: Ref,
        profile: Optional[ProfileConfig],
    ) -> Dict[str, "_Pool"]:
        """Create the pool of every group of ``nodes`` that runs on executors.

        Long-running pools start, preparing their executors, while the rest
        of the pipeline is created. Ephemeral pools start executors on
        demand; by default they share the GPUs the long-running pools leave
        in equal parts.

        Raises:
            ValueError: If the long-running pools do not fit the cluster at
                once.
        """
        # The maps of a group, once each: their copies in the children of a
        # group_by keep the fingerprint.
        stages: Dict[str, Set[Optional[str]]] = {}
        claims_of: Dict[str, Tuple[Dict[str, Any], ComputeResources]] = {}
        for node in nodes:
            match node:
                case IRMap(
                    compute=ComputeAssignment(
                        group=group, resources=resources
                    ) as compute,
                ) if _pooled(compute, claims[compute]):
                    stages.setdefault(group, set()).add(node.stage_fp)
                    claims_of[group] = (claims[compute], resources)
                case _:
                    pass

        cluster = ray.cluster_resources()
        long_running = {
            group: (options, _declared_replicas(resources))
            for group, (options, resources) in claims_of.items()
            if resources.task == TaskType.LONG_RUNNING
        }
        _check_pools_fit(long_running, cluster)
        held_gpus = sum(
            options["num_gpus"] * replicas
            for options, replicas in long_running.values()
        )
        pools = {
            group: _Pool(
                coordinator=RayComputeGroup.options(
                    runtime_env={"env_vars": numpy_hugepage_env()},
                ).remote(
                    options,
                    len(stages[group]),
                    replicas,
                    functions,
                    self.parameters,
                    group,
                    profile,
                    idle_s=None if group in long_running else self.ephemeral_idle_s,
                ),
                replicas=replicas,
                ephemeral=group not in long_running,
            )
            for group, (options, resources) in claims_of.items()
            for replicas in [
                _pool_replicas(
                    options,
                    resources,
                    cluster.get("GPU", 0),
                    held_gpus,
                    ephemeral_pools=len(claims_of) - len(long_running),
                ),
            ]
        }
        # A failed warm-up surfaces on the first lease, which prepares again.
        for group in long_running:
            pools[group].coordinator.start.remote()
        return pools


def _all_maps(nodes: Sequence[IROp]) -> List[IRMap]:
    """The maps among ``nodes`` and in the children of their ``group_by`` nodes."""
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


@dataclass(frozen=True)
class _Pool:
    """The pool of executors of a compute group."""

    coordinator: Any
    replicas: int
    # Whether the pool releases idle executors (ephemeral protocols).
    ephemeral: bool


def _pooled(
    compute: Optional[ComputeAssignment],
    options: Mapping[str, Any],
) -> bool:
    """Whether the calls of a map run on the executors of a pool.

    Long-running protocols hold their executors; calls claiming GPUs need
    processes that keep their GPU. The ``cpu_only`` maps of a protocol share
    its group id, but claim no GPUs and run as tasks.
    """
    match compute:
        case None:
            return False
        case ComputeAssignment(
            resources=ComputeResources(task=TaskType.LONG_RUNNING),
        ):
            return True
        case ComputeAssignment():
            return options["num_gpus"] > 0


def _declared_replicas(resources: ComputeResources) -> int:
    """The replicas a protocol declared, or one."""
    match resources.replicas:
        case None:
            return 1
        case replicas:
            return replicas


def _pool_replicas(
    options: Mapping[str, Any],
    resources: ComputeResources,
    cluster_gpus: float,
    held_gpus: float,
    ephemeral_pools: int,
) -> int:
    """Executors of a pool: declared, or the default of its task type.

    The ephemeral pools of a pipeline share the GPUs that the long-running
    pools (``held_gpus``) leave, in equal parts, so that they do not compete
    for the same GPUs; each may start as many executors as its part fits, at
    least one.
    """
    match (resources.task, resources.replicas):
        case (_, int(replicas)):
            return replicas
        case (TaskType.LONG_RUNNING, None):
            return 1
        case _:
            share = (cluster_gpus - held_gpus) / ephemeral_pools
            return max(1, math.floor(share / options["num_gpus"]))


def _without_prepare(
    compute: Optional[ComputeAssignment],
) -> Optional[ComputeAssignment]:
    """The claim without its warm-up, which holds the protocol.

    Routes are sent to every stage; the warm-up reaches the executors through
    the function table instead.
    """
    match compute:
        case None:
            return None
        case ComputeAssignment():
            return dataclasses.replace(compute, prepare=None)


def _stage_of(node: IRMap, label: str) -> str:
    """The fingerprint of a map of a protocol, or its label outside of protocols.

    Only maps of protocols run on executors, which lock by it.
    """
    match node.stage_fp:
        case None:
            return label
        case stage_fp:
            return stage_fp


def _pool_of(
    pools: Mapping[str, _Pool],
    compute: Optional[ComputeAssignment],
    options: Mapping[str, Any],
) -> Optional[_Pool]:
    """The pool running the calls of a map, if they run on executors."""
    match (compute, _pooled(compute, options)):
        case (ComputeAssignment(group=group), True):
            return pools[group]
        case _:
            return None


@dataclass
class _RouteBuilder:
    """Builds the routes of the stages with actors, through the hosted maps."""

    actor_map: Mapping[IROp, Any]
    labels: Mapping[IROp, str]
    pools: Mapping[str, _Pool]
    claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]]
    functions: Ref

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
        match _pool_of(self.pools, node.compute, options):
            case None:
                group, replicas, returns_idle_leases = None, 1, False
            case _Pool() as pool:
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
            case IRMap() if _is_hosted(downstream):
                return self._apply(downstream)
            case _:
                return _Push(
                    self.actor_map[downstream],
                    downstream.upstream.index(node),
                )


def _is_hosted(node: IROp) -> bool:
    """Whether ``node`` runs on the actor of another stage (a single-input map)."""
    match node:
        case IRMap(upstream=[_]):
            return True
        case _:
            return False


def _host_of(node: IROp) -> IROp:
    """The stage whose actor runs ``node``.

    A map with a single input runs on the actor of the nearest upstream
    stage; a map merging several inputs has an actor of its own.
    """
    match node:
        case IRMap(upstream=[upstream]):
            return _host_of(upstream)
        case _:
            return node


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


def _check_pools_fit(
    pools: Mapping[str, Tuple[Dict[str, Any], int]],
    cluster: Mapping[str, float],
) -> None:
    """Raise unless the long-running pools fit the cluster at once.

    Args:
        pools: The Ray options of an executor and the replicas, by group.
        cluster: The resources of the Ray cluster.

    The executors of a pool hold their resources until FLUSH: a pool that
    fits only once another released its GPUs would wait for them forever.
    """
    gpus = sum(options["num_gpus"] * replicas for options, replicas in pools.values())
    cpus = sum(options["num_cpus"] * replicas for options, replicas in pools.values())
    cluster_gpus, cluster_cpus = cluster.get("GPU", 0), cluster.get("CPU", 0)
    match (gpus <= cluster_gpus, cpus <= cluster_cpus):
        case (True, True):
            pass
        case _:
            raise ValueError(
                f"The long-running protocols {sorted(pools)} hold {gpus:g} GPUs "
                f"and {cpus:g} CPUs at once, but the Ray cluster has "
                f"{cluster_gpus:g} GPUs and {cluster_cpus:g} CPUs. Reduce "
                "their replicas or their claims.",
            )


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


def _describe(node: IROp) -> str:
    """Readable label of an IR node for metrics output."""
    match node:
        case IRSource(name=name):
            return f"source({name})"
        case IRMap(func=func, name=name):
            return name or f"map({getattr(func, '__qualname__', type(func).__name__)})"
        case IRAccumulate(name=name):
            return name
        case IRSink(writer=writer):
            return f"sink({type(writer).__name__})"
        case IRDemux(name=name):
            return name
        case _:
            return type(node).__name__
