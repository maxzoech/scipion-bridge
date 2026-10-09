from __future__ import annotations
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import (
    Any,
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
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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
    IROp,
    IRSource,
    IRMap,
    IRAccumulate,
    IRSink,
    IRDemux,
    Keyed,
    Tagged,
    clone_ir,
)
from scipion_bridge.core.streaming.node import FlushSignal, FLUSH
from scipion_bridge.core.streaming.sink_writer import SinkWriter
from scipion_bridge.core.streaming.spill import SpillStoreFactory, pickle_spill_store

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
    func: Callable[[Any], Any]
    compute: Optional[ComputeAssignment]
    # Ray options of the task of every call (resources, runtime env).
    options: Dict[str, Any]
    # Coordinator of the executor of a long-running compute group.
    group: Optional[Any]
    parameters: Mapping[str, Any]
    profile: Optional[ProfileConfig]
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


def _cancel(routing: Optional[asyncio.Task[List[_Delivery]]]) -> None:
    match routing:
        case None:
            pass
        case _:
            routing.cancel()


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


def _run_map_call(
    func: Callable[[Any], Any],
    parameters: Mapping[str, Any],
    label: str,
    profile: Optional[ProfileConfig],
    item: Any,
) -> Any:
    """Run one call of a map ``label``, holding the resources of the task.

    The worker process runs other calls afterwards or exits (see
    ``_run_in_gpu_task``), so the profiling events of the call are sent
    before it returns.
    """
    _serve_parameters(parameters)
    recorder = make_recorder(profile, "task worker")
    try:
        with recorder.span("execute", f"map:{label}", gpus=ray.get_gpu_ids()):
            return func(item)
    finally:
        recorder.flush_sync()


# Ray reuses the worker processes of CPU calls.
_run_in_task = ray.remote(_run_map_call)

# A GPU call runs in a process of its own. Ray sets CUDA_VISIBLE_DEVICES for
# every task, but CUDA (and JAX) binds a process to its GPU once: a reused
# worker would keep running on the GPU of its first call, holding the memory
# it preallocated there. Ray only disables reuse by itself for GPUs claimed
# in the decorator, not through ``.options()``.
_run_in_gpu_task = ray.remote(max_calls=1)(_run_map_call)


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
        _node_gpu_memory.options(
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
        self._locks: Dict[str, threading.Lock] = {}
        self._recorder = make_recorder(profile, f"executor({group})")
        self._recorder.instant("start", "executor", gpus=ray.get_gpu_ids())

    def call(self, label: str, func: Callable[[Any], Any], item: Any) -> Any:
        """Run ``func`` of the map ``label`` on ``item``."""
        lane = f"map:{label}"
        lock = self._locks.setdefault(label, threading.Lock())
        with self._recorder.span("lock_wait", lane, overlapping=True):
            lock.acquire()
        try:
            with self._recorder.span("execute", lane):
                return func(item)
        finally:
            lock.release()

    def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of the executor."""
        self._recorder.flush_sync()


@ray.remote
class RayComputeGroup:
    """Coordinator of the executor of a long-running group of maps.

    The executor is created on the first ``acquire`` and killed, releasing
    its resources, once every map of the group has passed FLUSH. Items
    arriving after a flush create a new executor.
    """

    def __init__(
        self,
        options: Dict[str, Any],
        members: int,
        parameters: Mapping[str, Any],
        group: str,
        profile: Optional[ProfileConfig],
    ) -> None:
        self._options = options
        self._members = members
        self._parameters = parameters
        self._group = group
        self._profile = profile
        self._recorder = make_recorder(profile, f"group({group})")
        self._executor: Optional[Any] = None
        self._released = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Any:
        """Return the executor of the group, creating it if necessary."""
        async with self._lock:
            if self._executor is None:
                self._recorder.instant("executor_create", "group")
                self._executor = RayComputeExecutor.options(
                    max_concurrency=self._members * _EXECUTOR_CALLS_IN_FLIGHT,
                    **self._options,
                ).remote(self._parameters, self._group, self._profile)
            return self._executor

    async def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of the group."""
        await self._recorder.flush()

    async def release(self) -> None:
        """Mark a map as flushed; kill the executor once all are flushed."""
        async with self._lock:
            self._released += 1
            match (self._released < self._members, self._executor):
                case (True, _):
                    pass
                case (False, None):
                    self._released = 0
                case (False, executor):
                    self._released = 0
                    self._executor = None
                    # Killing the executor discards the events it holds.
                    await executor.flush_profile.remote()
                    ray.kill(executor)
                    self._recorder.instant("executor_kill", "group")
                    await self._recorder.flush()


class _MapRunner:
    """Runs the calls of a map on a route, with the map's compute resources.

    - Ephemeral resources (and maps without resources) are taken by a Ray
      task per call, so Ray queues the calls of all maps on the resources. A
      call claiming GPUs runs in a process of its own (see
      ``_run_in_gpu_task``).
    - Long-running resources are held by the executor of the map's group.
      Its calls run one at a time, as in a stage of its own, while the next
      call is already being transferred to the executor.
    """

    def __init__(
        self,
        route: _Apply,
        completions: _Completions,
        recorder: Recorder,
    ) -> None:
        self._label = route.label
        self._func = route.func
        self._completions = completions
        self._compute = route.compute
        self._options = route.options
        match route.options["num_gpus"]:
            case num_gpus if num_gpus > 0:
                self._task = _run_in_gpu_task
            case _:
                self._task = _run_in_task
        self._group = route.group
        self._parameters = route.parameters
        self._profile = route.profile
        self._recorder = recorder
        self.stats = StageStats()
        self._executor_calls = asyncio.Semaphore(_EXECUTOR_CALLS_IN_FLIGHT)
        # The executor stays alive until this map released it on FLUSH.
        self._executor: Optional[Any] = None

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
        match self._compute:
            case ComputeAssignment(
                resources=ComputeResources(task=TaskType.LONG_RUNNING),
            ):
                async with self._executor_calls:
                    executor = await self._acquire()
                    return await self._completions.done(
                        executor.call.remote(self._label, self._func, item),
                    )
            case _:
                return await self._completions.done(self._submit(item))

    async def release(self) -> None:
        """Release the long-running resources of the map on FLUSH."""
        match self._group:
            case None:
                pass
            case group:
                self._executor = None
                await group.release.remote()

    def _submit(self, item: Any) -> ray.ObjectRef:
        return self._task.options(**self._options).remote(
            self._func,
            self._parameters,
            self._label,
            self._profile,
            item,
        )

    async def _acquire(self) -> Any:
        assert self._group is not None, "Long-running map has no compute group."
        if self._executor is None:
            self._executor = await self._group.acquire.remote()
        return self._executor


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
    """Sink of a ``group_by`` child, handing its results back to the router."""

    def __init__(self, router: Any, key: Any) -> None:
        self.router = router
        self.key = key

    async def write(self, item: Any) -> None:
        await self.router.emit.remote(self.key, item)

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
    """Router of a ``group_by`` (IRDemux) into a child pipeline per key.

    The child pipeline of a key is compiled from a clone of the template when
    the first item of the key arrives. Its sink hands every result back to
    ``emit``, which forwards it as ``Keyed(key, result)``:

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
        prefix: str,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        parameters: Mapping[str, Any],
        gpu_memory: Optional[Tuple[float, ...]] = None,
    ):
        from .container import configure_ray_env

        configure_ray_env(parameters=parameters)
        super().__init__(config, spill, label)
        self.gpu_memory = gpu_memory
        self.key_fn = key_fn
        self.template = template
        self.source_name = source_name
        self.max_keys = max_keys
        self.prefix = prefix
        self.config = config
        self.parameters = parameters
        self._children: Dict[Any, _Child] = {}

    async def process(self, item: Any, port: int) -> List[Any]:
        key = await asyncio.to_thread(self.key_fn, item)
        if key not in self._children:
            self._children[key] = self._start(key)

        await self._children[key].send(item)
        return []

    async def on_flush(self) -> List[Any]:
        await asyncio.gather(*(child.flush() for child in self._children.values()))
        return []

    async def emit(self, key: Any, item: Any) -> None:
        """Forward a result of the child pipeline of ``key``."""
        await self._outbox.put((Keyed(key=key, value=item), []))

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
        key: Any,
        start: Coroutine[Any, Any, RayCompiledPipeline],
    ) -> RayCompiledPipeline:
        """Start the child of ``key``, recording how long it takes."""
        with self._recorder.span(
            "start_child",
            "group_by",
            level=logging.INFO,
            overlapping=True,
            key=key,
        ):
            return await start

    def _start(self, key: Any) -> _Child:
        if self.max_keys is not None and len(self._children) >= self.max_keys:
            raise ValueError(
                f"group_by received key {key!r} after max_keys={self.max_keys} "
                f"keys: {list(self._children)}.",
            )

        (exit_,) = clone_ir([self.template])
        sink = IRSink(
            writer=_DemuxResultWriter(ray.get_runtime_context().current_actor, key),
        )
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
        )
        start = backend.compile_async(
            [sink],
            name=f"{self.prefix}group_by[{key}]:",
            abort_targets=self._abort_targets,
        )
        return _Child(
            self._record_start(key, start),
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
    ):
        """
        Args:
            sources: Source actors keyed by input name.
            stages: All stages of the pipeline keyed by a readable label, in
                topological order, each with the node and actor hosting it:
                its own actor, or for a map the actor submitting its calls.
            demuxes: The ``group_by`` routers among the stages, whose child
                pipelines are reported by ``stats``.
            groups: Coordinators of the long-running compute groups, whose
                executors terminate with them.
            spill_root: Temporary directory of the spilled items, owned by
                the pipeline and removed on ``close``.
            profile: Profiling of the pipeline, if it is profiled.
            owns_profile: Whether the pipeline completes the trace on
                ``close``; a nested pipeline writes into the trace of its
                parent.
        """
        self._sources = sources
        self._stages = stages
        self._demuxes = list(demuxes)
        self._groups = list(groups)
        self._spill_root = spill_root
        self._profile = profile
        self._owns_profile = owns_profile
        self._recorder = make_recorder(profile, "driver")

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
        """Flush all sources in parallel and wait for pipeline completion."""
        with self._recorder.span("flush", "driver", level=logging.INFO):
            ray.get([src.push.remote(FLUSH) for src in self._sources.values()])
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
        """
        if queue_size <= 0:
            raise ValueError(f"Queue size must be positive, got {queue_size}.")
        for option, value in {
            "max_in_flight": max_in_flight,
            "buffer_size": buffer_size,
            "spill_threshold": spill_threshold,
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

            context = ray.init(
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
        claims = {
            node.compute: self._claim(node.compute)
            for node in all_nodes
            if isinstance(node, IRMap)
        }
        profile, owns_profile = self._profile_config(pipeline_id)
        groups = self._compute_groups(all_nodes, claims, profile)
        spill, spill_root = self._spill_config()
        config = _StageConfig(
            queue_size=self.queue_size,
            max_in_flight=self.max_in_flight,
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
                self._stage_actor(node, name, labels[node], config),
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
        builder = _RouteBuilder(
            actor_map,
            labels,
            groups,
            claims,
            self.parameters,
            profile,
        )
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
            groups=list(groups.values()),
            spill_root=spill_root,
            profile=profile,
            owns_profile=owns_profile,
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
    ) -> Tuple[Any, Dict[str, Any]]:
        """The actor class of a stage, with its constructor arguments."""
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
            ):
                assert template is not None, "IRDemux has no template."
                return RayDemuxActor, {
                    "key_fn": key_fn,
                    "template": template,
                    "source_name": template_source,
                    "max_keys": max_keys,
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
        nodes: Sequence[IROp],
        claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]],
        profile: Optional[ProfileConfig],
    ) -> Dict[str, Any]:
        """Create a coordinator for every long-running group among ``nodes``."""
        members: Dict[str, int] = {}
        options: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            match node:
                case IRMap(
                    compute=ComputeAssignment(
                        resources=ComputeResources(task=TaskType.LONG_RUNNING),
                        group=group,
                    ) as compute,
                ):
                    members[group] = members.get(group, 0) + 1
                    options[group] = claims[compute]
                case _:
                    pass

        return {
            group: RayComputeGroup.options(
                runtime_env={"env_vars": numpy_hugepage_env()},
            ).remote(options[group], count, self.parameters, group, profile)
            for group, count in members.items()
        }


def _group_of(
    groups: Mapping[str, Any],
    compute: Optional[ComputeAssignment],
) -> Optional[Any]:
    """The coordinator of the long-running group of a map, if any.

    The ``cpu_only`` maps of a long-running protocol share its group id but
    run outside of its executor.
    """
    match compute:
        case ComputeAssignment(
            resources=ComputeResources(task=TaskType.LONG_RUNNING),
            group=group,
        ):
            return groups[group]
        case _:
            return None


@dataclass
class _RouteBuilder:
    """Builds the routes of the stages with actors, through the hosted maps."""

    actor_map: Mapping[IROp, Any]
    labels: Mapping[IROp, str]
    groups: Mapping[str, Any]
    claims: Mapping[Optional[ComputeAssignment], Dict[str, Any]]
    parameters: Mapping[str, Any]
    profile: Optional[ProfileConfig]

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
        return _Apply(
            label=self.labels[node],
            func=node.func,
            compute=node.compute,
            options=self.claims[node.compute],
            group=_group_of(self.groups, node.compute),
            parameters=self.parameters,
            profile=self.profile,
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
