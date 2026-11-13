"""The actors of the Ray backend, and the runtime of their stages.

``backend.py`` compiles a pipeline into these actors: stage actors (sources,
merges, accumulators, sinks, ``group_by`` routers) exchanging items, the maps
between them running on their routes, and the pools of executors of compute
groups.
"""

from __future__ import annotations
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)
import abc
import asyncio
import contextlib
import logging
import threading
import time
import ray  # type: ignore
from ray.util.placement_group import (  # type: ignore
    PlacementGroup,
    placement_group,
    remove_placement_group,
)
from ray.util.scheduling_strategies import (  # type: ignore
    PlacementGroupSchedulingStrategy,
)

from scipion_bridge.core.environment.compute import ComputeAssignment
from scipion_bridge.core.streaming.backend import StageStats
from scipion_bridge.core.streaming.ir import (
    IROp,
    IRSink,
    Keyed,
    Tagged,
    clone_ir,
)
from scipion_bridge.core.streaming.keyed import share_keys
from scipion_bridge.core.streaming.node import FlushSignal, FLUSH
from scipion_bridge.core.streaming.provenance import ProvenanceSampler, StageSamples
from scipion_bridge.core.streaming.sink_writer import SinkWriter
from scipion_bridge.core.streaming.spill import SpillStoreFactory

from .function_table import table_of
from .mailbox import Mailbox, Ref, SpillPolicy
from .placement import Bundle
from .profiling import ProfileConfig, Recorder, make_recorder
from .worker_env import configure_worker_env

logger = logging.getLogger(__name__)

_Barrier = asyncio.Future[None]
_InboxEntry = Tuple[Any, int, Optional[_Barrier]]
# An item ready to be pushed to a downstream stage: (actor, port, item).
_Delivery = Tuple[Any, int, Any]


@dataclass(frozen=True)
class _EmitItem:
    """An item a stage computed, waiting in its outbox to be routed."""

    item: Any


@dataclass(frozen=True)
class _InFlightItem:
    """An item being routed through the maps on the routes of a stage."""

    routing: asyncio.Task[List[_Delivery]]


@dataclass(frozen=True)
class _FlushBarrier:
    """A FLUSH, with the barriers it releases once it has propagated."""

    barriers: List[_Barrier]


_OutboxEntry = Union[_EmitItem, _FlushBarrier]
# Items being routed, in order, and the FLUSHes between them.
_InFlightEntry = Union[_InFlightItem, _FlushBarrier]


def _barriers_of(
    entry: Union[_EmitItem, _InFlightItem, _FlushBarrier],
) -> List[_Barrier]:
    """The barriers released once ``entry`` has propagated."""
    match entry:
        case _FlushBarrier(barriers=barriers):
            return barriers
        case _EmitItem() | _InFlightItem():
            return []


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
        self._provenance = ProvenanceSampler()
        self._label = label
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
        # The pools the stage runs maps of, told when FLUSH passes the stage.
        self._releases: List[Any] = []
        self._num_inputs = 1
        self._pending_flushes: List[_Barrier] = []
        self._error: Optional[BaseException] = None
        self._tasks: List[asyncio.Task[None]] = []

    async def connect(
        self,
        routes: List[_Route],
        abort_targets: List[Any],
        num_inputs: int,
        releases: Sequence[Any] = (),
    ) -> None:
        """Configure the stage and start its loops.

        Args:
            routes: Routes of the results to the downstream stages, through
                the maps between them.
            abort_targets: Actor handles aborted when this stage fails: the
                sources of the pipeline and, for a nested pipeline, the
                sources of its parent.
            num_inputs: Number of upstream stages, each sending one FLUSH.
            releases: Coordinators of the pools the stage runs maps of
                (for a ``group_by``, also in its children), told when FLUSH
                has passed the stage (``RayComputeGroup.flushed``).
        """
        self._routes = routes
        self._releases = list(releases)
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

    async def provenance(self) -> Dict[str, StageSamples]:
        """Return the samples of this stage and of the maps on its routes.

        A map on an actor of its own has the label of the actor; the samples
        of its calls are the ones of the map.
        """
        return {
            self._label: self._provenance.samples,
            **{
                label: runner.provenance.samples for label, runner in self._maps.items()
            },
        }

    async def flush_profile(self) -> None:
        """Wait until the collector wrote the profiling events of this stage."""
        await self._recorder.flush()

    async def close_spill(self) -> None:
        """Remove the spilled items of the stage from the disk of its node."""
        await asyncio.to_thread(self._mailbox.close)

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
                    if self._stats.first_in_t is None:
                        self._stats.first_in_t = time.time()
                    span = self._recorder.span(
                        "process",
                        "compute",
                        seq=self._stats.items_in,
                        port=port,
                    )

            if not isinstance(item, FlushSignal):
                self._provenance.receive(item)
            try:
                with span:
                    emissions = await self._compute(item, port)
            except Exception as error:
                self._set_error(error)
                _fail(barriers, error)
                continue

            t_processed = time.perf_counter()
            self._stats.process_s += t_processed - t_start
            self._provenance.emit(
                None if isinstance(item, FlushSignal) else item,
                emissions,
            )

            with self._recorder.span("blocked", "compute", items=len(emissions)):
                for out_item in emissions:
                    await self._outbox.put(_EmitItem(out_item))
                if barriers:
                    await self._outbox.put(_FlushBarrier(barriers))
            self._stats.blocked_s += time.perf_counter() - t_processed

    async def _compute(self, item: Any, port: int) -> List[Any]:
        match item:
            case FlushSignal():
                return await self.on_flush()

            case _:
                return await self.process(item, port)

    async def _run_emit(self) -> None:
        while True:
            entry = await self._outbox.get()
            if self._error is not None:
                _fail(_barriers_of(entry), self._error)
                continue

            match entry:
                case _EmitItem(item=item):
                    # The maps of the item start once a slot is free; the slot
                    # is released when the item has been forwarded.
                    await self._in_flight_slots.acquire()
                    routing = asyncio.create_task(self._route(self._routes, item))
                    self._in_flight.put_nowait(_InFlightItem(routing))
                case _FlushBarrier():
                    self._in_flight.put_nowait(entry)

    async def _run_forward(self) -> None:
        while True:
            entry = await self._in_flight.get()
            try:
                await self._forward(entry)
            finally:
                if isinstance(entry, _InFlightItem):
                    self._in_flight_slots.release()

    async def _forward(self, entry: _InFlightEntry) -> None:
        """Deliver a routed item, or a FLUSH, to the downstream stages."""
        if self._error is not None:
            _fail(_barriers_of(entry), self._error)
            if isinstance(entry, _InFlightItem):
                entry.routing.cancel()
            return

        t_start = time.perf_counter()
        try:
            with self._forward_span(entry):
                deliveries = await self._finish(entry)
                await asyncio.gather(
                    *(_deliver(handle, item, port) for handle, port, item in deliveries)
                )
        except Exception as error:
            self._set_error(error)
            _fail(_barriers_of(entry), error)
            return

        match entry:
            case _InFlightItem():
                self._stats.emit_s += time.perf_counter() - t_start
                self._stats.items_out += 1
                self._stats.last_out_t = time.time()
            case _FlushBarrier(barriers=barriers):
                # The events up to the FLUSH are in the trace when it returns.
                await self._recorder.flush()
                _release(barriers)

    def _forward_span(
        self,
        entry: _InFlightEntry,
    ) -> AbstractContextManager[Dict[str, Any]]:
        """Span of forwarding an item, from its routing until its delivery."""
        match entry:
            case _FlushBarrier():
                return self._recorder.span("flush", "forward", level=logging.INFO)
            case _InFlightItem():
                return self._recorder.span(
                    "forward",
                    "forward",
                    seq=self._stats.items_out + 1,
                )

    async def _finish(self, entry: _InFlightEntry) -> List[_Delivery]:
        """Wait for the maps of an item, or prepare a FLUSH for every route.

        All earlier items have been delivered when a FLUSH is forwarded, so
        the long-running maps release their resources first.
        """
        match entry:
            case _FlushBarrier():
                await asyncio.gather(*(m.release() for m in self._maps.values()))
                # The calls of this stage are done: a pool whose other stages
                # are done too releases its GPUs for the next phase.
                await asyncio.gather(
                    *(group.flushed.remote(self._label) for group in self._releases),
                )
                return [
                    (route.handle, route.port, FLUSH)
                    for route in _walk_routes(self._routes)
                    if isinstance(route, _Push)
                ]
            case _InFlightItem(routing=routing):
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
    if task is not None:
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
        configure_worker_env(parameters=parameters)
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
    global _served_parameters
    if _served_parameters != dict(parameters):
        configure_worker_env(parameters=parameters)
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

# Seconds a pool waits for its placement group before it warns that the
# cluster lacks the GPUs of the plan.
_RESERVE_WARNING_S = 30.0

# The label of every Ray node with its id, which pins a bundle to a node.
_NODE_ID_LABEL = "ray.io/node-id"


@dataclass(frozen=True)
class _Reservation:
    """The placement group a long-running pool reserves while it holds GPUs.

    Attributes:
        name: The prefix of the names of its placement groups: the id of the
            pipeline and the group.
        bundles: The resources of every bundle.
        selectors: The label selector of every bundle, pinning it to a node.
        executors: The executors that run in every bundle.
    """

    name: str
    bundles: Tuple[Dict[str, float], ...]
    selectors: Tuple[Dict[str, str], ...]
    executors: Tuple[int, ...]


def _reservation_of(
    name: str,
    bundles: Sequence[Bundle],
    node_ids: Sequence[str],
) -> Optional[_Reservation]:
    """The reservation of the planned ``bundles``, with the nodes resolved
    from their index to their Ray id; ``None`` without bundles."""
    if not bundles:
        return None

    return _Reservation(
        name=name,
        bundles=tuple(dict(bundle.resources) for bundle in bundles),
        selectors=tuple({_NODE_ID_LABEL: node_ids[bundle.node]} for bundle in bundles),
        executors=tuple(bundle.executors for bundle in bundles),
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
        blockers: Sequence[Any] = (),
        reservation: Optional[_Reservation] = None,
    ) -> None:
        """
        Args:
            members: The maps of the group, each running one call at a time
                on an executor.
            blockers: The coordinators of the pools of earlier phases: an
                executor starts once they have released their GPUs.
            reservation: The placement group the executors run in, reserved
                when the first one starts and removed when the pool is
                released; ``None`` lets Ray place them.
        """
        self._options = options
        self._members = members
        self._blockers = list(blockers)
        self._reservation = reservation
        # The placement group of the current stream, and its reservation.
        self._placement_group: Optional[PlacementGroup] = None
        self._reserved: Optional[asyncio.Future[None]] = None
        # The bundle of every ready executor, and of the executors starting,
        # by a token, with the generation they start in.
        self._bundle_of: Dict[int, int] = {}
        self._starting_in: Dict[int, Tuple[int, int]] = {}
        self._next_token = 0
        # The stages that run maps of the group, which release the pool once
        # all of them have passed FLUSH; and those that have, this stream.
        self._hosts: FrozenSet[str] = frozenset()
        self._flushed: Set[str] = set()
        # Set once the pool released its executors in the current stream.
        self._released = asyncio.Event()
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
            while generation == self._generation:
                leased = self._try_hand_out(held)
                if leased is not None:
                    return leased
                await self._changed.wait()
            return None

    def _try_hand_out(self, held: Sequence[int]) -> Optional[Tuple[int, Any]]:
        """Lease an executor the map does not hold, if one is to be shared
        now; else start a missing one. Call with the lock held.

        Raises:
            Exception: The error of an executor that failed to start, if no
                executor is unleased.
        """
        others = {i: n for i, n in self._leases.items() if i not in held}
        unleased = [i for i, leases in others.items() if leases == 0]
        if unleased:
            return self._hand_out(unleased[0])
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure
        if self._replicas - len(self._executors) - self._starting > 0:
            self._start_one()
            return None
        if self._starting > 0 or not others:
            return None
        return self._hand_out(min(others, key=others.__getitem__))

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
            if self._failure is not None:
                failure, self._failure = self._failure, None
                raise failure

    def _start_one(self) -> None:
        """Start an executor in the background; call with the lock held."""
        self._starting += 1
        task = asyncio.create_task(self._start_executor(self._generation))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _start_executor(self, generation: int) -> None:
        # A pool of a later phase waits for the GPUs of the earlier ones.
        if self._blockers:
            with self._recorder.span("wait_for_phase", "group", level=logging.INFO):
                await asyncio.gather(
                    *(b.wait_released.remote() for b in self._blockers)
                )
        token, bundle = self._take_bundle(generation)
        executor: Optional[Any] = None
        try:
            placement = await self._placement(bundle, generation)
            if placement is not None:
                self._recorder.instant("executor_create", "group")
                executor = RayComputeExecutor.options(
                    max_concurrency=self._members * _EXECUTOR_CALLS_IN_FLIGHT,
                    **self._options,
                    **placement,
                ).remote(self._parameters, self._group, self._profile)
                await executor.prepare.remote(self._functions)
        except Exception as error:
            if executor is not None:
                ray.kill(executor)
            async with self._changed:
                self._starting -= 1
                del self._starting_in[token]
                # A release meanwhile removed the placement group: the start
                # was cancelled, it did not fail.
                if generation == self._generation:
                    self._failure = error
                # A pool whose executors fail to start holds no GPUs.
                if not self._executors and not self._starting:
                    self._unreserve()
                self._changed.notify_all()
        else:
            async with self._changed:
                self._starting -= 1
                del self._starting_in[token]
                # Without an executor, the pool was released before it was
                # created.
                if executor is not None and generation == self._generation:
                    self._executors[self._next_id] = executor
                    self._leases[self._next_id] = 0
                    self._bundle_of[self._next_id] = bundle
                    self._next_id += 1
                elif executor is not None:
                    # The pool was released while the executor started.
                    ray.kill(executor)
                self._changed.notify_all()

    def _take_bundle(self, generation: int) -> Tuple[int, int]:
        """The bundle with the most room for an executor starting now, and a
        token for the start; the bundle is -1 without a reservation."""
        reservation = self._reservation
        if reservation is None:
            bundle = -1
        else:
            used = [
                *self._bundle_of.values(),
                *(b for g, b in self._starting_in.values() if g == generation),
            ]
            bundle = max(
                range(len(reservation.executors)),
                key=lambda b: reservation.executors[b] - used.count(b),
            )
        token = self._next_token
        self._next_token += 1
        self._starting_in[token] = (generation, bundle)
        return token, bundle

    async def _placement(
        self, bundle: int, generation: int
    ) -> Optional[Dict[str, Any]]:
        """The Ray options placing an executor in ``bundle``, once the
        placement group of the pool is reserved; none without a reservation.

        ``None`` if the pool was released since the start of ``generation``
        began: it must not reserve GPUs for a stream that has ended.

        Raises:
            Exception: If the placement group was removed while it was
                reserved.
        """
        if generation != self._generation:
            return None
        reservation = self._reservation
        if reservation is None:
            return {}
        if self._reserved is None:
            group = placement_group(
                list(reservation.bundles),
                strategy="PACK",
                name=f"{reservation.name}/{self._generation}",
                bundle_label_selector=list(reservation.selectors),
            )
            self._placement_group = group
            self._reserved = asyncio.ensure_future(self._reserve(group))
        assert self._reserved is not None and self._placement_group is not None
        group = self._placement_group
        await self._reserved
        return {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_bundle_index=bundle,
            ),
        }

    async def _reserve(self, group: PlacementGroup) -> None:
        """Wait until Ray reserved ``group``, warning if it takes long.

        Raises:
            Exception: If the group was removed meanwhile.
        """
        ready = asyncio.wrap_future(group.ready().future())
        with self._recorder.span("reserve", "group", level=logging.INFO):
            done, _ = await asyncio.wait({ready}, timeout=_RESERVE_WARNING_S)
            if not done:
                logger.warning(
                    "The executors of %s wait for their GPUs (%s) since %g s: "
                    "other jobs may hold them. RayBackend(placement='ray') "
                    "lets Ray place the executors instead.",
                    self._group,
                    self._reservation,
                    _RESERVE_WARNING_S,
                )
            await ready

    async def return_leases(self, executor_ids: Sequence[int]) -> None:
        """Return the leases of a map without calls in flight."""
        async with self._changed:
            self._return(executor_ids)

    def _return(self, executor_ids: Sequence[int]) -> None:
        # Leases of executors released meanwhile are gone.
        executor_ids = [i for i in executor_ids if i in self._leases]
        for executor_id in executor_ids:
            self._leases[executor_id] -= 1
        # Only the executors of ephemeral protocols are reaped when idle.
        if self._idle_s is None:
            return

        now = time.monotonic()
        for executor_id in executor_ids:
            if self._leases[executor_id] == 0:
                self._idle_since[executor_id] = now
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap(self._idle_s))

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

    async def set_hosts(self, labels: Sequence[str]) -> None:
        """Set the stages running maps of the group (see ``flushed``)."""
        self._hosts = frozenset(labels)

    async def flushed(self, label: str) -> None:
        """Note that FLUSH has passed the stage ``label``, after its last call.

        Once it has passed every stage running maps of the group, the pool
        releases its executors, so that the pools of a later phase can take
        the GPUs while the stream drains downstream.
        """
        self._flushed.add(label)
        if self._hosts <= self._flushed and not self._released.is_set():
            await self._release()
            self._released.set()

    async def wait_released(self) -> None:
        """Return once the pool released its executors in the current stream."""
        await self._released.wait()

    async def release_all(self) -> None:
        """Kill the executors of the pool once its pipeline has flushed.

        Ends the stream for the gates: the next one starts with phase 0.
        """
        await self._release()
        self._flushed.clear()
        self._released.clear()

    async def _release(self) -> None:
        async with self._changed:
            self._generation += 1
            self._changed.notify_all()
            await self._kill(list(self._executors))
            self._unreserve()

    def _unreserve(self) -> None:
        """Remove the placement group of the pool, freeing its GPUs."""
        if self._placement_group is not None:
            remove_placement_group(self._placement_group)
            self._placement_group = None
            self._reserved = None

    async def _kill(self, executor_ids: Sequence[int]) -> None:
        """Kill the executors ``executor_ids``, releasing their resources."""
        executors = [self._executors.pop(executor_id) for executor_id in executor_ids]
        for executor_id in executor_ids:
            del self._leases[executor_id]
            self._bundle_of.pop(executor_id, None)
            self._idle_since.pop(executor_id, None)
        # Killing an executor discards the events it holds.
        await asyncio.gather(*(e.flush_profile.remote() for e in executors))
        for executor in executors:
            ray.kill(executor)
        if executors:
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
        if self.idle:
            self._on_idle()

    async def _acquire(self) -> int:
        """An idle executor, else the least busy with room, else wait."""
        async with self._changed:
            self._waiting += 1
            try:
                while True:
                    executor_id = self._try_acquire_slot()
                    if executor_id is not None:
                        self._in_flight[executor_id] += 1
                        return executor_id
                    await self._changed.wait()
            finally:
                self._waiting -= 1

    def _try_acquire_slot(self) -> Optional[int]:
        """An idle executor, else the least busy with room, else ``None``.

        Without an idle executor, leases another one in the background while
        the map holds fewer than ``replicas``. Call with the lock held.

        Raises:
            Exception: The error of a failed lease, if no executor has room.
        """
        idle = [i for i, calls in self._in_flight.items() if calls == 0]
        free = [i for i, calls in self._in_flight.items() if calls < self._capacity]
        can_lease = len(self._executors) < self._replicas and self._leasing is None
        if not idle and can_lease:
            self._leasing = asyncio.create_task(self._grow())
        if idle:
            return idle[0]
        if free:
            return min(free, key=self._in_flight.__getitem__)
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure
        return None

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
                # Without a lease, the pool was released while it waited.
                if leased is not None:
                    executor_id, executor = leased
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
        self.provenance = ProvenanceSampler()
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
        if self.stats.first_in_t is None:
            self.stats.first_in_t = time.time()
        with self._recorder.span(
            "call",
            f"map:{self._label}",
            overlapping=True,
            seq=self.stats.items_in + 1,
        ):
            result, waited = await self._call(item)

        self.stats.items_in += 1
        self.stats.items_out += 1
        self.stats.wait_s += waited
        self.stats.wait_max_s = max(self.stats.wait_max_s, waited)
        self.stats.process_s += time.perf_counter() - t_start - waited
        self.stats.last_out_t = time.time()
        if self.provenance.sampling:
            await self._sample(item, result)
        return result

    async def _sample(self, item: Any, result: ray.ObjectRef) -> None:
        """Sample the item of a call and its result, fetched from the object
        store (for the first items only, see ``provenance.py``)."""
        received = await item if isinstance(item, ray.ObjectRef) else item
        self.provenance.receive(received)
        self.provenance.emit(received, [await result])

    async def _call(self, item: Any) -> Tuple[ray.ObjectRef, float]:
        """The result of a call, and how long it waited for an executor.

        Calls without a pool queue inside Ray, which the map cannot see.
        """
        if self._group is None:
            return await self._completions.done(self._submit(item)), 0.0
        else:
            _cancel(self._linger)
            t_wait = time.perf_counter()
            async with self._executors.slot() as executor:
                waited = time.perf_counter() - t_wait
                result = await self._completions.done(
                    executor.call.remote(
                        self._functions,
                        self._label,
                        self._stage,
                        item,
                    ),
                )
            return result, waited

    def _schedule_linger(self) -> None:
        """Return the leases soon, if the map is idle (ephemeral protocols)."""
        if self._returns_idle_leases:
            _cancel(self._linger)
            self._linger = asyncio.create_task(self._return_idle_leases())

    async def _return_idle_leases(self) -> None:
        """Return the leases once the map had no call in flight for a moment."""
        await asyncio.sleep(_LEASE_LINGER_S)
        if self._executors.idle:
            leases = self._executors.ids
            self._executors.clear()
            assert self._group is not None
            await self._group.return_leases.remote(leases)

    async def release(self) -> None:
        """Release the resources of the map's pool on FLUSH."""
        _cancel(self._linger)
        if self._group is not None:
            leases = self._executors.ids
            self._executors.stop()
            await self._group.return_leases.remote(leases)

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


class ChildPipeline(Protocol):
    """A compiled child pipeline of a ``group_by`` (a ``RayCompiledPipeline``)."""

    async def start_async(self) -> None: ...

    def source_handle(self, source_name: str) -> Any: ...

    def stats(self) -> Dict[str, StageStats]: ...

    def provenance(self) -> Dict[str, StageSamples]: ...

    def close_spill(self) -> List[ray.ObjectRef]: ...

    def flush_profile(self) -> None: ...


class ChildBackend(Protocol):
    """Compiles the child pipelines of a ``group_by`` (a ``RayBackend``
    configured by the parent: its buffering, profile and pools)."""

    def compile(
        self,
        ir_sinks: List[IROp],
        *,
        name: str = "",
        abort_targets: Sequence[Any] = (),
    ) -> ChildPipeline: ...


class _Child:
    """Child pipeline of a ``group_by`` key, fed in order once it has started.

    Starting a child waits for its actors to start their processes. The items
    arriving meanwhile wait in its queue, so that the router keeps routing
    the items of the other keys and the children of new keys start in
    parallel.
    """

    def __init__(
        self,
        start: Coroutine[Any, Any, ChildPipeline],
        source_name: str,
        on_error: Callable[[BaseException], None],
        buffer_size: Optional[int],
    ) -> None:
        """
        Args:
            start: Compiles and starts the child pipeline.
            source_name: Input of the child pipeline fed with the items.
            on_error: Called with the first error of the child's start or
                feed.
            buffer_size: Most items queued for the child; ``send`` waits for
                room beyond it. ``None`` queues without limit.
        """
        self.ready: asyncio.Task[ChildPipeline] = asyncio.create_task(start)
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
        backend: ChildBackend,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        parameters: Mapping[str, Any],
    ):
        """
        Args:
            workers: Children shared by the keys; ``None`` starts a child per
                key.
            outer_keyed: Whether the items arrive as ``Keyed(outer, item)``.
            backend: Compiles the children, with the buffering, profile and
                pools of the parent pipeline.
        """
        configure_worker_env(parameters=parameters)
        super().__init__(config, spill, label)
        self.key_fn = key_fn
        self.template = template
        self.source_name = source_name
        self.max_keys = max_keys
        self.workers = workers
        self.outer_keyed = outer_keyed
        self.backend = backend
        # The children are labelled under it, ``<label>[<child>]/``.
        self.label = label
        self.config = config
        # Children by key, or by index of a shared child.
        self._children: Dict[Any, _Child] = {}
        # Child of every key.
        self._keys: Dict[Any, _Child] = {}

    async def process(self, item: Any, port: int) -> List[Any]:
        if self.outer_keyed:
            outer, value = item
            key = (outer, await asyncio.to_thread(self.key_fn, value))
        else:
            value = item
            key = await asyncio.to_thread(self.key_fn, value)

        if key not in self._keys:
            self._keys[key] = self._assign(key)

        child_item = value if self.workers is None else Keyed(key=key, value=value)
        await self._keys[key].send(child_item)
        return []

    async def on_flush(self) -> List[Any]:
        await asyncio.gather(*(child.flush() for child in self._children.values()))
        return []

    async def emit(self, key: Any, item: Any) -> None:
        """Forward a result of the child pipeline of ``key``."""
        if self.outer_keyed:
            outer, inner = key
            keyed = Keyed(key=outer, value=Keyed(key=inner, value=item))
        else:
            keyed = Keyed(key=key, value=item)
        await self._outbox.put(_EmitItem(keyed))

    async def children_stats(self) -> Dict[str, StageStats]:
        """Return the execution metrics of every stage of the started children."""
        stats = await asyncio.gather(
            *(asyncio.to_thread(child.stats) for child in self._started()),
        )
        return {label: stage for child in stats for label, stage in child.items()}

    async def children_provenance(self) -> Dict[str, StageSamples]:
        """Return the provenance samples of every stage of the started children."""
        samples = await asyncio.gather(
            *(asyncio.to_thread(child.provenance) for child in self._started()),
        )
        return {label: stage for child in samples for label, stage in child.items()}

    async def close_spill(self) -> None:
        """Remove the spilled items of the router and of its children."""
        await asyncio.gather(
            super().close_spill(),
            *(ref for child in self._started() for ref in child.close_spill()),
        )

    async def flush_profile(self) -> None:
        """Wait until the collector wrote the events of the router and its children."""
        await asyncio.gather(
            *(asyncio.to_thread(child.flush_profile) for child in self._started()),
        )
        await super().flush_profile()

    def _started(self) -> List[ChildPipeline]:
        """The child pipelines that have started."""
        return [
            child.ready.result()
            for child in self._children.values()
            if child.ready.done()
        ]

    def _assign(self, key: Any) -> _Child:
        """The child of a new key, started if necessary."""
        if self.max_keys is not None and len(self._keys) >= self.max_keys:
            raise ValueError(
                f"group_by received key {key!r} after max_keys={self.max_keys} "
                f"keys: {list(self._keys)}.",
            )

        router = ray.get_runtime_context().current_actor
        if self.workers is None:
            (exit_,) = clone_ir([self.template])
            self._children[key] = self._start(
                exit_,
                _DemuxResultWriter(router, key),
                f"{key}",
            )
            return self._children[key]
        else:
            worker = len(self._keys) % self.workers
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
        return _Child(
            self._start_child(sink, child),
            self.source_name,
            self._set_error,
            self.config.buffer_size,
        )

    async def _start_child(self, sink: IRSink, child: str) -> ChildPipeline:
        """Compile and start the child ending in ``sink``, recording how long
        it takes.

        Starting the actors of a child takes seconds; the router awaits it
        while it keeps routing, and starts several children at once.
        """
        with self._recorder.span(
            "start_child",
            "group_by",
            level=logging.INFO,
            overlapping=True,
            child=child,
        ):
            pipeline = await asyncio.to_thread(
                self.backend.compile,
                [sink],
                name=f"{self.label}[{child}]/",
                abort_targets=self._abort_targets,
            )
            await pipeline.start_async()
            return pipeline
