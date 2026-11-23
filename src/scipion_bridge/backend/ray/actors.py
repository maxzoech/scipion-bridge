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
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import abc
import asyncio
import contextlib
import dataclasses
import logging
import threading
import time
import ray  # type: ignore
from ray.exceptions import RayError  # type: ignore
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
from scipion_bridge.core.streaming.lineage import (
    AccumulatorLineage,
    ElementSegment,
    ItemId,
    LineagePolicy,
    LineageRecord,
    PassThrough,
    SeenElements,
    SourceLineage,
    call_elements,
    element_segments,
    fill_created,
    input_files,
    map_id,
)
from scipion_bridge.core.streaming.run_store import Commit, RunStore
from scipion_bridge.core.streaming.node import FlushSignal, FLUSH
from scipion_bridge.core.streaming.provenance import ProvenanceSampler, StageSamples
from scipion_bridge.core.streaming.sink_writer import (
    CommittingSinkWriter,
    SinkWriter,
    UniqueElements,
)
from scipion_bridge.core.streaming.spill import SpillStoreFactory

from .function_table import table_of
from .mailbox import Mailbox, Ref, SpillPolicy
from .placement import Bundle
from .profiling import ProfileConfig, Recorder, make_recorder
from .worker_env import configure_worker_env

logger = logging.getLogger(__name__)

_Barrier = asyncio.Future[None]
# A received item, the port it arrived on, its id and, for a FLUSH, its
# barrier.
_InboxEntry = Tuple[Any, int, Optional[ItemId], Optional[_Barrier]]
# An item ready to be pushed to a downstream stage: (actor, port, item, id).
_Delivery = Tuple[Any, int, Any, Optional[ItemId]]


@dataclass(frozen=True)
class _EmitItem:
    """An item a stage computed, with its id, waiting in its outbox to be
    routed."""

    item: Any
    lid: ItemId


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


# Seconds a stage keeps lineage records at most before it writes them, while
# items arrive; so that a run store shows the lineage of a running pipeline.
_LINEAGE_INTERVAL_S = 30.0

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
    # Where the stages record the lineage of the items they make.
    run_store: Optional[RunStore] = None
    # Lineage records a stage writes to the run store at once.
    lineage_batch: int = 2000


_T = TypeVar("_T")


class _Batches(Generic[_T]):
    """Values a stage writes to its run store in the background.

    They are written in batches of ``size``, at least every
    ``_LINEAGE_INTERVAL_S`` while values arrive, and the rest on ``drain``.
    A write starts after the writes before it, so that the parts of a stage
    keep their order. The time writing adds to ``lineage_s`` of ``stats``.
    """

    def __init__(
        self,
        write: Callable[[List[_T]], None],
        size: int,
        stats: StageStats,
    ) -> None:
        self._write = write
        self._size = size
        self._stats = stats
        self._values: List[_T] = []
        self._last_write = time.monotonic()
        self._writes: Optional[asyncio.Task[None]] = None

    def add(self, value: _T) -> None:
        """Write ``value`` with the next batch; start it if it is full or the
        last one is old enough."""
        self._values.append(value)
        if (
            len(self._values) >= self._size
            or time.monotonic() - self._last_write >= _LINEAGE_INTERVAL_S
        ):
            self._start()

    async def drain(self) -> None:
        """Write the values not written yet, and wait until every write
        started has finished."""
        self._start()
        assert self._writes is not None
        await self._writes

    def _start(self) -> None:
        values, self._values = self._values, []
        self._last_write = time.monotonic()
        self._writes = asyncio.create_task(self._append(self._writes, values))

    async def _append(
        self,
        previous: Optional[asyncio.Task[None]],
        values: List[_T],
    ) -> None:
        if previous is not None:
            await previous
        t_start = time.perf_counter()
        await asyncio.to_thread(self._write, values)
        self._stats.lineage_s += time.perf_counter() - t_start


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


async def _deliver(
    handle: Any,
    item: Any,
    port: int,
    lid: Optional[ItemId] = None,
) -> None:
    """Push ``item``, whose id is ``lid``, into the stage of ``handle`` on
    ``port``.

    Returns once the item is buffered by the stage, or for FLUSH (which has
    no id) once the flush has propagated through the stage and its
    downstream.
    """
    await handle.push.remote(_transport(item), port, lid)


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

    Each item also travels with its id (see ``lineage.py``), which the
    stage's ``LineagePolicy`` derives the ids of its emissions from. Rows the
    stage created get element ids from the ids of the items they are emitted
    in.

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
        # A stage processes one item at a time.
        self._stats = StageStats(capacity=1)
        self._provenance = ProvenanceSampler()
        self._label = label
        self._lineage = self._lineage_policy(label)
        self._run_store = config.run_store
        # Lineage records, written in batches while items arrive and the
        # rest at every FLUSH.
        self._records: Optional[_Batches[LineageRecord]] = (
            None
            if config.run_store is None
            else _Batches(
                config.run_store.append_lineage,
                config.lineage_batch,
                self._stats,
            )
        )
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

    async def push(
        self,
        item: Any,
        port: int = 0,
        lid: Optional[ItemId] = None,
    ) -> None:
        """Buffer an item; for FLUSH, wait until the pipeline has drained.

        ``lid`` is the id of the item; the driver and the loaders push
        items without one into sources, which derive it.
        """
        if self._error is not None:
            raise self._error

        match item:
            case FlushSignal():
                done: _Barrier = asyncio.get_running_loop().create_future()
                await self._mailbox.put(item, port, None, done)
                await done
            case _:
                await self._mailbox.put(item, port, lid, None)

    def _lineage_policy(self, label: str) -> LineagePolicy:
        """How the stage derives the ids of the items it emits."""
        return PassThrough(label)

    @abc.abstractmethod
    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
        """Process an item arriving on ``port`` with the id ``lid`` and return
        the items to emit."""
        ...

    async def on_flush(self) -> List[Any]:
        """Finalize the stage on FLUSH and return the items to emit downstream."""
        return []

    async def _run_fetch(self) -> None:
        while True:
            entry, port, lid, done = await self._mailbox.get()
            try:
                item = await self._mailbox.resolve(entry)
            except Exception as error:
                self._set_error(error)
                _fail([] if done is None else [done], error)
                continue

            await self._inbox.put((item, port, lid, done))

    async def _run_compute(self) -> None:
        while True:
            t_wait = time.perf_counter()
            item, port, lid, done = await self._inbox.get()
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
                    emissions = await self._compute(item, port, lid)
            except Exception as error:
                self._set_error(error)
                _fail(barriers, error)
                continue

            t_processed = time.perf_counter()
            self._stats.process_s += t_processed - t_start
            if not isinstance(item, FlushSignal):
                self._stats.last_done_t = time.time()
            self._provenance.emit(
                None if isinstance(item, FlushSignal) else item,
                [out_item for out_item, _ in emissions],
            )

            with self._recorder.span("blocked", "compute", items=len(emissions)):
                for out_item, out_lid in emissions:
                    await self._outbox.put(_EmitItem(out_item, out_lid))
                if barriers:
                    await self._outbox.put(_FlushBarrier(barriers))
            self._stats.blocked_s += time.perf_counter() - t_processed

    async def _compute(
        self,
        item: Any,
        port: int,
        lid: Optional[ItemId],
    ) -> List[Tuple[Any, ItemId]]:
        """The items the stage emits for ``item``, with their ids; rows they
        created get element ids."""
        match item:
            case FlushSignal():
                emissions = await self.on_flush()
                ids = self._lineage.flush(len(emissions))

            case _:
                received = self._lineage.receive(item, lid)
                if lid is None:
                    self._record_source(item, received)
                emissions = await self.process(item, port, received)
                ids = self._lineage.emit(received, len(emissions))

        for emission, emission_id in zip(emissions, ids):
            fill_created(emission, emission_id)
        if self._lineage.kind == "accumulate" and self._run_store is not None:
            t_start = time.perf_counter()
            for emission, emission_id in zip(emissions, ids):
                self._record(
                    emission_id,
                    "accumulate",
                    self._lineage.emitted_parents,
                    element_segments(emission),
                )
            self._stats.lineage_s += time.perf_counter() - t_start
        return list(zip(emissions, ids))

    def _record_source(self, item: Any, lid: ItemId) -> None:
        """Record an item the driver or a loader sent into this source."""
        if self._run_store is not None:
            t_start = time.perf_counter()
            fill_created(item, lid)
            self._keep(
                LineageRecord(
                    lid,
                    self._label,
                    "source",
                    (),
                    element_segments(item),
                    input_files(item),
                ),
            )
            self._stats.lineage_s += time.perf_counter() - t_start

    def _record(
        self,
        lid: ItemId,
        kind: str,
        parents: Tuple[ItemId, ...],
        elements: Tuple[ElementSegment, ...],
        stage: Optional[str] = None,
        inherited: bool = False,
    ) -> None:
        """Record an item this stage, or the map ``stage`` on its routes, made."""
        if self._run_store is not None:
            self._keep(
                LineageRecord(
                    lid,
                    stage or self._label,
                    kind,
                    parents,
                    elements,
                    inherited=inherited,
                ),
            )

    def _keep(self, record: LineageRecord) -> None:
        """Write ``record`` with the next batch."""
        assert self._records is not None
        self._records.add(record)

    async def _write_lineage(self) -> None:
        """Write the records not written yet, and wait until every write
        started has finished: once FLUSH has passed, readers see them all."""
        if self._records is not None:
            await self._records.drain()

    async def _run_emit(self) -> None:
        while True:
            entry = await self._outbox.get()
            if self._error is not None:
                _fail(_barriers_of(entry), self._error)
                continue

            match entry:
                case _EmitItem(item=item, lid=lid):
                    # The maps of the item start once a slot is free; the slot
                    # is released when the item has been forwarded.
                    await self._in_flight_slots.acquire()
                    routing = asyncio.create_task(
                        self._route(self._routes, item, lid),
                    )
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
                    *(
                        _deliver(handle, item, port, lid)
                        for handle, port, item, lid in deliveries
                    ),
                )
        except Exception as error:
            self._set_error(error)
            _fail(_barriers_of(entry), error)
            return

        match entry:
            case _InFlightItem():
                self._stats.emit_s += time.perf_counter() - t_start
                self._stats.items_out += 1
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
                await self._write_lineage()
                await asyncio.gather(*(m.release() for m in self._maps.values()))
                # The calls of this stage are done: a pool whose other stages
                # are done too releases its GPUs for the next phase.
                await asyncio.gather(
                    *(group.flushed.remote(self._label) for group in self._releases),
                )
                return [
                    (route.handle, route.port, FLUSH, None)
                    for route in _walk_routes(self._routes)
                    if isinstance(route, _Push)
                ]
            case _InFlightItem(routing=routing):
                return await routing

    async def _route(
        self,
        routes: List[_Route],
        item: Any,
        lid: ItemId,
    ) -> List[_Delivery]:
        deliveries = await asyncio.gather(
            *(self._route_one(route, item, lid) for route in routes),
        )
        return [delivery for nested in deliveries for delivery in nested]

    async def _route_one(
        self,
        route: _Route,
        item: Any,
        lid: ItemId,
    ) -> List[_Delivery]:
        match route:
            case _Push(handle=handle, port=port):
                return [(handle, port, item, lid)]
            case _Apply(label=label, routes=routes):
                # The result stays in the object store; the next map or the
                # receiving stage resolves it. Its id is derived here, and
                # the call gives the rows it creates element ids from it.
                result_id = map_id(label, lid)
                result, elements = await self._maps[label].run(item, result_id)
                self._record(
                    result_id,
                    "map",
                    (lid,),
                    () if elements is None else elements,
                    stage=label,
                    inherited=elements is None,
                )
                return await self._route(routes, result, result_id)


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

    def _lineage_policy(self, label: str) -> LineagePolicy:
        return SourceLineage(label)

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
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

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
        return [item]


_STOP_ITERATION = object()


@ray.remote
class RayFlatMapActor(_PipelinedStage):
    """Streaming 1:N generator stage (IRFlatMap).

    Pulls one item at a time from ``func(item)`` in a worker thread and
    forwards each yielded item immediately into the outbox. An infinite
    generator (such as ``ImportMode.WATCH``) streams continuously and never
    finishes to forward ``FLUSH``.
    """

    def __init__(
        self,
        func: Callable[[Any], Any],
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        parameters: Mapping[str, Any],
    ):
        configure_worker_env(parameters=parameters)
        super().__init__(config, spill, label)
        self.func = func

    def _lineage_policy(self, label: str) -> LineagePolicy:
        return AccumulatorLineage(label)

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
        iterator = iter(await asyncio.to_thread(self.func, item))
        while True:
            out_item = await asyncio.to_thread(next, iterator, _STOP_ITERATION)
            if out_item is _STOP_ITERATION:
                break

            (out_lid,) = self._lineage.emit(lid, 1)
            fill_created(out_item, out_lid)
            if self._run_store is not None:
                t_lin = time.perf_counter()
                self._record(
                    out_lid,
                    "accumulate",
                    self._lineage.emitted_parents,
                    element_segments(out_item),
                )
                self._stats.lineage_s += time.perf_counter() - t_lin
            self._provenance.emit(item, [out_item])
            await self._outbox.put(_EmitItem(out_item, out_lid))

        return []


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

    def _lineage_policy(self, label: str) -> LineagePolicy:
        return AccumulatorLineage(label)

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
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


@dataclass(frozen=True)
class CallTiming:
    """Where a call of a map ran: how long it waited there and executed.

    Returned next to the result of every call (``num_returns=2``), so the
    map accounts the execution, not the time the call queued in Ray or
    waited for the other call in flight on its executor.
    """

    waited_s: float
    execute_s: float
    # The element ids of the result, for its lineage record; ``None`` if they
    # are the ids of the item of the call.
    elements: Optional[Tuple[ElementSegment, ...]] = ()
    # Time spent finding them.
    lineage_s: float = 0.0


def _run_map_call(
    functions: Ref,
    label: str,
    item: Any,
    lid: ItemId,
    submitted_t: float,
) -> Tuple[Any, CallTiming]:
    """Run one call of a map ``label``, holding the resources of the task.

    The function comes from the pipeline's function table, unpacked once per
    process. Ray may reuse the worker process for other calls or end it, so
    the profiling events of the call are sent before it returns. The call
    waited in Ray's queue since ``submitted_t`` (seconds since the epoch).
    The rows the call creates get element ids from the id ``lid`` of its
    result.
    """
    waited = max(time.time() - submitted_t, 0.0)
    table = table_of(functions)
    _serve_parameters(table.parameters)
    recorder = make_recorder(table.profile, "task worker")
    try:
        with recorder.span("execute", f"map:{label}", gpus=ray.get_gpu_ids()):
            t_start = time.perf_counter()
            result = table.functions[label](item)
            executed = time.perf_counter() - t_start
            fill_created(result, lid)
            return result, _with_lineage(
                CallTiming(waited, executed),
                table.record_lineage,
                item,
                result,
            )
    finally:
        recorder.flush_sync()


def _with_lineage(
    timing: CallTiming,
    record: bool,
    item: Any,
    result: Any,
) -> CallTiming:
    """``timing`` with the element ids of ``result`` for its lineage record,
    if the pipeline records lineage, and the time it took to find them."""
    if not record:
        return timing
    t_start = time.perf_counter()
    elements = call_elements(item, result)
    return dataclasses.replace(
        timing,
        elements=elements,
        lineage_s=time.perf_counter() - t_start,
    )


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

    def call(
        self,
        functions: Ref,
        label: str,
        stage: str,
        item: Any,
        lid: ItemId,
    ) -> Tuple[Any, CallTiming]:
        """Run the function of the map ``label`` from ``functions`` on ``item``.

        The calls of a map of the protocol (``stage``, its fingerprint) run
        one at a time; the copies of a map in the children of a ``group_by``
        are the same map. So an executor runs as many calls at once as the
        protocol has maps, however many children share it. Returns the
        result, whose created rows get element ids from its id ``lid``, and
        its timing (called with ``num_returns=2``).
        """
        table = table_of(functions)
        func = table.functions[label]
        lane = f"map:{label}"
        lock = self._locks.setdefault(stage, threading.Lock())
        t_wait = time.perf_counter()
        with self._recorder.span("lock_wait", lane, overlapping=True):
            lock.acquire()
        try:
            t_start = time.perf_counter()
            with self._recorder.span("execute", lane):
                result = func(item)
            executed = time.perf_counter() - t_start
        finally:
            lock.release()
        # Outside the lock: the next call of the map starts meanwhile.
        fill_created(result, lid)
        return result, _with_lineage(
            CallTiming(t_start - t_wait, executed),
            table.record_lineage,
            item,
            result,
        )

    def prepare(self, functions: Ref) -> Tuple[str, ...]:
        """Build the process-scope resources of the group before the first call.

        Returns, once the executor holds its resources, runs, and e.g. has
        loaded the model of its protocol, the ids of its GPUs.
        """
        with self._recorder.span("prepare", "executor", level=logging.INFO):
            table_of(functions).preparers[self._group]()
        return tuple(str(gpu) for gpu in ray.get_gpu_ids())

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
        # The GPUs the executors of every bundle ran on, kept after release.
        self._devices: Dict[int, Tuple[str, ...]] = {}
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
        gpus: Tuple[str, ...] = ()
        try:
            placement = await self._placement(bundle, generation)
            if placement is not None:
                self._recorder.instant("executor_create", "group")
                executor = RayComputeExecutor.options(
                    max_concurrency=self._members * _EXECUTOR_CALLS_IN_FLIGHT,
                    **self._options,
                    **placement,
                ).remote(self._parameters, self._group, self._profile)
                gpus = await executor.prepare.remote(self._functions)
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
                    self._add_executor(executor, bundle, gpus)
                elif executor is not None:
                    # The pool was released while the executor started.
                    ray.kill(executor)
                self._changed.notify_all()

    def _add_executor(self, executor: Any, bundle: int, gpus: Tuple[str, ...]) -> None:
        """Add a started executor to the pool; call with the lock held."""
        self._executors[self._next_id] = executor
        self._leases[self._next_id] = 0
        self._bundle_of[self._next_id] = bundle
        if bundle >= 0:
            self._devices[bundle] = gpus
        self._next_id += 1

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

    async def devices(self) -> Tuple[str, Dict[int, Tuple[str, ...]]]:
        """The group, and the GPU ids its executors ran on by bundle."""
        return self._group, dict(self._devices)

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

    async def run(
        self,
        item: Any,
        lid: ItemId,
    ) -> Tuple[ray.ObjectRef, Optional[Tuple[ElementSegment, ...]]]:
        """Submit a call, whose result has the id ``lid``, and wait until its
        result is in the object store. Returns the result and its element ids
        (``None``: the ids of ``item``).

        The span of the call covers its queueing for resources and its
        transfer; the worker running it records its execution, and returns
        its timing with the result.
        """
        if self.stats.first_in_t is None:
            self.stats.first_in_t = time.time()
        with self._recorder.span(
            "call",
            f"map:{self._label}",
            overlapping=True,
            seq=self.stats.items_in + 1,
        ):
            result, timing, waited = await self._call(item, lid)

        call = await _timing_of(timing)
        account_call(self.stats, waited, call)
        if self._group is not None:
            self.stats.capacity = max(
                self.stats.capacity or 0,
                len(self._executors.ids),
            )
        if self.provenance.sampling:
            await self._sample(item, result)
        return result, () if call is None else call.elements

    async def _sample(self, item: Any, result: ray.ObjectRef) -> None:
        """Sample the item of a call and its result, fetched from the object
        store (for the first items only, see ``provenance.py``)."""
        received = await item if isinstance(item, ray.ObjectRef) else item
        self.provenance.receive(received)
        self.provenance.emit(received, [await result])

    async def _call(
        self,
        item: Any,
        lid: ItemId,
    ) -> Tuple[ray.ObjectRef, ray.ObjectRef, float]:
        """The result of a call, its timing, and how long it waited for an
        executor of the pool (0 for Ray tasks, which queue inside Ray)."""
        if self._group is None:
            result, timing = self._submit(item, lid)
            return await self._completions.done(result), timing, 0.0
        else:
            _cancel(self._linger)
            t_wait = time.perf_counter()
            async with self._executors.slot() as executor:
                waited = time.perf_counter() - t_wait
                result, timing = executor.call.options(num_returns=2).remote(
                    self._functions,
                    self._label,
                    self._stage,
                    item,
                    lid,
                )
                await self._completions.done(result)
            return result, timing, waited

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

    def _submit(
        self,
        item: Any,
        lid: ItemId,
    ) -> Tuple[ray.ObjectRef, ray.ObjectRef]:
        """Submit a call as a Ray task: the refs of its result and timing."""
        refs = _run_in_task.options(**self._options, num_returns=2).remote(
            self._functions,
            self._label,
            item,
            lid,
            time.time(),
        )
        # The stubs of Ray do not model num_returns.
        return cast(Tuple[ray.ObjectRef, ray.ObjectRef], refs)

    async def _lease(self, held: Sequence[int]) -> Optional[Tuple[int, Any]]:
        assert self._group is not None, "Long-running map has no compute group."
        return await self._group.lease.remote(held)


def account_call(
    stats: StageStats,
    waited: float,
    timing: Optional[CallTiming],
) -> None:
    """Add a call of a map to its stats.

    ``waited`` is how long the call waited for an executor of the pool;
    ``timing`` how long it waited and executed where it ran, or ``None`` if
    the call failed (its error is raised where its result is read).
    """
    waited_there, executed = (
        (0.0, 0.0)
        if timing is None
        else (
            timing.waited_s,
            timing.execute_s,
        )
    )
    stats.items_in += 1
    stats.items_out += 1
    stats.wait_s += waited + waited_there
    stats.wait_max_s = max(stats.wait_max_s, waited + waited_there)
    stats.process_s += executed
    stats.lineage_s += 0.0 if timing is None else timing.lineage_s
    stats.last_done_t = time.time()


async def _timing_of(timing: ray.ObjectRef) -> Optional[CallTiming]:
    """The timing of a call; ``None`` if the call failed."""
    try:
        return await timing
    except RayError:
        # The error of the call reaches the stage reading its result.
        return None


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
    """Terminal stage wrapping a SinkWriter, finalized on FLUSH.

    With a run store, the sink records a ``Commit`` after every write
    (batched, like lineage records), and skips the items committed in the
    store before, by an earlier run: rerunning a pipeline in the store of a
    run writes only what that run did not. Before its first item it lets a
    ``CommittingSinkWriter`` remove the writes no commit covers.

    The sink of a ``group_by`` child hands its results back to the router
    with their ids (``_ResultWriter``); it commits nothing.
    """

    def __init__(
        self,
        writer: SinkWriter,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        unique_elements: UniqueElements = UniqueElements.OFF,
    ):
        super().__init__(config, spill, label)
        self.writer = writer
        # The ids of the items written, in order.
        self._received: List[ItemId] = []
        self._unique_elements = unique_elements
        self._seen = SeenElements()
        self._commits: Optional[_Batches[Commit]] = (
            None
            if config.run_store is None or isinstance(writer, _ResultWriter)
            else _Batches(
                config.run_store.append_commits,
                config.lineage_batch,
                self._stats,
            )
        )
        # The ids committed by earlier runs, read before the first item.
        self._committed: Optional[FrozenSet[ItemId]] = None

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
        if self._commits is not None and self._committed is None:
            await self._recover()
        if self._committed is not None and lid in self._committed:
            self._stats.skipped += 1
            return []

        elements = (
            element_segments(item)
            if self._commits is not None
            or self._unique_elements is not UniqueElements.OFF
            else ()
        )
        self._check_unique(elements, lid)
        receipt = None
        match self.writer:
            case _ResultWriter():
                await self.writer.write_identified(item, lid)
            case CommittingSinkWriter():
                receipt = await self.writer.write_identified(item, lid)
            case _:
                await self.writer.write(item)
        self._received.append(lid)
        if self._commits is not None:
            self._commits.add(
                Commit(lid, self._label, sum(e.count for e in elements), receipt),
            )
        return []

    async def received_ids(self) -> List[ItemId]:
        """The ids of the items written so far, in order."""
        return self._received

    async def on_flush(self) -> List[Any]:
        await self.writer.finalize()
        if self._commits is not None:
            await self._commits.drain()
        return []

    async def _recover(self) -> None:
        """Read the commits of earlier runs; let the writer remove the writes
        after the last one."""
        assert self._run_store is not None
        commits = await asyncio.to_thread(self._run_store.read_commits, self._label)
        self._committed = frozenset(commit.item_id for commit in commits)
        if isinstance(self.writer, CommittingSinkWriter):
            await self.writer.recover(commits[-1].receipt if commits else None)

    def _check_unique(self, elements: Tuple[ElementSegment, ...], lid: ItemId) -> None:
        """Warn about or refuse elements written before, by their ids."""
        if self._unique_elements is UniqueElements.OFF:
            return
        repeated = self._seen.add(elements)
        if repeated == 0:
            return
        message = (
            f"The sink {self._label} writes {repeated} elements a second time, "
            f"in the item {lid.hex()}."
        )
        if self._unique_elements is UniqueElements.RAISE:
            raise ValueError(message)
        else:
            logger.warning(message)


# Items a checkpoint writes at once at most; beyond, it waits before taking
# the next item, so that a slow disk holds up the stream instead of filling
# the memory of the actor.
_CHECKPOINT_WRITES = 8


@ray.remote
class RayCheckpointActor(_PipelinedStage):
    """Stage keeping the items passing through in the run store
    (``IRCheckpoint``).

    An item is forwarded at once; it is written in the background, and
    committed once written, so the commits list the items kept, in order.
    The checkpoint is complete once FLUSH passed it and every item was
    written.
    """

    def __init__(
        self,
        config: _StageConfig,
        spill: Optional[SpillPolicy],
        label: str,
        replay: bool,
    ):
        super().__init__(config, spill, label)
        assert config.run_store is not None, "A checkpoint needs a run store."
        self._store = config.run_store
        self._store.begin_checkpoint(label, replay)
        self._commits: _Batches[Commit] = _Batches(
            self._store.append_commits,
            config.lineage_batch,
            self._stats,
        )
        self._slots = asyncio.Semaphore(_CHECKPOINT_WRITES)
        self._writes: Optional[asyncio.Task[None]] = None
        self._seen = SeenElements()
        self._repeated = 0

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
        elements = element_segments(item)
        self._repeated += self._seen.add(elements)
        await self._slots.acquire()
        self._writes = asyncio.create_task(
            self._write(
                self._writes,
                item,
                Commit(lid, self._label, sum(e.count for e in elements)),
            ),
        )
        return [item]

    async def on_flush(self) -> List[Any]:
        if self._writes is not None:
            await self._writes
        await self._commits.drain()
        await asyncio.to_thread(
            self._store.complete_checkpoint,
            self._label,
            self._repeated,
        )
        return []

    async def _write(
        self,
        previous: Optional[asyncio.Task[None]],
        item: Any,
        commit: Commit,
    ) -> None:
        """Write ``item``, then commit it after the items before it."""
        try:
            t_start = time.perf_counter()
            await asyncio.to_thread(
                self._store.store_item,
                self._label,
                commit.item_id,
                item,
            )
            self._stats.lineage_s += time.perf_counter() - t_start
            if previous is not None:
                await previous
            self._commits.add(commit)
        except Exception as error:
            # Fails the pipeline now rather than at FLUSH.
            self._set_error(error)
            raise
        finally:
            self._slots.release()


class _ResultWriter(abc.ABC):
    """Sink of a ``group_by`` child, handing its results back to the router
    with their ids."""

    def __init__(self, router: Any) -> None:
        self.router = router

    async def write(self, item: Any) -> None:
        raise NotImplementedError(
            "The sink of a group_by child writes its results with their ids "
            "(write_identified).",
        )

    @abc.abstractmethod
    async def write_identified(self, item: Any, lid: ItemId) -> None:
        """Hand the result ``item``, whose id is ``lid``, to the router."""
        ...

    async def finalize(self) -> None:
        pass


class _DemuxResultWriter(_ResultWriter):
    """Sink of the ``group_by`` child of a key."""

    def __init__(self, router: Any, key: Any) -> None:
        super().__init__(router)
        self.key = key

    async def write_identified(self, item: Any, lid: ItemId) -> None:
        await self.router.emit.remote(self.key, item, lid)


class _SharedResultWriter(_ResultWriter):
    """Sink of a ``group_by`` child shared by several keys (see ``share_keys``).

    Its results arrive as ``Keyed(key, result)``.
    """

    async def write_identified(self, item: Any, lid: ItemId) -> None:
        assert isinstance(item, Keyed), "A shared group_by child lost the key."
        await self.router.emit.remote(item.key, item.value, lid)


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
        self._queue: asyncio.Queue[Tuple[Any, Optional[ItemId], Optional[_Barrier]]] = (
            asyncio.Queue(maxsize=0 if buffer_size is None else buffer_size)
        )
        self._feeder = asyncio.create_task(self._feed())

    async def send(self, item: Any, lid: ItemId) -> None:
        """Queue an item, whose id is ``lid``, for the child, waiting while
        its queue is full."""
        await self._queue.put((item, lid, None))

    async def flush(self) -> None:
        """Flush the child after all queued items; returns once it has drained."""
        done: _Barrier = asyncio.get_running_loop().create_future()
        await self._queue.put((FLUSH, None, done))
        await done

    async def _feed(self) -> None:
        error: Optional[BaseException] = None
        while True:
            item, lid, done = await self._queue.get()
            barriers = [] if done is None else [done]
            if error is not None:
                _fail(barriers, error)
                continue

            try:
                child = await self.ready
                await _deliver(child.source_handle(self._source_name), item, 0, lid)
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

    async def process(self, item: Any, port: int, lid: ItemId) -> List[Any]:
        if self.outer_keyed:
            outer, value = item
            key = (outer, await asyncio.to_thread(self.key_fn, value))
        else:
            value = item
            key = await asyncio.to_thread(self.key_fn, value)

        if key not in self._keys:
            self._keys[key] = self._assign(key)

        child_item = value if self.workers is None else Keyed(key=key, value=value)
        await self._keys[key].send(child_item, lid)
        return []

    async def on_flush(self) -> List[Any]:
        await asyncio.gather(*(child.flush() for child in self._children.values()))
        return []

    async def emit(self, key: Any, item: Any, lid: ItemId) -> None:
        """Forward a result of the child pipeline of ``key``, with its id."""
        if self.outer_keyed:
            outer, inner = key
            keyed = Keyed(key=outer, value=Keyed(key=inner, value=item))
        else:
            keyed = Keyed(key=key, value=item)
        await self._outbox.put(_EmitItem(keyed, lid))

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
