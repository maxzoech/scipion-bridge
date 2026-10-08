from __future__ import annotations
from dataclasses import dataclass
from functools import cache
from typing import (
    Any,
    Callable,
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
import subprocess
import sys
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
    routes: List[_Route]


_Route = Union[_Push, _Apply]


def _fail(barriers: List[_Barrier], error: BaseException) -> None:
    """Fail pending FLUSH barriers."""
    for barrier in barriers:
        barrier.set_exception(error)


class _PipelinedStage(abc.ABC):
    """Pipelined stage of a compiled Ray pipeline.

    Every stage decouples receiving, computing and forwarding items so that
    consecutive stages run concurrently:

    - ``push`` only enqueues into a bounded inbox. It blocks while the inbox is
      full, which propagates backpressure upstream to the driver.
    - A single compute task processes items one at a time in arrival order, so
      stage logic (e.g. a GPU model or accumulator state) never runs
      concurrently.
    - A single emitter task routes results downstream, so the compute task
      is never stalled by serialization or downstream stages. Results pass
      the maps on their routes (Ray tasks) with up to ``queue_size`` items in
      flight; a forwarder task pushes them on in order.

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

    def __init__(self, queue_size: int) -> None:
        self._inbox: asyncio.Queue[_InboxEntry] = asyncio.Queue(maxsize=queue_size)
        self._outbox: asyncio.Queue[_OutboxEntry] = asyncio.Queue(maxsize=queue_size)
        self._in_flight: asyncio.Queue[_InFlightEntry] = asyncio.Queue(
            maxsize=queue_size,
        )
        self._routes: List[_Route] = []
        self._maps: Dict[str, _MapRunner] = {}
        self._abort_targets: List[Any] = []
        self._num_inputs = 1
        self._pending_flushes: List[_Barrier] = []
        self._error: Optional[BaseException] = None
        self._tasks: List[asyncio.Task[None]] = []
        self._stats = StageStats()

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
        self._maps = {
            route.label: _MapRunner(route)
            for route in _walk_routes(routes)
            if isinstance(route, _Apply)
        }
        self._abort_targets = abort_targets
        self._num_inputs = num_inputs
        self._tasks = [
            asyncio.create_task(self._run_compute()),
            asyncio.create_task(self._run_emit()),
            asyncio.create_task(self._run_forward()),
        ]

    async def stats(self) -> StageStats:
        """Return the execution metrics of this stage."""
        return self._stats

    async def map_stats(self) -> Dict[str, StageStats]:
        """Return the execution metrics of the maps on the routes, by label."""
        return {label: runner.stats for label, runner in self._maps.items()}

    async def abort(self, failure: _Failure) -> None:
        """Fail this stage with an error raised elsewhere in the pipeline."""
        if self._error is None:
            self._store_error(failure.error)

    def _set_error(self, error: BaseException) -> None:
        """Fail this stage and abort the abort targets on the first error."""
        if self._error is not None:
            return

        self._store_error(error)
        for target in self._abort_targets:
            target.abort.remote(_Failure(error))

    def _store_error(self, error: BaseException) -> None:
        self._error = error
        _fail(self._pending_flushes, error)
        self._pending_flushes = []

    async def push(self, item: Any, port: int = 0) -> None:
        """Enqueue an item; for FLUSH, wait until the pipeline has drained."""
        if self._error is not None:
            raise self._error

        match item:
            case FlushSignal():
                done: _Barrier = asyncio.get_running_loop().create_future()
                await self._inbox.put((item, port, done))
                await done
            case _:
                await self._inbox.put((item, port, None))

    @abc.abstractmethod
    async def process(self, item: Any, port: int) -> List[Any]:
        """Process an item arriving on ``port`` and return the items to emit."""
        ...

    async def on_flush(self) -> List[Any]:
        """Finalize the stage on FLUSH and return the items to emit downstream."""
        return []

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

                case _:
                    self._stats.items_in += 1

            try:
                emissions = await self._compute(item, port)
            except Exception as error:
                self._set_error(error)
                _fail(barriers, error)
                continue

            t_processed = time.perf_counter()
            self._stats.process_s += t_processed - t_start

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
                    # The maps of the item start right away; the queue bounds
                    # the items in flight.
                    routing = asyncio.create_task(self._route(self._routes, item))
                    await self._in_flight.put((routing, []))
                case _:
                    await self._in_flight.put((None, barriers))

    async def _run_forward(self) -> None:
        while True:
            routing, barriers = await self._in_flight.get()
            if self._error is not None:
                _fail(barriers, self._error)
                _cancel(routing)
                continue

            t_start = time.perf_counter()
            try:
                deliveries = await self._finish(routing)
                await asyncio.gather(
                    *(
                        handle.push.remote(item, port)
                        for handle, port, item in deliveries
                    )
                )
            except Exception as error:
                self._set_error(error)
                _fail(barriers, error)
                continue

            match barriers:
                case []:
                    self._stats.emit_s += time.perf_counter() - t_start
                    self._stats.items_out += 1
                case _:
                    for barrier in barriers:
                        barrier.set_result(None)

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

    def __init__(self, name: str, queue_size: int):
        super().__init__(queue_size)
        self.name = name

    async def process(self, item: Any, port: int) -> List[Any]:
        return [item]


@ray.remote
class RayMergeActor(_PipelinedStage):
    """Joins the inputs of a map with several upstream stages.

    The items pass unchanged; the map is applied on the route of the merged
    stream, after the FLUSH of every input arrived.
    """

    def __init__(self, queue_size: int):
        super().__init__(queue_size)

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
        queue_size: int,
        parameters: Mapping[str, Any],
        flush_fn: Optional[Callable[[Any], tuple[Any, List[Any]]]] = None,
        tag_inputs: bool = False,
    ):
        from .container import configure_ray_env

        configure_ray_env(parameters=parameters)
        super().__init__(queue_size)
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

    Ray reuses its worker processes for many map calls.
    """
    from .container import configure_ray_env

    global _served_parameters
    match _served_parameters == dict(parameters):
        case True:
            pass
        case False:
            configure_ray_env(parameters=parameters)
            _served_parameters = dict(parameters)


@ray.remote
def _run_in_task(
    func: Callable[[Any], Any],
    parameters: Mapping[str, Any],
    item: Any,
) -> Any:
    """Run one call of a map, holding the resources of the task."""
    _serve_parameters(parameters)
    return func(item)


@cache
def _cluster_cpus() -> int:
    return int(ray.cluster_resources().get("CPU", 1))


def _ray_options(resources: ComputeResources, num_gpus: float) -> Dict[str, Any]:
    """Ray options of a task or actor holding ``resources``.

    ``num_gpus`` is the GPU claim resolved from ``gpus`` and ``min_vram``.

    Without reserved CPUs, the process may use all cores of the machine and
    the OS schedules them. Ray would set ``OMP_NUM_THREADS`` to the number of
    reserved cores, making NumPy, BLAS and PyTorch single-threaded, so it is
    set to all cores instead. With reserved cores, user code learns their
    number from ``CPUS_ENV_VAR`` (e.g. to size the pools of map_element).
    """
    match resources.cpus:
        case None:
            return {
                "num_gpus": num_gpus,
                "num_cpus": 0,
                "runtime_env": {
                    "env_vars": {"OMP_NUM_THREADS": str(_cluster_cpus())},
                },
            }
        case cpus:
            return {
                "num_gpus": num_gpus,
                "num_cpus": cpus,
                "runtime_env": {"env_vars": {CPUS_ENV_VAR: str(cpus)}},
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


@ray.remote
class RayComputeExecutor:
    """Process holding the resources of a long-running group of maps.

    Every map of the group runs its calls here, so the maps share the
    resources and the process-scope resources of the protocol (e.g. a model
    on the GPU) are built once. A threaded actor: the maps call concurrently.
    """

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        _serve_parameters(parameters)

    def call(self, func: Callable[[Any], Any], item: Any) -> Any:
        return func(item)


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
    ) -> None:
        self._options = options
        self._members = members
        self._parameters = parameters
        self._executor: Optional[Any] = None
        self._released = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Any:
        """Return the executor of the group, creating it if necessary."""
        async with self._lock:
            if self._executor is None:
                self._executor = RayComputeExecutor.options(
                    max_concurrency=self._members,
                    **self._options,
                ).remote(self._parameters)
            return self._executor

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
                    ray.kill(executor)


class _MapRunner:
    """Runs the calls of a map on a route, with the map's compute resources.

    - Ephemeral resources (and maps without resources) are taken by a Ray
      task per call, so Ray queues the calls of all maps on the resources.
    - Long-running resources are held by the executor of the map's group.
      Its calls run one at a time, as in a stage of its own.
    """

    def __init__(self, route: _Apply) -> None:
        self._func = route.func
        self._compute = route.compute
        self._options = route.options
        self._group = route.group
        self._parameters = route.parameters
        self.stats = StageStats()
        self._lock = asyncio.Lock()
        # The executor stays alive until this map released it on FLUSH.
        self._executor: Optional[Any] = None

    async def run(self, item: Any) -> ray.ObjectRef:
        """Submit a call and wait until its result is in the object store."""
        t_start = time.perf_counter()
        match self._compute:
            case ComputeAssignment(
                resources=ComputeResources(task=TaskType.LONG_RUNNING),
            ):
                async with self._lock:
                    executor = await self._acquire()
                    result = await _done(executor.call.remote(self._func, item))
            case _:
                result = await _done(self._submit(item))

        self.stats.items_in += 1
        self.stats.items_out += 1
        self.stats.process_s += time.perf_counter() - t_start
        return result

    async def release(self) -> None:
        """Release the long-running resources of the map on FLUSH."""
        match self._group:
            case None:
                pass
            case group:
                self._executor = None
                await group.release.remote()

    def _submit(self, item: Any) -> ray.ObjectRef:
        return _run_in_task.options(**self._options).remote(
            self._func,
            self._parameters,
            item,
        )

    async def _acquire(self) -> Any:
        assert self._group is not None, "Long-running map has no compute group."
        if self._executor is None:
            self._executor = await self._group.acquire.remote()
        return self._executor


async def _done(result: ray.ObjectRef) -> ray.ObjectRef:
    """Wait for a result without fetching it into this process."""
    await asyncio.to_thread(ray.wait, [result], fetch_local=False)
    return result


@ray.remote
class RaySinkActor(_PipelinedStage):
    """Terminal stage wrapping a SinkWriter, finalized on FLUSH."""

    def __init__(self, writer: SinkWriter, queue_size: int):
        super().__init__(queue_size)
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


@ray.remote
class RayDemuxActor(_PipelinedStage):
    """Router of a ``group_by`` (IRDemux) into a child pipeline per key.

    The child pipeline of a key is compiled from a clone of the template when
    the first item of the key arrives. Its sink hands every result back to
    ``emit``, which forwards it as ``Keyed(key, result)``:

    - The compute task pushes into the inbox of a child, whose sink waits for
      room in the outbox of the router. The outbox is drained by the emitter
      task, independently of the compute task, so the two never wait on each
      other.
    - On FLUSH, the router flushes every child before it forwards the FLUSH.
      A child returns from its flush only after its sink handed back all
      results, so they precede the FLUSH downstream.
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
        queue_size: int,
        parameters: Mapping[str, Any],
        gpu_memory: Optional[Tuple[float, ...]] = None,
    ):
        from .container import configure_ray_env

        configure_ray_env(parameters=parameters)
        super().__init__(queue_size)
        self.gpu_memory = gpu_memory
        self.key_fn = key_fn
        self.template = template
        self.source_name = source_name
        self.max_keys = max_keys
        self.prefix = prefix
        self.queue_size = queue_size
        self.parameters = parameters
        self._children: Dict[Any, RayCompiledPipeline] = {}

    async def process(self, item: Any, port: int) -> List[Any]:
        key = await asyncio.to_thread(self.key_fn, item)
        if key not in self._children:
            self._children[key] = await self._spawn(key)

        await self._children[key].source_handle(self.source_name).push.remote(item)
        return []

    async def on_flush(self) -> List[Any]:
        await asyncio.gather(
            *(
                child.source_handle(self.source_name).push.remote(FLUSH)
                for child in self._children.values()
            ),
        )
        return []

    async def emit(self, key: Any, item: Any) -> None:
        """Forward a result of the child pipeline of ``key``."""
        await self._outbox.put((Keyed(key=key, value=item), []))

    async def children_stats(self) -> Dict[str, StageStats]:
        """Return the execution metrics of every stage of the children."""
        stats = await asyncio.gather(
            *(asyncio.to_thread(child.stats) for child in self._children.values()),
        )
        return {label: stage for child in stats for label, stage in child.items()}

    async def _spawn(self, key: Any) -> RayCompiledPipeline:
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
        backend = RayBackend(
            init_ray=False,
            queue_size=self.queue_size,
            parameters=self.parameters,
            gpu_memory=self.gpu_memory,
        )
        # Compiling waits for the child actors; a thread keeps the event loop,
        # and with it the emitter, running meanwhile.
        return await asyncio.to_thread(
            backend.compile,
            [sink],
            name=f"{self.prefix}group_by[{key}]:",
            abort_targets=self._abort_targets,
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
        """
        self._sources = sources
        self._stages = stages
        self._demuxes = list(demuxes)
        self._groups = list(groups)

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
        """Push an item into a named input source."""
        ray.get(self.source_handle(source_name).push.remote(value))

    def flush(self) -> None:
        """Flush all sources in parallel and wait for pipeline completion."""
        ray.get([src.push.remote(FLUSH) for src in self._sources.values()])

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
        the executors of compute groups with their coordinator.
        """
        hosts = {node: actor for node, actor in self._stages.values()}
        for actor in [*hosts.values(), *self._groups]:
            try:
                ray.kill(actor)
            except Exception:
                pass


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
    ):
        """
        Args:
            init_ray: Initialize a local Ray instance if none is running.
            parameters: Values of the protocol parameters, served to
                ``Field.value`` inside the worker stages.
            gpu_memory: Memory in GiB of the cluster's GPUs, for claims of
                ``min_vram``. Probed on the cluster when a map needs it if not
                given; ``group_by`` children receive it from their parent.
            queue_size: Number of items each stage buffers in its inbox and
                outbox. Stages run concurrently while their queues have room;
                the default of 2 double-buffers every stage.
        """
        if queue_size <= 0:
            raise ValueError(f"Queue size must be positive, got {queue_size}.")
        self.queue_size = queue_size
        self.parameters = dict(parameters or {})
        self._gpu_memory = None if gpu_memory is None else tuple(gpu_memory)

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
        groups = self._compute_groups(all_nodes, claims)
        actor_map: Dict[IROp, Any] = {
            node: actor_cls.options(name=f"{pipeline_id}:{labels[node]}").remote(
                **kwargs,
            )
            for node in all_nodes
            if not _is_hosted(node)
            for actor_cls, kwargs in [self._stage_actor(node, name)]
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
        builder = _RouteBuilder(actor_map, labels, groups, claims, self.parameters)
        abort_handles = [*compiled_sources.values(), *abort_targets]
        ray.get(
            [
                actor.connect.remote(
                    builder.stage_routes(node),
                    abort_handles,
                    max(1, len(node.upstream)),
                )
                for node, actor in actor_map.items()
            ]
        )

        return RayCompiledPipeline(
            sources=compiled_sources,
            stages={
                labels[node]: (_host_of(node), actor_map[_host_of(node)])
                for node in all_nodes
            },
            demuxes=[
                actor_map[node] for node in all_nodes if isinstance(node, IRDemux)
            ],
            groups=list(groups.values()),
        )

    def _stage_actor(self, node: IROp, name: str) -> Tuple[Any, Dict[str, Any]]:
        """The actor class of a stage, with its constructor arguments."""
        match node:
            case IRMap():
                return RayMergeActor, {"queue_size": self.queue_size}
            case IRSource(name=source_name):
                return RaySourceActor, {
                    "name": source_name,
                    "queue_size": self.queue_size,
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
                    "queue_size": self.queue_size,
                    "parameters": self.parameters,
                    "flush_fn": flush_fn,
                    "tag_inputs": tag_inputs,
                }
            case IRSink(writer=writer):
                return RaySinkActor, {
                    "writer": writer,
                    "queue_size": self.queue_size,
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
                    "queue_size": self.queue_size,
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
            group: RayComputeGroup.remote(options[group], count, self.parameters)
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
