from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import abc
import asyncio
import os
import sys
import time
import uuid
import ray

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
    Tagged,
)
from scipion_bridge.core.streaming.node import FlushSignal, FLUSH
from scipion_bridge.core.streaming.sink_writer import SinkWriter

_Barrier = asyncio.Future[None]
_InboxEntry = Tuple[Any, int, Optional[_Barrier]]
_OutboxEntry = Tuple[Any, List[_Barrier]]


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
    - A single emitter task forwards results downstream in order, so the
      compute task is never stalled by serialization or downstream stages.

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
        self._downstream: List[Tuple[Any, int]] = []
        self._abort_targets: List[Any] = []
        self._num_inputs = 1
        self._pending_flushes: List[_Barrier] = []
        self._error: Optional[BaseException] = None
        self._tasks: List[asyncio.Task[None]] = []
        self._stats = StageStats()

    async def connect(
        self,
        downstream: List[Tuple[Any, int]],
        abort_targets: List[Any],
        num_inputs: int,
    ) -> None:
        """Configure the stage and start its loops.

        Args:
            downstream: Downstream actor handles, each with the port this stage
                feeds on it.
            abort_targets: Actor handles aborted when this stage fails: the
                sources of the pipeline and, for a nested pipeline, the
                sources of its parent.
            num_inputs: Number of upstream stages, each sending one FLUSH.
        """
        self._downstream = downstream
        self._abort_targets = abort_targets
        self._num_inputs = num_inputs
        self._tasks = [
            asyncio.create_task(self._run_compute()),
            asyncio.create_task(self._run_emit()),
        ]

    async def stats(self) -> StageStats:
        """Return the execution metrics of this stage."""
        return self._stats

    async def abort(self, error: BaseException) -> None:
        """Fail this stage with an error raised elsewhere in the pipeline."""
        if self._error is None:
            self._store_error(error)

    def _set_error(self, error: BaseException) -> None:
        """Fail this stage and abort the abort targets on the first error."""
        if self._error is not None:
            return

        self._store_error(error)
        for target in self._abort_targets:
            target.abort.remote(error)

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

            t_start = time.perf_counter()
            try:
                await asyncio.gather(
                    *(h.push.remote(item, port) for h, port in self._downstream)
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


@ray.remote
class RaySourceActor(_PipelinedStage):
    """Input source of the streaming pipeline, forwarding items unchanged."""

    def __init__(self, name: str, queue_size: int):
        super().__init__(queue_size)
        self.name = name

    async def process(self, item: Any, port: int) -> List[Any]:
        return [item]


@ray.remote
class RayWorkerActor(_PipelinedStage):
    """Transformation stage (IRMap) of the streaming pipeline."""

    def __init__(
        self,
        func: Callable[[Any], Any],
        queue_size: int,
        parameters: Mapping[str, Any],
    ):
        from .container import configure_ray_env

        configure_ray_env(parameters=parameters)
        super().__init__(queue_size)
        self.func = func

    async def process(self, item: Any, port: int) -> List[Any]:
        return [await asyncio.to_thread(self.func, item)]


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


class RayCompiledPipeline(CompiledPipeline):
    """
    Executable compiled streaming pipeline running on Ray.
    """

    def __init__(
        self,
        sources: Dict[str, Any],
        stages: Dict[str, Any],
    ):
        """
        Args:
            sources: Source actors keyed by input name.
            stages: All actors of the pipeline keyed by a readable stage label,
                in topological order.
        """
        self._sources = sources
        self._stages = stages

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
        """Return execution metrics per stage, in topological order."""
        stats = ray.get([actor.stats.remote() for actor in self._stages.values()])
        return dict(zip(self._stages, stats))

    def close(self) -> None:
        """Terminate all actors allocated for this pipeline."""
        for actor in self._stages.values():
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
    ):
        """
        Args:
            init_ray: Initialize a local Ray instance if none is running.
            parameters: Values of the protocol parameters, served to
                ``Field.value`` inside the worker stages.
            queue_size: Number of items each stage buffers in its inbox and
                outbox. Stages run concurrently while their queues have room;
                the default of 2 double-buffers every stage.
        """
        if queue_size <= 0:
            raise ValueError(f"Queue size must be positive, got {queue_size}.")
        self.queue_size = queue_size
        self.parameters = dict(parameters or {})

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

        # 2. Instantiate Ray actors for every IR node. Actors are named so that
        # stages can be identified in the Ray dashboard; the pipeline id keeps
        # the names unique within the Ray namespace.
        pipeline_id = uuid.uuid4().hex[:8]
        actor_map: Dict[IROp, Any] = {}
        for index, node in enumerate(all_nodes):
            match node:
                case IRSource(name=source_name):
                    actor_cls: Any = RaySourceActor
                    kwargs: Dict[str, Any] = {
                        "name": source_name,
                        "queue_size": self.queue_size,
                    }
                case IRMap(func=func):
                    actor_cls = RayWorkerActor
                    kwargs = {
                        "func": func,
                        "queue_size": self.queue_size,
                        "parameters": self.parameters,
                    }
                case IRAccumulate(
                    accumulate_fn=accumulate_fn,
                    initial_state_fn=initial_state_fn,
                    flush_fn=flush_fn,
                    tag_inputs=tag_inputs,
                ):
                    actor_cls = RayAccumulatorActor
                    kwargs = {
                        "accumulate_fn": accumulate_fn,
                        "initial_state_fn": initial_state_fn,
                        "queue_size": self.queue_size,
                        "parameters": self.parameters,
                        "flush_fn": flush_fn,
                        "tag_inputs": tag_inputs,
                    }
                case IRSink(writer=writer):
                    actor_cls = RaySinkActor
                    kwargs = {
                        "writer": writer,
                        "queue_size": self.queue_size,
                    }
                case _:
                    raise NotImplementedError(
                        f"Unsupported IR op type for Ray backend: {type(node).__name__}",
                    )

            actor_map[node] = actor_cls.options(
                name=f"{pipeline_id}:{name}{index}:{_describe(node)}",
            ).remote(**kwargs)

        compiled_sources = {
            source_name: actor_map[src_node]
            for source_name, src_node in sources_map.items()
        }

        # 3. Wire downstream and source actor handles and start the stage loops.
        # Every stage feeds a downstream stage on the port given by its position
        # among the upstream stages of the downstream stage. A failing stage
        # aborts the sources and the further abort targets.
        abort_handles = [*compiled_sources.values(), *abort_targets]
        ray.get(
            [
                actor.connect.remote(
                    [(actor_map[d], d.upstream.index(node)) for d in node.downstream],
                    abort_handles,
                    max(1, len(node.upstream)),
                )
                for node, actor in actor_map.items()
            ]
        )

        return RayCompiledPipeline(
            sources=compiled_sources,
            stages={
                f"{name}{index}:{_describe(node)}": actor_map[node]
                for index, node in enumerate(all_nodes)
            },
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
        case _:
            return type(node).__name__
