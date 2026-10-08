"""Declarative streaming operations and DAG nodes."""

from __future__ import annotations

import operator
from collections.abc import Hashable, Iterable, Mapping
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set as PySet,
    Tuple,
    TypeVar,
    overload,
)

from ..environment.compute import ComputeAssignment
from ..struct import Set, concat
from .node import Node, LoweringContext
from .sink import Sink
from .sink_writer import SinkWriter, CallbackSinkWriter
from .ir import IROp, IRSource, IRMap, IRAccumulate, IRDemux, Tagged
from .element_mapper import (
    ElementMapConfig,
    ElementMapper,
    Executor,
    StartMethod,
    Workers,
)

_NodeT = TypeVar("_NodeT", bound=Node)
K = TypeVar("K", bound=Hashable)


class Op(Node):
    """Intermediate operation node that allows chaining downstream operations."""

    def op(self, node: _NodeT) -> _NodeT:
        """Connect a downstream node to this op."""
        node.upstream.append(self)
        self.downstream.append(node)
        return node

    def map_batch(
        self,
        func: Callable[[Any], Any],
        *,
        cpu_only: bool = False,
    ) -> MapOp:
        """Transform entire incoming stream item / batch (1:1).

        Every map is executed as its own pipeline stage, and consecutive stages
        process different items concurrently. Splitting a step into separate
        maps therefore overlaps its parts, e.g. CPU post-processing of one
        batch with the GPU forward pass of the next::

            particles.chunk(256).map(forward).map(build_metadata)

        Args:
            func: Function applied to each item.
            cpu_only: Run without the GPUs of the protocol's compute
                resources (``@resources``), e.g. for cheap bookkeeping maps
                that should not wait for a GPU.
        """
        return self.op(MapOp(func, cpu_only=cpu_only))

    def map(self, func: Callable[[Any], Any], *, cpu_only: bool = False) -> MapOp:
        """Alias for map_batch."""
        return self.map_batch(func, cpu_only=cpu_only)

    def map_element(
        self,
        func: Callable[[Any], Any],
        *,
        workers: Workers = "auto",
        executor: Executor = "thread",
        start_method: Optional[StartMethod] = None,
        chunksize: Optional[int] = None,
        cpu_only: bool = False,
    ) -> MapElementOp:
        """Apply ``func`` to every element of the incoming collections, in parallel.

        For each collection ``col`` (any collection with ``__len__``,
        ``__getitem__`` and ``__setitem__``) this runs
        ``col[i] = func(col[i])`` for all ``i`` on a thread pool and forwards
        the same, modified collection. ``func`` may modify the element in
        place (e.g. a Set row view) or return a new value. Every index is
        processed by exactly one worker.

        Use it after ``.chunk(n)`` to preprocess elements on the CPU in a
        stage of its own::

            particles.chunk(256).map_element(preprocess).map(forward)

        Args:
            func: Function applied to each element.
            workers: Number of workers, or ``"auto"`` for one worker per
                element capped at the available CPUs.
            executor: ``"thread"`` (default) or ``"process"``. Threads are safe
                next to CUDA/JAX and scale when ``func`` releases the GIL
                (NumPy, PyTorch, OpenCV); processes help for pure-Python work.
            start_method: Process start method (``"spawn"`` by default; only
                with ``executor="process"``). Avoid ``"fork"`` in processes that
                initialized CUDA or JAX.
            chunksize: Consecutive indices per pool task.
            cpu_only: Run without the GPUs of the protocol's compute
                resources (``@resources``).
        """
        return self.op(
            MapElementOp(
                func,
                ElementMapConfig(
                    workers=workers,
                    executor=executor,
                    start_method=start_method,
                    chunksize=chunksize,
                ),
                cpu_only=cpu_only,
            )
        )

    def chunk(self, n: int, drop_last: bool = False) -> ChunkOp:
        """Accumulate Set[T] instances into batches of target size `n`.

        Args:
            n: Target chunk size (number of elements in the output Set). Must be > 0.
            drop_last: If True, any partial remainder Set smaller than `n` upon
                stream completion (FlushSignal) is dropped. This prevents downstream
                JIT-compiled models from triggering recompilations for a non-standard
                batch size.
        """
        return self.op(ChunkOp(n=n, drop_last=drop_last))

    def collect(self, n: int) -> CollectOp:
        """Collect the first `n` elements of a stream of Set[T] into a single Set.

        The collected Set is emitted once, as soon as `n` elements have arrived;
        all later items are ignored. If the stream ends (FlushSignal) before `n`
        elements arrived, the elements collected so far are emitted instead.

        Use it to train a model on an initial sample of the stream::

            model = particles.collect(5_000).map(train)

        Args:
            n: Number of elements to collect. Must be > 0.
        """
        return self.op(CollectOp(n=n))

    def flatten(self) -> FlattenOp:
        """Emit every element of incoming iterables as a separate item (1:N).

        Accepts any iterable, e.g. lists, tuples, generators or a Collection,
        which yields its initialized items in index order. Strings, bytes and
        mappings are rejected, as iterating them yields characters or keys.

        A Set is a batch of rows, not an iterable, and is rejected as well:
        sending its rows as separate items would be much more expensive than
        sending the batch. Use ``chunk`` to change batch sizes instead.
        """
        return self.op(FlattenOp())

    def combine_latest(self, other: Op) -> CombineLatestOp:
        """Pair every item of this stream with the latest item of `other`.

        Emits `(item, latest)` for every item of this stream, where `latest` is
        the most recent item of `other`. Items arriving before `other` produced
        its first item are buffered and emitted once it has. Items of `other`
        only update `latest` and emit nothing themselves. Items still buffered
        at the end of the stream (FlushSignal) are dropped, as `other` never
        produced an item to pair them with.

        Use it to apply a model trained on a sample of the stream to the whole
        stream::

            model = particles.collect(5_000).map(train)
            particles.chunk(256).combine_latest(model).map(predict)

        Args:
            other: Stream providing the latest value. Must be a different stream
                than this one.
        """
        if other is self:
            raise ValueError("combine_latest() requires two different streams.")

        node = self.op(CombineLatestOp())
        other.op(node)
        return node

    @overload
    def group_by(
        self,
        key: int | str,
        pipeline: Callable[[Op], Op],
        *,
        max_keys: Optional[int] = None,
    ) -> KeyedOp[Any]: ...

    @overload
    def group_by(
        self,
        key: Callable[[Any], K],
        pipeline: Callable[[Op], Op],
        *,
        max_keys: Optional[int] = None,
    ) -> KeyedOp[K]: ...

    def group_by(
        self,
        key: int | str | Callable[[Any], Any],
        pipeline: Callable[[Op], Op],
        *,
        max_keys: Optional[int] = None,
    ) -> KeyedOp[Any]:
        """Run ``pipeline`` separately on the items of every key (demux).

        Every item is routed by its key into a pipeline of its own, so that
        stateful operations (``chunk``, ``collect``, ``combine_latest``) only
        see the items of one key. The pipeline of a key is created when its
        first item arrives. Results are emitted as ``Keyed(key, result)``;
        call ``unkey()`` to continue with the merged stream::

            classes.flatten()
                .group_by(lambda cls: cls.class_id, pipeline=refine)
                .unkey()
                .map(write_class)

        Args:
            key: Index or field name selecting the key of an item
                (``item[key]``), or a function computing it. Keys must be
                hashable.
            pipeline: Builds the pipeline of one key from its input stream.
                It may only consume that input, and every branch must lead to
                the stream it returns.
            max_keys: Maximum number of keys; a further key fails the
                pipeline. Every key allocates the stages of its own pipeline.
        """
        if max_keys is not None and max_keys <= 0:
            raise ValueError(f"max_keys must be positive, got {max_keys}.")

        entry = Source(_GROUP_INPUT)
        exit_ = pipeline(entry)
        if not isinstance(exit_, Op):
            raise TypeError(
                "The group_by pipeline must return a stream (Op), got "
                f"'{type(exit_).__name__}'.",
            )

        dangling = _branches_off(exit_)
        if dangling:
            raise ValueError(
                "Every branch of a group_by pipeline must lead to the stream it "
                "returns; found branches ending in "
                f"{[type(node).__name__ for node in dangling]}.",
            )

        return self.op(KeyedOp(_make_key_extractor(key), entry, exit_, max_keys))

    def write_to(self, writer: SinkWriter) -> Sink:
        """Attach a terminal SinkWriter."""
        sink_node = Sink(writer)
        self.op(sink_node)
        return sink_node

    def checkpoint(self, writer: SinkWriter) -> Op:
        """Attach an asynchronous persistence checkpoint without cutting off the stream."""
        sink_node = Sink(writer)
        self.op(sink_node)
        return self

    def sink(self, callback: Callable[[Any], Any]) -> Sink:
        """Attach a callback-based sink (convenience for testing/debugging)."""
        return self.write_to(CallbackSinkWriter(callback))


class Source(Op):
    """Entry point input stream node."""

    def __init__(self, name: str):
        super().__init__(upstream=[])
        self.name = name

    def lower(self, ctx: LoweringContext) -> IROp:
        # Source nodes with the same name are one input of the pipeline, e.g.
        # when a protocol accesses the same input several times.
        if self.name in ctx.sources:
            return ctx.sources[self.name]

        ir = IRSource(name=self.name)
        ctx.sources[self.name] = ir
        return ir


class MapOp(Op):
    """1:1 batch/element mapping operation node."""

    def __init__(self, func: Callable[[Any], Any], *, cpu_only: bool = False):
        super().__init__(upstream=None)
        self.func = func
        self.cpu_only = cpu_only
        # Set by the protocol whose steps created this op.
        self.compute: Optional[ComputeAssignment] = None

    def lower(self, ctx: LoweringContext) -> IROp:
        func_name = getattr(self.func, "__qualname__", type(self.func).__name__)
        return IRMap(func=self.func, name=f"map({func_name})", compute=self.compute)


class MapElementOp(Op):
    """Operation node applying a function to every element of a collection in parallel."""

    def __init__(
        self,
        func: Callable[[Any], Any],
        config: ElementMapConfig,
        *,
        cpu_only: bool = False,
    ):
        super().__init__(upstream=None)
        self.func = func
        self.config = config
        self.cpu_only = cpu_only
        # Set by the protocol whose steps created this op.
        self.compute: Optional[ComputeAssignment] = None

    def lower(self, ctx: LoweringContext) -> IROp:
        # A stateless map: the worker pools are cached per process.
        func_name = getattr(self.func, "__qualname__", type(self.func).__name__)
        return IRMap(
            func=ElementMapper(self.func, self.config),
            name=f"map_element({func_name})",
            compute=self.compute,
        )


def _make_set_chunk_accumulator(
    n: int,
    drop_last: bool = False,
) -> Tuple[
    Callable[
        [Tuple[List[Set[Any]], int], Any],
        Tuple[Tuple[List[Set[Any]], int], List[Set[Any]]],
    ],
    Callable[[], Tuple[List[Set[Any]], int]],
    Callable[
        [Tuple[List[Set[Any]], int]],
        Tuple[Tuple[List[Set[Any]], int], List[Set[Any]]],
    ],
]:
    def initial_state() -> Tuple[List[Set[Any]], int]:
        return ([], 0)

    def accumulate(
        state: Tuple[List[Set[Any]], int],
        item: Any,
    ) -> Tuple[Tuple[List[Set[Any]], int], List[Set[Any]]]:
        if not isinstance(item, Set):
            raise TypeError(
                f"ChunkOp expected an instance of Set, got '{type(item).__name__}'.",
            )
        if len(item) == 0:
            return (state, [])

        buffer, buffered_count = state
        new_buffer = buffer + [item]
        new_count = buffered_count + len(item)

        if new_count < n:
            return ((new_buffer, new_count), [])

        combined = new_buffer[0] if len(new_buffer) == 1 else concat(new_buffer)
        emissions: List[Set[Any]] = []

        while len(combined) >= n:
            emissions.append(combined[:n])
            combined = combined[n:]

        if len(combined) > 0:
            return (([combined], len(combined)), emissions)

        return (([], 0), emissions)

    def flush(
        state: Tuple[List[Set[Any]], int],
    ) -> Tuple[Tuple[List[Set[Any]], int], List[Set[Any]]]:
        buffer, buffered_count = state
        match (drop_last, buffered_count > 0):
            case (False, True):
                rem = buffer[0] if len(buffer) == 1 else concat(buffer)
                return (([], 0), [rem])
            case _:
                return (([], 0), [])

    return (accumulate, initial_state, flush)


class ChunkOp(Op):
    """Operation node that accumulates Set[T] instances into fixed-size batches."""

    def __init__(self, n: int, drop_last: bool = False):
        super().__init__(upstream=None)
        if n <= 0:
            raise ValueError(f"Chunk size n must be positive, got {n}.")
        self.n = n
        self.drop_last = drop_last

    def lower(self, ctx: LoweringContext) -> IROp:
        accumulate_fn, initial_state_fn, flush_fn = _make_set_chunk_accumulator(
            self.n,
            self.drop_last,
        )
        return IRAccumulate(
            accumulate_fn=accumulate_fn,
            initial_state_fn=initial_state_fn,
            flush_fn=flush_fn,
            name=f"chunk({self.n})",
        )


class _CollectState(NamedTuple):
    buffer: List[Set[Any]]
    buffered: int
    done: bool


def _make_set_collect_accumulator(
    n: int,
) -> Tuple[
    Callable[[_CollectState, Any], Tuple[_CollectState, List[Set[Any]]]],
    Callable[[], _CollectState],
    Callable[[_CollectState], Tuple[_CollectState, List[Set[Any]]]],
]:
    def initial_state() -> _CollectState:
        return _CollectState(buffer=[], buffered=0, done=False)

    def accumulate(
        state: _CollectState,
        item: Any,
    ) -> Tuple[_CollectState, List[Set[Any]]]:
        if not isinstance(item, Set):
            raise TypeError(
                f"CollectOp expected an instance of Set, got '{type(item).__name__}'.",
            )

        if state.done or len(item) == 0:
            return (state, [])

        buffer = state.buffer + [item]
        buffered = state.buffered + len(item)
        match buffered >= n:
            case True:
                collected = concat(buffer)[:n]
                return (_CollectState(buffer=[], buffered=0, done=True), [collected])

            case False:
                return (_CollectState(buffer=buffer, buffered=buffered, done=False), [])

    def flush(state: _CollectState) -> Tuple[_CollectState, List[Set[Any]]]:
        match (state.done, state.buffered > 0):
            case (False, True):
                collected = concat(state.buffer)
                return (_CollectState(buffer=[], buffered=0, done=True), [collected])

            case _:
                return (state, [])

    return (accumulate, initial_state, flush)


class CollectOp(Op):
    """Operation node that collects the first `n` elements of a stream into one Set."""

    def __init__(self, n: int):
        super().__init__(upstream=None)
        if n <= 0:
            raise ValueError(f"Collect size n must be positive, got {n}.")
        self.n = n

    def lower(self, ctx: LoweringContext) -> IROp:
        accumulate_fn, initial_state_fn, flush_fn = _make_set_collect_accumulator(
            self.n,
        )
        return IRAccumulate(
            accumulate_fn=accumulate_fn,
            initial_state_fn=initial_state_fn,
            flush_fn=flush_fn,
            name=f"collect({self.n})",
        )


def _flatten(state: None, item: Any) -> Tuple[None, List[Any]]:
    match item:
        case str() | bytes():
            raise TypeError(
                f"FlattenOp does not flatten '{type(item).__name__}' into characters.",
            )

        case Mapping():
            raise TypeError(
                "FlattenOp does not flatten mappings; flatten their .values() or "
                ".items() instead.",
            )

        case Iterable():
            return (state, list(item))

        case _:
            raise TypeError(
                f"FlattenOp expected an iterable, got '{type(item).__name__}'.",
            )


def _no_state() -> None:
    return None


class FlattenOp(Op):
    """Operation node emitting every element of incoming iterables (1:N)."""

    def __init__(self) -> None:
        super().__init__(upstream=None)

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRAccumulate(
            accumulate_fn=_flatten,
            initial_state_fn=_no_state,
            name="flatten",
        )


class _CombineLatestState(NamedTuple):
    latest: Any
    has_latest: bool
    pending: List[Any]


def _combine_latest_initial_state() -> _CombineLatestState:
    return _CombineLatestState(latest=None, has_latest=False, pending=[])


def _combine_latest(
    state: _CombineLatestState,
    tagged: Tagged,
) -> Tuple[_CombineLatestState, List[Tuple[Any, Any]]]:
    match tagged:
        case Tagged(port=0, item=item) if state.has_latest:
            return (state, [(item, state.latest)])

        case Tagged(port=0, item=item):
            return (state._replace(pending=state.pending + [item]), [])

        case Tagged(port=1, item=latest):
            released = [(item, latest) for item in state.pending]
            return (
                _CombineLatestState(latest=latest, has_latest=True, pending=[]),
                released,
            )

        case _:
            raise ValueError(f"CombineLatestOp has two inputs, got {tagged!r}.")


def _combine_latest_flush(
    state: _CombineLatestState,
) -> Tuple[_CombineLatestState, List[Tuple[Any, Any]]]:
    return (state._replace(pending=[]), [])


class CombineLatestOp(Op):
    """Operation node pairing every item of its first input with the latest of its second."""

    def __init__(self) -> None:
        super().__init__(upstream=None)

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRAccumulate(
            accumulate_fn=_combine_latest,
            initial_state_fn=_combine_latest_initial_state,
            flush_fn=_combine_latest_flush,
            name="combine_latest",
            tag_inputs=True,
        )


_GROUP_INPUT = "group_by.input"


def _make_key_extractor(
    key: int | str | Callable[[Any], Any],
) -> Callable[[Any], Any]:
    match key:
        case int() | str():
            return operator.itemgetter(key)

        case _ if callable(key):
            return key

        case _:
            raise TypeError(
                "group_by key must be an index, a field name or a function, got "
                f"'{type(key).__name__}'.",
            )


def _branches_off(exit_: Node) -> List[Node]:
    """Return the nodes consuming the lineage of ``exit_`` outside of it."""
    lineage: PySet[Node] = set()
    stack: List[Node] = [exit_]
    while stack:
        node = stack.pop()
        if node in lineage:
            continue

        lineage.add(node)
        stack.extend(node.upstream)

    return [
        consumer
        for node in lineage
        for consumer in node.downstream
        if consumer not in lineage
    ]


class KeyedOp(Op, Generic[K]):
    """Operation node routing items into a pipeline per key (``group_by``).

    The pipeline is kept as a template, outside of the enclosing graph, and
    lowered on its own.
    """

    def __init__(
        self,
        key_fn: Callable[[Any], K],
        entry: Source,
        exit_: Op,
        max_keys: Optional[int],
    ) -> None:
        super().__init__(upstream=None)
        self.key_fn = key_fn
        self.entry = entry
        self.exit_ = exit_
        self.max_keys = max_keys

    def unkey(self) -> Op:
        """End the keyed region; the stream carries ``Keyed(key, result)`` items."""
        return self

    def lower(self, ctx: LoweringContext) -> IROp:
        template_ctx = LoweringContext()
        template = template_ctx.lower_node(self.exit_)
        if set(template_ctx.sources) != {self.entry.name}:
            raise ValueError(
                "The group_by pipeline may only consume its input, but it "
                f"consumes {sorted(template_ctx.sources)}.",
            )

        return IRDemux(
            key_fn=self.key_fn,
            template=template,
            source_name=self.entry.name,
            max_keys=self.max_keys,
            name="group_by",
        )


def lineage(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every node upstream of it, each once.

    The pipelines of ``group_by`` ops belong to the lineage of the op.
    """
    seen: PySet[Node] = set()
    stack: List[Node] = [node]
    while stack:
        current = stack.pop()
        if current in seen:
            continue

        seen.add(current)
        yield current
        stack.extend(current.upstream)
        match current:
            case KeyedOp():
                stack.append(current.exit_)
