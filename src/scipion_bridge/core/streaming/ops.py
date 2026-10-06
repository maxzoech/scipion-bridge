"""Declarative streaming operations and DAG nodes."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    List,
    Tuple,
    TypeVar,
)

from ..struct import Set, concat
from .node import Node, LoweringContext
from .sink import Sink
from .sink_writer import SinkWriter, CallbackSinkWriter
from .ir import IROp, IRSource, IRMap, IRAccumulate

_NodeT = TypeVar("_NodeT", bound=Node)


class Op(Node):
    """Intermediate operation node that allows chaining downstream operations."""

    def op(self, node: _NodeT) -> _NodeT:
        """Connect a downstream node to this op."""
        node.upstream.append(self)
        self.downstream.append(node)
        return node

    def map_batch(self, func: Callable[[Any], Any]) -> MapOp:
        """Transform entire incoming stream item / batch (1:1).

        Every map is executed as its own pipeline stage, and consecutive stages
        process different items concurrently. Splitting a step into separate
        maps therefore overlaps its parts, e.g. CPU post-processing of one
        batch with the GPU forward pass of the next::

            particles.chunk(256).map(forward).map(build_metadata)
        """
        return self.op(MapOp(func))

    def map(self, func: Callable[[Any], Any]) -> MapOp:
        """Alias for map_batch."""
        return self.map_batch(func)

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
        ir = IRSource(name=self.name)
        ctx.sources[self.name] = ir
        return ir


class MapOp(Op):
    """1:1 batch/element mapping operation node."""

    def __init__(self, func: Callable[[Any], Any]):
        super().__init__(upstream=None)
        self.func = func

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRMap(func=self.func)


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
        )
