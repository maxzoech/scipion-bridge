from functools import partial, reduce
from pyrsistent import pdeque, PDeque

from streamz import Stream, Tuple
from scipion_bridge.core.struct import Struct
from scipion_bridge.core import struct

from .node import Node
from .sink import Sink

from typing import Optional, List, Dict, Callable, Any, Type, Union


class Op(Node):
    """
    Intermediate operation node that allows chaining downstream operations.
    """

    def op(self, node: Node) -> Node:
        """Connect a downstream node to this op."""
        node.upstream.append(self)
        return node

    def map(self, func: Callable[[Any], Any]) -> "MapOp":
        return self.op(MapOp(func))

    def batch(self, size: int) -> "BatchOp":
        return self.op(BatchOp(size))

    def sink(self, callback: Callable[[Any], Any]) -> Sink:
        """Attach a terminal Sink node and return it."""
        sink_node = Sink(callback)
        self.op(sink_node)
        return sink_node


class Source(Op):
    """
    Entry point input stream node.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(upstream=[])
        self.name = name

    def compile(
        self,
        sources_map: Dict[str, Stream],
        compile_cache: Optional[Dict[Node, Stream]] = None,
    ) -> Stream:
        del compile_cache
        assert (
            self.name is not None
        ), f"Source node {self} has no name assigned before building."
        return sources_map[self.name]

    def transform(self, *streams: Stream) -> Stream:
        if streams:
            return streams[0]
        raise NotImplementedError(
            "Source node must be compiled via sources_map lookup."
        )


class MapOp(Op):
    """Mapping operation node."""

    def __init__(self, func: Callable[[Any], Any]):
        super().__init__(upstream=None)
        self.func = func

    def transform(self, stream: Stream) -> Stream:
        return stream.map(self.func)


class BatchOp(Op):
    """
    Merges or splits a Set[...] to a specific batch size.
    """

    def __init__(self, size: int, upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.batch_size = size

    def _accumulate_batches(
        self,
        state: List[struct.Set],
        new_chunks: List[struct.Set],
    ) -> Tuple[List[struct.Set], List[struct.Set]]:
        initial_queue = [s for s in (state + new_chunks) if len(s) > 0]

        def _extract_one_batch(
            queue: List[struct.Set],
            accumulated: List[struct.Set],
            needed: int,
        ) -> Tuple[List[struct.Set], struct.Set]:
            if needed == 0:
                batch = (
                    accumulated[0]
                    if len(accumulated) == 1
                    else struct.Set.concat(*accumulated)
                )
                return queue, batch
            elif len(queue[0]) <= needed:
                head, tail = queue[0], queue[1:]
                return _extract_one_batch(tail, accumulated + [head], needed - len(head))
            else:
                head, tail = queue[0], queue[1:]
                take = head[:needed]
                leftover = head[needed:]
                return _extract_one_batch([leftover] + tail, accumulated + [take], 0)

        def _process_queue(
            queue: List[struct.Set], emitted: List[struct.Set]
        ) -> Tuple[List[struct.Set], List[struct.Set]]:
            total_capacity = sum(len(s) for s in queue)
            if total_capacity < self.batch_size:
                return queue, emitted
            else:
                remaining_queue, batch = _extract_one_batch(queue, [], self.batch_size)
                return _process_queue(remaining_queue, emitted + [batch])

        return _process_queue(initial_queue, [])

    def _chunk_set(self, x):
        if not isinstance(x, struct.Set):
            raise ValueError("Input for batch needs to be a set")

        n = len(x)
        if n > self.batch_size:
            return [x[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        return [x]

    def transform(self, stream: Stream) -> Stream:
        return (
            stream.map(self._chunk_set)
            .accumulate(
                self._accumulate_batches,
                start=[],
                returns_state=True,
            )
            .flatten()
        )


