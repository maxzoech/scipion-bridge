from typing import Optional, List, Dict, Callable, Any, Type, Union, Tuple
from functools import partial, reduce
from pyrsistent import pdeque, PDeque

from streamz import Stream
from scipion_bridge.core.struct import Struct
from scipion_bridge.core import struct

from .node import Node, FlushSignal, FLUSH
from .sink import Sink


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

    def chunk(self, size: int) -> "ChunkOp":
        return self.op(ChunkOp(size))

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

    def _map_func(self, x: Any) -> Any:
        if isinstance(x, FlushSignal):
            return x
        return self.func(x)

    def transform(self, stream: Stream) -> Stream:
        return stream.map(self._map_func)


class ChunkOp(Op):
    """
    Merges or splits Set[...] containers to a specific chunk size.
    """

    def __init__(self, size: int, upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.chunk_size = size

    def _accumulate_chunks(
        self,
        state: Tuple[List[struct.Set], int],
        new_set: Union[struct.Set, FlushSignal],
    ) -> Tuple[Tuple[List[struct.Set], int], List[Union[struct.Set, FlushSignal]]]:
        queue, capacity = state

        if isinstance(new_set, FlushSignal):
            emitted: List[Union[struct.Set, FlushSignal]] = []
            if queue:
                leftover = (
                    queue[0]
                    if len(queue) == 1
                    else struct.Set.concat(*queue)
                )
                emitted.append(leftover)
            emitted.append(FLUSH)
            
            return ([], 0), emitted

        if not isinstance(new_set, struct.Set):
            raise ValueError("Input for chunk needs to be a set")

        if len(new_set) > 0:
            queue.append(new_set)
            capacity += len(new_set)

        emitted: List[Union[struct.Set, FlushSignal]] = []

        while capacity >= self.chunk_size:
            accumulated: List[struct.Set] = []
            needed = self.chunk_size

            while queue and needed > 0:
                head = queue[0]
                head_len = len(head)

                if head_len <= needed:
                    accumulated.append(head)
                    needed -= head_len
                    queue.pop(0)
                else:
                    accumulated.append(head[:needed])
                    queue[0] = head[needed:]
                    needed = 0

            chunk = (
                accumulated[0]
                if len(accumulated) == 1
                else struct.Set.concat(*accumulated)
            )
            emitted.append(chunk)
            capacity -= self.chunk_size

        return (queue, capacity), emitted

    def transform(self, stream: Stream) -> Stream:
        return stream.accumulate(
            self._accumulate_chunks,
            start=([], 0),
            returns_state=True,
        ).flatten()
