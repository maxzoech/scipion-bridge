from typing import Optional, List, Dict, Callable, Any, Type, Union
from streamz import Stream
from scipion_bridge.core.struct import Struct, Set

from .node import Node
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

    def batch(self, batch_size: int, *, drop_last = False):
        return self.op(BatchOp(batch_size, drop_last))

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
        assert self.name is not None, f"Source node {self} has no name assigned before building."
        return sources_map[self.name]


    def transform(self, *streams: Stream) -> Stream:
        if streams:
            return streams[0]
        raise NotImplementedError("Source node must be compiled via sources_map lookup.")


class MapOp(Op):
    """Mapping operation node."""

    def __init__(self, func: Callable[[Any], Any]):
        super().__init__(upstream=None)
        self.func = func

    def transform(self, stream: Stream) -> Stream:
        return stream.map(self.func)


class BatchOp(Op):
    """
    Merges or splits a Set[...] to a specific batch size
    """

    def __init__(self, batch_size: int, drop_last=False):
        super().__init__(upstream=None)

        if drop_last:
            raise NotImplementedError("Drop last is not implemented yet")

    def transform(self, streams: Stream):
        raise NotImplementedError
