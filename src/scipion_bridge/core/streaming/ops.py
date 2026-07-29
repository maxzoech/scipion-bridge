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

    def sink(self, callback: Callable[[Any], Any]) -> Sink:
        """Attach a terminal Sink node and return it."""
        sink_node = Sink(callback)
        self.op(sink_node)
        return sink_node


class Source(Op):
    """
    Entry point input stream node.
    """

    def __init__(self, name: str, *, dtype: Optional[Type[Union[Struct, Set]]] = None):
        super().__init__(upstream=[])
        self.name = name
        self.dtype = dtype

    def compile(
        self,
        sources_map: Dict[str, Stream],
        compile_cache: Optional[Dict[Node, Stream]] = None,
    ) -> Stream:
        del compile_cache
        return sources_map[self.name]

    def transform(self, *streams: Stream) -> Stream:
        if streams:
            return streams[0]
        raise NotImplementedError("Source node must be compiled via sources_map lookup.")


class MapOp(Op):
    """Mapping operation node."""

    def __init__(self, func: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.func = func

    def transform(self, stream: Stream) -> Stream:
        return stream.map(self.func)


class CombineOp(Op):
    """Combines multiple upstream streams using streamz.combine_latest."""

    def __init__(self, upstream: List[Node]):
        super().__init__(upstream=upstream)

    def transform(self, *streams: Stream) -> Stream:
        primary_stream = streams[0]
        other_streams = streams[1:]
        return primary_stream.combine_latest(*other_streams)


class ZipOp(Op):
    """Zips multiple upstream streams element-by-element."""

    def __init__(self, upstream: List[Node]):
        super().__init__(upstream=upstream)

    def transform(self, *streams: Stream) -> Stream:
        primary_stream = streams[0]
        other_streams = streams[1:]
        return primary_stream.zip(*other_streams)
