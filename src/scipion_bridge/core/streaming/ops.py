from typing import Optional, List, Dict, Callable, Any, Type, Union
from streamz import Stream
from scipion_bridge.core.struct import Struct, Set


class Node:
    """
    Base class for nodes in the streaming computational graph.
    """

    def __init__(self, upstream: Optional[List["Node"]] = None):
        self.upstream: List[Node] = upstream if upstream is not None else []

    def transform(self, *streams: Stream) -> Stream:
        """
        Transforms upstream streamz stream(s) into a new streamz stream operator.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def compile(
        self,
        sources_map: Dict[str, Stream],
        compile_cache: Optional[Dict["Node", Stream]] = None,
    ) -> Stream:
        """
        Recursively compile this node and its upstream dependencies into a streamz stream.
        """
        if compile_cache is None:
            compile_cache = {}

        if self in compile_cache:
            return compile_cache[self]

        upstream_streams = [
            upstream.compile(sources_map, compile_cache) for upstream in self.upstream
        ]
        compiled = self.transform(*upstream_streams)

        compile_cache[self] = compiled
        return compiled

    def op(self, node: "Node") -> "Node":
        """Add a downstream operation node to this node."""
        node.upstream.append(self)
        return node

    def build(self) -> Any:
        """Convenience method to compile the computational graph from root sources to this node."""
        from .pipeline import Pipeline

        visited = set()
        sources: List[Source] = []

        def _find_sources(n: Node):
            if n in visited:
                return
            visited.add(n)
            if isinstance(n, Source):
                sources.append(n)
            for up in n.upstream:
                _find_sources(up)

        _find_sources(self)
        pipeline = Pipeline(sources=sources, leaf=self)
        return pipeline.build()

    # Fluent helper methods
    def map(self, func: Callable[[Any], Any]) -> "MapOp":
        return self.op(MapOp(func))

    def sink(self, callback: Callable[[Any], Any]) -> "SinkOp":
        return self.op(SinkOp(callback))



class Source(Node):
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


class MapOp(Node):
    """Mapping operation node."""

    def __init__(self, func: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.func = func

    def transform(self, stream: Stream) -> Stream:
        return stream.map(self.func)

class SinkOp(Node):
    """Sink output operation node."""

    def __init__(self, callback: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.callback = callback

    def transform(self, stream: Stream) -> Stream:
        return stream.sink(self.callback)

