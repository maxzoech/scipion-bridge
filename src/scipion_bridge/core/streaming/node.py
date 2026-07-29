from typing import Optional, List, Dict
from streamz import Stream

class Node:
    """
    Base class for all nodes in the streaming computational graph.
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
            up.compile(sources_map, compile_cache) for up in self.upstream
        ]
        compiled = self.transform(*upstream_streams)

        compile_cache[self] = compiled
        return compiled