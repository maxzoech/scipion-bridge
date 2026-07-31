import abc
from streamz import Stream

from typing import Optional, List, Dict, Any


class FlushSignal:
    """Sentinel object emitted through the stream graph to trigger state flushing."""

    def __repr__(self) -> str:
        return "<FLUSH_SIGNAL>"


FLUSH = FlushSignal()


class Node(metaclass=abc.ABCMeta):
    """
    Base class for all nodes in the streaming computational graph.
    """

    def __init__(self, upstream: Optional[List["Node"]] = None):
        self.upstream: List[Node] = upstream if upstream is not None else []

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other


    @abc.abstractmethod
    def transform(self, *streams: Stream) -> Stream:
        """
        Transforms upstream streamz stream(s) into a new streamz stream operator.
        Must be implemented by subclasses.
        """

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