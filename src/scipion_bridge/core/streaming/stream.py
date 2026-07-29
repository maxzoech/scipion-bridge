from __future__ import annotations
import functools
from typing import (
    TypeVar,
    Generic,
    Optional,
    Any,
    Callable,
    Dict,
    List,
    Union,
    Type,
)

import streamz
from streamz import Stream as BackendStream

from scipion_bridge.core.struct import Struct, Set

T = TypeVar("T", bound=Union[Struct, Set, Any])


class Node:
    """
    Base class for nodes in the streaming computational graph.
    """

    def __init__(self, upstream: Optional[List[Node]] = None):
        self.upstream: List[Node] = upstream if upstream is not None else []

    def transform(self, inputs: Union[BackendStream, List[BackendStream]]) -> BackendStream:
        """
        Transforms upstream streamz stream(s) into a new streamz stream operator.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def compile(
        self,
        sources_map: Dict[str, BackendStream],
        compiled_cache: Optional[Dict[Node, BackendStream]] = None,
    ) -> BackendStream:
        """
        Recursively compile this node and its upstream dependencies into a streamz stream.
        """
        if compiled_cache is None:
            compiled_cache = {}

        if self in compiled_cache:
            return compiled_cache[self]

        if isinstance(self, Source):
            compiled = sources_map[self.name]
        else:
            upstream_compiled = [
                up.compile(sources_map, compiled_cache) for up in self.upstream
            ]
            in_arg = upstream_compiled[0] if len(upstream_compiled) == 1 else upstream_compiled
            compiled = self.transform(in_arg)

        compiled_cache[self] = compiled
        return compiled

    def op(self, node: Node) -> Node:
        """Add a downstream operation node to this node."""
        node.upstream.append(self)
        return node

    def build(self) -> Stream:
        """Convenience method to compile the computational graph from root sources to this node."""
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
        stream = Stream(sources=sources, leaf=self)
        return stream.build()

    # Fluent helper methods
    def map(self, func: Callable[[Any], Any]) -> MapOp:
        return self.op(MapOp(func)) 

    def filter(self, predicate: Callable[[Any], bool]) -> FilterOp:
        return self.op(FilterOp(predicate))

    def sink(self, callback: Callable[[Any], Any]) -> SinkOp:
        return self.op(SinkOp(callback))


class Source(Node):
    """
    Entry point input stream node.
    """

    def __init__(self, name: str, *, dtype: Optional[Type[Union[Struct, Set]]] = None):
        super().__init__(upstream=[])
        self.name = name
        self.dtype = dtype

    def transform(self, inputs: Union[BackendStream, List[BackendStream]]) -> BackendStream:
        if isinstance(inputs, list):
            return inputs[0]
        return inputs


class MapOp(Node):
    """Mapping operation node."""

    def __init__(self, func: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.func = func

    def transform(self, inputs: Union[BackendStream, List[BackendStream]]) -> BackendStream:
        in_stream = inputs[0] if isinstance(inputs, list) else inputs
        return in_stream.map(self.func)


class FilterOp(Node):
    """Filter operation node."""

    def __init__(self, predicate: Callable[[Any], bool], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.predicate = predicate

    def transform(self, inputs: Union[BackendStream, List[BackendStream]]) -> BackendStream:
        in_stream = inputs[0] if isinstance(inputs, list) else inputs
        return in_stream.filter(self.predicate)


class SinkOp(Node):
    """Sink output operation node."""

    def __init__(self, callback: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.callback = callback

    def transform(self, inputs: Union[BackendStream, List[BackendStream]]) -> BackendStream:
        in_stream = inputs[0] if isinstance(inputs, list) else inputs
        return in_stream.sink(self.callback)


class Stream:
    """
    Compiled streaming engine backed by streamz.
    """

    def __init__(self, sources: List[Source], leaf: Optional[Node] = None):

        self.sources: Dict[str, Source] = {s.name: s for s in sources}
        self.leaf = leaf

        self._backend_sources: Dict[str, BackendStream] = {}
        self._backend_stream: Optional[BackendStream] = None

        self._is_built: bool = False

    def build(self) -> Stream:
        """
        Compile the Node DAG into a streamz pipeline.
        """
        for name in self.sources.keys():
            self._backend_sources[name] = BackendStream(stream_name=name)

        if self.leaf:
            self._backend_stream = self.leaf.compile(self._backend_sources)
        else:
            self._backend_stream = list(self._backend_sources.values())[0]

        self._is_built = True
        return self

    def send(self, **kwargs: Any) -> None:
        """
        Submit data into the compiled stream using named keyword arguments.

        Usage:
            stream.send(particles=particle_set)
        """
        if not self._is_built:
            self.build()

        if not kwargs:
            raise ValueError("send() requires named keyword arguments (e.g., stream.send(particles=...))")

        for name, value in kwargs.items():
            if isinstance(value, list):
                raise TypeError(
                    f"Python lists are not supported for input '{name}'. Use core.struct.Set instead."
                )
            if name in self._backend_sources:
                self._backend_sources[name].emit(value)
            else:
                raise KeyError(f"Input source '{name}' is not registered in this stream.")