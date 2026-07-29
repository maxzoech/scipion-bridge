from __future__ import annotations
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
from streamz import Stream

from scipion_bridge.core.struct import Struct, Set
from .ops import Source
from .sink import Sink
from .node import Node

T = TypeVar("T", bound=Union[Struct, Set, Any])

class Pipeline:
    """
    Compiled streaming engine backed by streamz.
    """

    def __init__(
        self,
        sinks: List[Sink],
        sources: Dict[str, Source],
        backend_sources: Dict[str, Stream],
    ):
        self.sinks = sinks
        self.sources = sources
        self._backend_sources = backend_sources

    @classmethod
    def from_sink(cls, *nodes: Sink) -> Pipeline:
        """
        Factory method: Discovers sources, compiles the DAG starting from target nodes,
        and returns a fully built Pipeline instance.
        """
        if not nodes:
            raise ValueError("Pipeline.from_sink() requires at least one target Node.")

        sources: Dict[str, Source] = {}
        visited = set()

        def _find_sources(n: Node):
            if n in visited:
                return
            visited.add(n)
            if isinstance(n, Source):
                sources[n.name] = n
            for up in n.upstream:
                _find_sources(up)

        for node in nodes:
            _find_sources(node)

        backend_sources = {name: Stream(stream_name=name) for name in sources.keys()}

        compile_cache: Dict[Node, Stream] = {}
        for node in nodes:
            node.compile(backend_sources, compile_cache)

        sinks = [n for n in nodes if isinstance(n, Sink)]
        return cls(sinks=sinks, sources=sources, backend_sources=backend_sources)

    def send(self, **kwargs: Any) -> None:
        """
        Submit data into the compiled stream using named keyword arguments.

        Usage:
            pipeline.send(particles=particle_set)
        """
        if not kwargs:
            raise ValueError("send() requires named keyword arguments (e.g., pipeline.send(particles=...))")

        for name, value in kwargs.items():
            if isinstance(value, list):
                raise TypeError(
                    f"Python lists are not supported for input '{name}'. Use core.struct.Set instead."
                )
            
            if name in self._backend_sources:
                self._backend_sources[name].emit(value)
            else:
                raise KeyError(f"Input source '{name}' is not registered in this pipeline.")