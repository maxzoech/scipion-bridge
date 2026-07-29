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
from streamz import Stream

from scipion_bridge.core.struct import Struct, Set
from .ops import Source, Node


T = TypeVar("T", bound=Union[Struct, Set, Any])

class Pipeline:
    """
    Compiled streaming engine backed by streamz.
    """

    def __init__(self, sources: List[Source], leaf: Optional[Node] = None):

        self.sources: Dict[str, Source] = {s.name: s for s in sources}
        self.leaf = leaf

        self._backend_sources: Dict[str, Stream] = {}
        self._backend_stream: Optional[Stream] = None

        self._is_built: bool = False

    def build(self) -> Pipeline:
        """
        Compile the Node DAG into a streamz pipeline.
        """
        for name in self.sources.keys():
            self._backend_sources[name] = Stream(stream_name=name)

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