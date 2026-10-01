"""Declarative streaming operations and DAG nodes."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    TYPE_CHECKING,
)

from .node import Node, LoweringContext
from .sink import Sink
from .sink_writer import SinkWriter, CallbackSinkWriter
from .ir import IROp, IRSource, IRMap

_NodeT = TypeVar("_NodeT", bound=Node)


class Op(Node):
    """Intermediate operation node that allows chaining downstream operations."""

    def op(self, node: _NodeT) -> _NodeT:
        """Connect a downstream node to this op."""
        node.upstream.append(self)
        self.downstream.append(node)
        return node

    def map_batch(self, func: Callable[[Any], Any]) -> MapOp:
        """Transform entire incoming stream item / batch (1:1)."""
        return self.op(MapOp(func))

    def map(self, func: Callable[[Any], Any]) -> MapOp:
        """Alias for map_batch."""
        return self.map_batch(func)

    def write_to(self, writer: SinkWriter) -> Sink:
        """Attach a terminal SinkWriter."""
        sink_node = Sink(writer)
        self.op(sink_node)
        return sink_node

    def checkpoint(self, writer: SinkWriter) -> Op:
        """Attach an asynchronous persistence checkpoint without cutting off the stream."""
        sink_node = Sink(writer)
        self.op(sink_node)
        return self

    def sink(self, callback: Callable[[Any], Any]) -> Sink:
        """Attach a callback-based sink (convenience for testing/debugging)."""
        return self.write_to(CallbackSinkWriter(callback))


class Source(Op):
    """Entry point input stream node."""

    def __init__(self, name: str):
        super().__init__(upstream=[])
        self.name = name

    def lower(self, ctx: LoweringContext) -> IROp:
        ir = IRSource(name=self.name)
        ctx.sources[self.name] = ir
        return ir


class MapOp(Op):
    """1:1 batch/element mapping operation node."""

    def __init__(self, func: Callable[[Any], Any]):
        super().__init__(upstream=None)
        self.func = func

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRMap(func=self.func)
