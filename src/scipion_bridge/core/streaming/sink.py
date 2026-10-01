"""Terminal output and checkpoint node for streaming pipelines."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union, TYPE_CHECKING

from .ir import IROp, IRSink
from .node import Node, LoweringContext
from .sink_writer import SinkWriter, CallbackSinkWriter


class Sink(Node):
    """Terminal output or checkpoint node backed by an async SinkWriter."""

    def __init__(
        self,
        writer: Union[SinkWriter, Callable[[Any], Any]],
        upstream: Optional[List[Node]] = None,
    ):
        super().__init__(upstream=upstream)
        if callable(writer) and not isinstance(writer, SinkWriter):
            self.writer: SinkWriter = CallbackSinkWriter(writer)
        else:
            self.writer = writer

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRSink(writer=self.writer)
