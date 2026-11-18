"""Terminal output and checkpoint node for streaming pipelines."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from .ir import IROp, IRSink
from .node import Node, LoweringContext
from .sink_writer import SinkWriter, CallbackSinkWriter, UniqueElements


class Sink(Node):
    """Terminal output or checkpoint node backed by an async SinkWriter.

    ``unique_elements`` says what the sink does with an element it writes a
    second time (by its element id).
    """

    def __init__(
        self,
        writer: Union[SinkWriter, Callable[[Any], Any]],
        upstream: Optional[List[Node]] = None,
        unique_elements: UniqueElements = UniqueElements.OFF,
    ):
        super().__init__(upstream=upstream)
        self.unique_elements = unique_elements
        if callable(writer) and not isinstance(writer, SinkWriter):
            self.writer: SinkWriter = CallbackSinkWriter(writer)
        else:
            self.writer = writer

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRSink(writer=self.writer, unique_elements=self.unique_elements)
