"""Streaming Intermediate Representation (IR) primitives.

The IR is a backend-agnostic DAG representation of the streaming pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .sink_writer import SinkWriter


@dataclass(frozen=True)
class Tagged:
    """Item arriving on a stage with several inputs, tagged with its input port.

    The port is the position of the sending stage among the upstream stages
    of the receiving stage.
    """

    port: int
    item: Any


@dataclass(eq=False)
class IROp:
    """Base class for all IR primitives with DAG edge management."""

    downstream: List[IROp] = field(default_factory=list, repr=False)
    upstream: List[IROp] = field(default_factory=list, repr=False)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other

    def add_downstream(self, child: IROp) -> None:
        """Wire a downstream edge and reciprocal upstream edge."""
        if child not in self.downstream:
            self.downstream.append(child)

        if self not in child.upstream:
            child.upstream.append(self)


@dataclass(eq=False)
class IRSource(IROp):
    """Ingestion entry point."""

    name: str = ""


@dataclass(eq=False)
class IRMap(IROp):
    """Stateless 1:1 batch/element transformation."""

    func: Callable[[Any], Any] = field(default=lambda x: x)
    name: Optional[str] = None


@dataclass(eq=False)
class IRAccumulate(IROp):
    """Stateful stream accumulation primitive.

    Maintains internal state across incoming items and flushes,
    emitting zero or more output items downstream.
    """

    accumulate_fn: Callable[[Any, Any], Tuple[Any, List[Any]]] = field(
        default=lambda state, item: (state, [item]),
    )
    initial_state_fn: Callable[[], Any] = field(default=lambda: None)
    flush_fn: Optional[Callable[[Any], Tuple[Any, List[Any]]]] = None
    name: str = "accumulate"
    # Pass items to accumulate_fn as Tagged(port, item), with the port they
    # arrive on, to tell the upstream stages apart.
    tag_inputs: bool = False


@dataclass(eq=False)
class IRSink(IROp):
    """Terminal/Checkpoint node delegating to an async SinkWriter."""

    writer: Optional[SinkWriter] = None
