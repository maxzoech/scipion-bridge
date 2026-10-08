"""Streaming Intermediate Representation (IR) primitives.

The IR is a backend-agnostic DAG representation of the streaming pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .sink_writer import SinkWriter


@dataclass(frozen=True)
class Tagged:
    """Item arriving on a stage with several inputs, tagged with its input port.

    The port is the position of the sending stage among the upstream stages
    of the receiving stage.
    """

    port: int
    item: Any


class Keyed(NamedTuple):
    """Result of a ``group_by`` pipeline, together with the key of its group."""

    key: Any
    value: Any


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


@dataclass(eq=False)
class IRDemux(IROp):
    """Routing of items into a child pipeline per key (``group_by``).

    ``template`` is the exit node of the child pipeline, lowered on its own; it
    is not wired into the enclosing DAG. Its only source is ``source_name``.
    The backend compiles a clone of it for every new key and emits the
    results of each child as ``Keyed(key, result)``.
    """

    key_fn: Callable[[Any], Any] = field(default=lambda item: item)
    template: Optional[IROp] = field(default=None, repr=False)
    source_name: str = ""
    max_keys: Optional[int] = None
    name: str = "group_by"


def clone_ir(sinks: Sequence[IROp]) -> List[IROp]:
    """Copy the IR DAG reachable upstream from ``sinks``.

    Every node is copied with fresh edges, so the copy can be wired (e.g. to
    a new sink) and compiled without touching the original. Upstream edges
    keep their order, which preserves the input ports of multi-input stages.
    Functions, writers and other attributes are shared, not copied.

    Returns:
        The copies of ``sinks``, in the same order.
    """
    copies: Dict[IROp, IROp] = {}

    def copy(node: IROp) -> IROp:
        if node in copies:
            return copies[node]

        node_copy = dataclasses.replace(node, upstream=[], downstream=[])
        copies[node] = node_copy
        for up in node.upstream:
            copy(up).add_downstream(node_copy)
        return node_copy

    return [copy(sink) for sink in sinks]
