"""Base class for all nodes in the streaming computational graph."""

from __future__ import annotations

import abc
from typing import Any, List, Dict, Optional


from .ir import IROp, IRSource


class LoweringContext:
    """Context coordinator that manages memoization and DAG wiring during lowering."""

    def __init__(self) -> None:
        self.memo: Dict[int, IROp] = {}
        self.sources: Dict[str, IRSource] = {}

    def lower_node(self, node: "Node") -> IROp:
        """Recursively lowers a Node using polymorphism and wires DAG dependencies."""
        node_id = id(node)
        if node_id in self.memo:
            return self.memo[node_id]

        # Polymorphic lowering dispatch to the node's own implementation
        ir_node = node.lower(self)
        self.memo[node_id] = ir_node

        # Recursively lower upstream nodes and wire DAG edges
        for up in node.upstream:
            up_ir = self.lower_node(up)
            up_ir.add_downstream(ir_node)

        return ir_node


class FlushSignal:
    """Sentinel object emitted through the stream graph to trigger state flushing."""

    def __repr__(self) -> str:
        return "<FLUSH_SIGNAL>"


FLUSH = FlushSignal()


class Node(metaclass=abc.ABCMeta):
    """Base class for all nodes in the streaming computational graph."""

    def __init__(self, upstream: Optional[List[Node]] = None):
        self.upstream: List[Node] = upstream if upstream is not None else []
        self.downstream: List[Node] = []

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other

    @abc.abstractmethod
    def lower(self, ctx: "LoweringContext") -> IROp:
        """Polymorphically lower this surface Node to its low-level IR representation."""
        ...


def lower(nodes: List["Node"]) -> List[IROp]:
    """Lower one or more DAG root/sink nodes into lowered IR nodes.

    Returns:
        List of lowered IROp nodes corresponding to the input nodes.
    """
    ctx = LoweringContext()
    return [ctx.lower_node(node) for node in nodes]