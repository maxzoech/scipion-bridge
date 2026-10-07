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

        # Inputs are identified by their position among the upstream stages, so
        # a stage cannot consume the same stream on several inputs.
        if len(ir_node.upstream) != len(node.upstream):
            raise ValueError(
                f"{type(node).__name__} receives the same stream on several "
                "inputs. Inputs of an operation must be distinct streams.",
            )

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


def replace_node(old: Node, new: Node) -> None:
    """Rewire every downstream consumer of ``old`` to consume ``new`` instead.

    The position of ``old`` in each consumer's ``upstream`` list is preserved, so
    the argument order of multi-input nodes does not change. ``old`` is left
    without downstream nodes.
    """
    for consumer in old.downstream:
        consumer.upstream = [new if up is old else up for up in consumer.upstream]
        new.downstream.append(consumer)

    old.downstream = []


def lower(nodes: List["Node"]) -> List[IROp]:
    """Lower one or more DAG root/sink nodes into lowered IR nodes.

    Returns:
        List of lowered IROp nodes corresponding to the input nodes.
    """
    ctx = LoweringContext()
    return [ctx.lower_node(node) for node in nodes]
