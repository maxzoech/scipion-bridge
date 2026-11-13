"""Reports of a pipeline: its graph, annotated by layers.

A report is pure data. ``report_of`` builds the graph of a pipeline from its
IR; features of the library add what they know about it as a ``Layer``, such
as where the stages run (placement) or what they measured (stats). Renderers
(``render.py``) draw any report without knowing the
layers.

The ids of the nodes are the stage labels the backends use in ``stats()`` and
actor names (see ``stage_labels``). The nodes of a ``group_by`` template are
labelled under the id of their ``group_by`` node, ``<group_by id>/<label>``,
since its children are compiled at runtime.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from ...environment.compute import ComputeAssignment
from ..ir import (
    IRAccumulate,
    IRDemux,
    IRMap,
    IROp,
    IRSink,
    IRSource,
    describe,
    stage_labels,
    topological_order,
)

# A value of an annotation: anything JSON represents, with tuples for lists.
Value = Union[str, int, float, bool, None, Tuple["Value", ...]]
Annotations = Mapping[str, Value]

NodeKind = Literal["source", "map", "accumulate", "sink", "group_by"]


@dataclass(frozen=True)
class ReportNode:
    """A stage of the pipeline.

    Attributes:
        id: The stage label, as in ``stats()``.
        kind: The kind of IR node.
        title: Readable name, e.g. ``map(Protocol._forward)``.
        group: The compute group (protocol) of a map, if it has one.
        group_name: The readable name of the group, e.g. the class of the
            protocol; its id if it has none.
        parent: The id of the ``group_by`` whose template holds the node.
        stage_fp: The fingerprint of a stage of a protocol.
    """

    id: str
    kind: NodeKind
    title: str
    group: Optional[str] = None
    group_name: Optional[str] = None
    parent: Optional[str] = None
    stage_fp: Optional[str] = None


@dataclass(frozen=True)
class ReportEdge:
    """An edge from ``source`` into input ``port`` of ``target``."""

    source: str
    target: str
    port: int = 0


@dataclass(frozen=True)
class TablePanel:
    """A table shown below the graph."""

    title: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple[Value, ...], ...]


@dataclass(frozen=True)
class GpuOccupant:
    """A process on a GPU: its compute group and its share of the GPU."""

    group: str
    share: float


@dataclass(frozen=True)
class GpuSlot:
    """A GPU of the cluster, by node and index on the node.

    Attributes:
        node: The index of the node.
        gpu: The index of the GPU on the node in the plan.
        occupants: The processes on the GPU.
        device: The id of the GPU its executors ran on (as in
            ``CUDA_VISIBLE_DEVICES`` and ``nvidia-smi``), once they ran.
    """

    node: int
    gpu: int
    occupants: Tuple[GpuOccupant, ...] = ()
    device: Optional[str] = None

    @property
    def name(self) -> str:
        """``GPU <id>``: the device its executors ran on, else its index."""
        return f"GPU {self.gpu if self.device is None else self.device}"


@dataclass(frozen=True)
class GpuMapPanel:
    """The GPUs of the cluster and the processes on each."""

    title: str
    gpus: Tuple[GpuSlot, ...]


# The views of a layer besides the graph. New kinds are added here, together
# with their rendering.
Panel = Union[TablePanel, GpuMapPanel]


@dataclass(frozen=True)
class Layer:
    """What one feature knows about the pipeline.

    Attributes:
        name: Identifier of the layer, e.g. ``placement``.
        title: Readable name.
        nodes: Annotations per node id.
        edges: Annotations per edge, by the ids of its source and target.
        groups: Annotations per compute group.
        metric: A numeric annotation of the nodes that renderers may colour
            the nodes by.
        panels: Views that are not the graph.
        warnings: Findings worth the attention of the user.
    """

    name: str
    title: str
    nodes: Mapping[str, Annotations] = field(default_factory=dict)
    edges: Mapping[Tuple[str, str], Annotations] = field(default_factory=dict)
    groups: Mapping[str, Annotations] = field(default_factory=dict)
    metric: Optional[str] = None
    panels: Tuple[Panel, ...] = ()
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineReport:
    """The graph of a pipeline and the layers annotating it."""

    nodes: Tuple[ReportNode, ...]
    edges: Tuple[ReportEdge, ...]
    layers: Tuple[Layer, ...] = ()

    @property
    def groups(self) -> Tuple[str, ...]:
        """The compute groups of the maps, in the order of their first map."""
        return tuple(
            dict.fromkeys(node.group for node in self.nodes if node.group is not None),
        )

    @property
    def group_labels(self) -> Mapping[str, str]:
        """A readable label of every compute group: its name, with the start
        of its id if several groups have that name (e.g. two instances of a
        protocol)."""
        names = {
            node.group: node.group_name or node.group
            for node in self.nodes
            if node.group is not None
        }
        counts = Counter(names.values())
        return {
            group: name if counts[name] == 1 else f"{name} #{group[:6]}"
            for group, name in names.items()
        }

    @property
    def warnings(self) -> Tuple[str, ...]:
        """The warnings of all layers."""
        return tuple(warning for layer in self.layers for warning in layer.warnings)

    def layer(self, name: str) -> Layer:
        """The layer called ``name``.

        Raises:
            KeyError: If the report has no such layer.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"The report has no layer {name!r}.")

    def with_layer(self, layer: Layer) -> PipelineReport:
        """The report with ``layer`` added.

        Raises:
            ValueError: If the report has a layer of that name already, or the
                layer annotates nodes, edges or groups the graph does not
                have.
        """
        if layer.name in {existing.name for existing in self.layers}:
            raise ValueError(f"The report has a layer {layer.name!r} already.")
        unknown = [
            *(f"node {n!r}" for n in layer.nodes if n not in self._node_ids),
            *(f"edge {e!r}" for e in layer.edges if e not in self._edge_ids),
            *(f"group {g!r}" for g in layer.groups if g not in self.groups),
        ]
        if unknown:
            raise ValueError(
                f"Layer {layer.name!r} annotates what the graph does not have: "
                f"{', '.join(unknown)}.",
            )
        return replace(self, layers=(*self.layers, layer))

    @property
    def _node_ids(self) -> frozenset[str]:
        return frozenset(node.id for node in self.nodes)

    @property
    def _edge_ids(self) -> frozenset[Tuple[str, str]]:
        return frozenset((edge.source, edge.target) for edge in self.edges)

    def to_json(self) -> str:
        """The report as JSON with sorted keys, the format for other tools."""
        return json.dumps(
            {
                "nodes": [_node_json(node) for node in self.nodes],
                "edges": [_edge_json(edge) for edge in self.edges],
                "layers": [_layer_json(layer) for layer in self.layers],
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(text: str) -> PipelineReport:
        """Read a report written by ``to_json``."""
        data = json.loads(text)
        return PipelineReport(
            nodes=tuple(ReportNode(**node) for node in data["nodes"]),
            edges=tuple(ReportEdge(**edge) for edge in data["edges"]),
            layers=tuple(_layer_from_json(layer) for layer in data["layers"]),
        )


# The child of a group_by in the label of one of its stages: the stages of
# child ``a`` of ``2:group_by`` are labelled ``2:group_by[a]/<stage>``.
_CHILD = re.compile(r"\[[^\]]*\](?=/)")


def template_label(label: str) -> str:
    """The id of the node of a stage label: a stage of a ``group_by`` child
    is a node of its template."""
    return _CHILD.sub("", label)


def report_of(sinks: Sequence[IROp], prefix: str = "") -> PipelineReport:
    """The graph of the pipeline ending in ``sinks``, without layers.

    Nodes are in topological order; the nodes of a ``group_by`` template
    follow their ``group_by`` node. ``prefix`` is the name a backend compiles
    the pipeline with.
    """
    nodes: List[ReportNode] = []
    edges: List[ReportEdge] = []
    _add_graph(topological_order(sinks), prefix, None, nodes, edges)
    return PipelineReport(nodes=tuple(nodes), edges=tuple(edges))


def _add_graph(
    order: Sequence[IROp],
    prefix: str,
    parent: Optional[str],
    nodes: List[ReportNode],
    edges: List[ReportEdge],
) -> None:
    labels = stage_labels(order, prefix)
    for node in order:
        nodes.append(
            ReportNode(
                id=labels[node],
                kind=_kind(node),
                title=describe(node),
                group=_group(node),
                group_name=_group_name(node),
                parent=parent,
                stage_fp=node.stage_fp,
            ),
        )
        edges.extend(
            ReportEdge(labels[upstream], labels[node], port)
            for port, upstream in enumerate(node.upstream)
        )
        match node:
            case IRDemux(template=IROp() as template):
                _add_graph(
                    topological_order([template]),
                    f"{labels[node]}/",
                    labels[node],
                    nodes,
                    edges,
                )
            case _:
                pass


def _kind(node: IROp) -> NodeKind:
    match node:
        case IRSource():
            return "source"
        case IRMap():
            return "map"
        case IRAccumulate():
            return "accumulate"
        case IRSink():
            return "sink"
        case IRDemux():
            return "group_by"
        case _:
            raise TypeError(f"Unknown IR node {node!r}.")


def _group(node: IROp) -> Optional[str]:
    match node:
        case IRMap(compute=ComputeAssignment(group=group)):
            return group
        case _:
            return None


def _group_name(node: IROp) -> Optional[str]:
    match node:
        case IRMap(compute=ComputeAssignment(group=group, name="")):
            return group
        case IRMap(compute=ComputeAssignment(name=name)):
            return name
        case _:
            return None


def _node_json(node: ReportNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind,
        "title": node.title,
        "group": node.group,
        "group_name": node.group_name,
        "parent": node.parent,
        "stage_fp": node.stage_fp,
    }


def _edge_json(edge: ReportEdge) -> Dict[str, Any]:
    return {"source": edge.source, "target": edge.target, "port": edge.port}


def _layer_json(layer: Layer) -> Dict[str, Any]:
    return {
        "name": layer.name,
        "title": layer.title,
        "nodes": {node: dict(values) for node, values in layer.nodes.items()},
        "edges": [
            {"source": source, "target": target, "annotations": dict(values)}
            for (source, target), values in layer.edges.items()
        ],
        "groups": {group: dict(values) for group, values in layer.groups.items()},
        "metric": layer.metric,
        "panels": [_panel_json(panel) for panel in layer.panels],
        "warnings": list(layer.warnings),
    }


def _panel_json(panel: Panel) -> Dict[str, Any]:
    match panel:
        case TablePanel(title=title, columns=columns, rows=rows):
            return {
                "kind": "table",
                "title": title,
                "columns": list(columns),
                "rows": [list(row) for row in rows],
            }
        case GpuMapPanel(title=title, gpus=gpus):
            return {
                "kind": "gpu_map",
                "title": title,
                "gpus": [
                    {
                        "node": slot.node,
                        "gpu": slot.gpu,
                        "occupants": [
                            {"group": o.group, "share": o.share} for o in slot.occupants
                        ],
                        "device": slot.device,
                    }
                    for slot in gpus
                ],
            }


def _layer_from_json(data: Mapping[str, Any]) -> Layer:
    return Layer(
        name=data["name"],
        title=data["title"],
        nodes={node: _annotations(values) for node, values in data["nodes"].items()},
        edges={
            (edge["source"], edge["target"]): _annotations(edge["annotations"])
            for edge in data["edges"]
        },
        groups={
            group: _annotations(values) for group, values in data["groups"].items()
        },
        metric=data["metric"],
        panels=tuple(_panel_from_json(panel) for panel in data["panels"]),
        warnings=tuple(data["warnings"]),
    )


def _panel_from_json(data: Mapping[str, Any]) -> Panel:
    match data["kind"]:
        case "table":
            return TablePanel(
                title=data["title"],
                columns=tuple(data["columns"]),
                rows=tuple(tuple(_value(v) for v in row) for row in data["rows"]),
            )
        case "gpu_map":
            return GpuMapPanel(
                title=data["title"],
                gpus=tuple(
                    GpuSlot(
                        node=slot["node"],
                        gpu=slot["gpu"],
                        occupants=tuple(
                            GpuOccupant(o["group"], o["share"])
                            for o in slot["occupants"]
                        ),
                        device=slot.get("device"),
                    )
                    for slot in data["gpus"]
                ),
            )
        case kind:
            raise ValueError(f"Unknown panel kind {kind!r}.")


def _annotations(values: Mapping[str, Any]) -> Dict[str, Value]:
    return {key: _value(value) for key, value in values.items()}


def _value(value: Any) -> Value:
    """A value read from JSON, with lists as tuples again."""
    match value:
        case list():
            return tuple(_value(item) for item in value)
        case _:
            return value
