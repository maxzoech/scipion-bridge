"""What a drawing of a pipeline report shows, shared by its layout engines.

Both engines (Graphviz and the built-in layout) draw the same nodes, edges and
clusters, with the same SVG ids and classes: node ``s<i>`` is the i-th node of
the report, edge ``e<i>`` its i-th edge. The HTML report finds them by these
ids, whichever engine drew them.

Besides the edges of the report, a drawing has *virtual* edges (``v<i>``)
through the template of every ``group_by``: from the ``group_by`` node into
the entries of its template, and from the exits of the template to where the
``group_by`` sends its results. They are not edges of the IR, but show where
the items of the children go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from .model import NodeKind, PipelineReport

# Fill colours of the compute groups, light enough for dark text, in order of
# the groups' first map.
PALETTE = (
    "#a6cee3",
    "#b2df8a",
    "#fdbf6f",
    "#cab2d6",
    "#fb9a99",
    "#ffed6f",
    "#8dd3c7",
    "#bebada",
    "#80b1d3",
    "#fccde5",
)
# Fill of stages without a compute group, and of group_by nodes.
PLAIN_FILL = "#eeeeee"
GROUP_BY_FILL = "#fff3d6"

# Graphviz shapes of the kinds of stages.
SHAPES: Mapping[NodeKind, str] = {
    "source": "ellipse",
    "map": "box",
    "accumulate": "box3d",
    "sink": "box",
    "group_by": "folder",
}


@dataclass(frozen=True)
class DrawnNode:
    """A node of the drawing.

    Attributes:
        svg_id: ``s<i>`` for the i-th node of the report.
        lines: The lines of its label.
        classes: Its SVG classes: kind and compute group.
        fill: Its fill colour.
        cluster: The index of the ``group_by`` node whose cluster holds it.
    """

    svg_id: str
    lines: Tuple[str, ...]
    kind: NodeKind
    classes: str
    fill: str
    cluster: Optional[int]


@dataclass(frozen=True)
class DrawnEdge:
    """An edge between the nodes ``source`` and ``target`` (indices).

    Attributes:
        svg_id: ``e<i>`` for the i-th edge of the report, ``v<i>`` for a
            virtual edge.
        dashed: Drawn dashed: a phase boundary or a virtual edge.
    """

    svg_id: str
    source: int
    target: int
    classes: str
    dashed: bool


@dataclass(frozen=True)
class Drawing:
    """The nodes, edges and ``group_by`` clusters of a report."""

    nodes: Tuple[DrawnNode, ...]
    edges: Tuple[DrawnEdge, ...]
    # The clusters, by the index of their group_by node, with the indices of
    # the nodes directly inside.
    clusters: Mapping[int, Tuple[int, ...]]
    # The fill of every compute group, by its label.
    group_fills: Mapping[str, str]


def drawing_of(report: PipelineReport) -> Drawing:
    """The drawing of ``report``."""
    index = {node.id: i for i, node in enumerate(report.nodes)}
    labels = report.group_labels
    group_fills = {
        labels[group]: PALETTE[i % len(PALETTE)]
        for i, group in enumerate(report.groups)
    }
    group_classes = {group: f"group-{i}" for i, group in enumerate(report.groups)}

    nodes = tuple(
        DrawnNode(
            svg_id=f"s{i}",
            lines=_lines(
                node.title, None if node.group is None else labels[node.group]
            ),
            kind=node.kind,
            classes=" ".join(
                [
                    f"kind-{node.kind}",
                    *(() if node.group is None else (group_classes[node.group],)),
                ],
            ),
            fill=_fill(
                node.kind,
                None if node.group is None else labels[node.group],
                group_fills,
            ),
            cluster=None if node.parent is None else index[node.parent],
        )
        for i, node in enumerate(report.nodes)
    )
    boundaries = {
        edge
        for layer in report.layers
        for edge, values in layer.edges.items()
        if "boundary" in values
    }
    edges = [
        DrawnEdge(
            svg_id=f"e{i}",
            source=index[edge.source],
            target=index[edge.target],
            classes="boundary" if (edge.source, edge.target) in boundaries else "",
            dashed=(edge.source, edge.target) in boundaries,
        )
        for i, edge in enumerate(report.edges)
    ]
    clusters: Dict[int, List[int]] = {}
    for i, node in enumerate(nodes):
        match node.cluster:
            case None:
                pass
            case cluster:
                clusters.setdefault(cluster, []).append(i)
    virtual = [
        (source, target)
        for demux, members in clusters.items()
        for source, target in _through_template(demux, members, edges)
    ]
    return Drawing(
        nodes=nodes,
        edges=(
            *edges,
            *(
                DrawnEdge(f"v{i}", source, target, "virtual", dashed=True)
                for i, (source, target) in enumerate(virtual)
            ),
        ),
        clusters={demux: tuple(members) for demux, members in clusters.items()},
        group_fills=group_fills,
    )


def _through_template(
    demux: int,
    members: List[int],
    edges: List[DrawnEdge],
) -> List[Tuple[int, int]]:
    """Virtual edges into the template of ``demux`` and out of it."""
    inside = set(members)
    entries = [m for m in members if not any(e.target == m for e in edges)]
    exits = [
        m
        for m in members
        if not any(e.source == m and e.target in inside for e in edges)
    ]
    targets = [e.target for e in edges if e.source == demux]
    return [
        *((demux, entry) for entry in entries),
        *((exit_, target) for exit_ in exits for target in targets),
    ]


def _lines(title: str, group: Optional[str]) -> Tuple[str, ...]:
    """The label of a node: its title, and its group unless the title names it."""
    match group:
        case None:
            return (title,)
        case _ if group in title:
            return (title,)
        case _:
            return (title, group)


def _fill(kind: NodeKind, group: Optional[str], fills: Mapping[str, str]) -> str:
    match (kind, group):
        case ("group_by", _):
            return GROUP_BY_FILL
        case (_, None):
            return PLAIN_FILL
        case (_, group):
            return fills[group]
