"""Built-in layout of a pipeline drawing as SVG, used without Graphviz.

Layered from top to bottom: a node's layer is the longest path to it from a
node without inputs, and the nodes of a layer are ordered by the mean
position of their inputs. ``group_by`` clusters are the boxes around their
members. Good enough for the chains and fan-outs of pipelines; Graphviz,
when installed, avoids crossings and overlaps of clusters better.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from .drawing import Drawing, DrawnNode

_CHAR_WIDTH = 7.0
_LINE_HEIGHT = 16.0
_PADDING = 10.0
_NODE_GAP = 28.0
_LAYER_GAP = 56.0
_CLUSTER_PADDING = 14.0
_MARGIN = 24.0


@dataclass(frozen=True)
class Box:
    """The rectangle of a node: its centre, width and height."""

    x: float
    y: float
    width: float
    height: float

    @property
    def top(self) -> float:
        return self.y - self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height / 2

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2


def layers_of(drawing: Drawing) -> List[List[int]]:
    """The nodes of every layer, in their order within the layer."""
    inputs: Dict[int, List[int]] = {i: [] for i in range(len(drawing.nodes))}
    for edge in drawing.edges:
        inputs[edge.target].append(edge.source)

    rank: Dict[int, int] = {}

    def rank_of(node: int) -> int:
        if node not in rank:
            rank[node] = max((rank_of(u) + 1 for u in inputs[node]), default=0)
        return rank[node]

    layers: List[List[int]] = [[] for _ in range(max(map(rank_of, inputs)) + 1)]
    for node in sorted(inputs):
        layers[rank_of(node)].append(node)

    position: Dict[int, float] = {}
    for layer in layers:
        layer.sort(
            key=lambda node: (
                _mean([position[u] for u in inputs[node]], default=node),
                node,
            ),
        )
        position.update({node: float(i) for i, node in enumerate(layer)})
    return layers


def boxes_of(drawing: Drawing) -> Dict[int, Box]:
    """The box of every node, without overlaps within a layer."""
    sizes = {i: _size(node) for i, node in enumerate(drawing.nodes)}
    layers = layers_of(drawing)
    widest = max(
        sum(sizes[n][0] for n in layer) + _NODE_GAP * (len(layer) - 1)
        for layer in layers
    )
    boxes: Dict[int, Box] = {}
    y = _MARGIN
    for layer in layers:
        height = max(sizes[n][1] for n in layer)
        width = sum(sizes[n][0] for n in layer) + _NODE_GAP * (len(layer) - 1)
        x = _MARGIN + _CLUSTER_PADDING + (widest - width) / 2
        for node in layer:
            node_width, node_height = sizes[node]
            boxes[node] = Box(
                x + node_width / 2, y + height / 2, node_width, node_height
            )
            x += node_width + _NODE_GAP
        y += height + _LAYER_GAP
    return boxes


def render_svg(drawing: Drawing) -> str:
    """The drawing as an SVG document."""
    boxes = boxes_of(drawing)
    clusters = {
        demux: _cluster_box(demux, drawing.clusters, boxes)
        for demux in drawing.clusters
    }
    width = max(box.right for box in boxes.values()) + _MARGIN + _CLUSTER_PADDING
    height = max(box.bottom for box in boxes.values()) + _MARGIN + _CLUSTER_PADDING
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" class="arrowhead"/></marker></defs>',
        *(
            _cluster_svg(i, drawing.nodes[demux].lines[0], box)
            for i, (demux, box) in enumerate(sorted(clusters.items()))
        ),
        *(
            _edge_svg(
                edge.svg_id,
                edge.classes,
                edge.dashed,
                boxes[edge.source],
                boxes[edge.target],
            )
            for edge in drawing.edges
        ),
        *(_node_svg(node, boxes[i]) for i, node in enumerate(drawing.nodes)),
        "</svg>",
    ]
    return "\n".join(parts)


def _size(node: DrawnNode) -> Tuple[float, float]:
    width = max(len(line) for line in node.lines) * _CHAR_WIDTH + 2 * _PADDING
    return width, len(node.lines) * _LINE_HEIGHT + _PADDING


def _mean(values: List[float], default: float) -> float:
    match values:
        case []:
            return default
        case _:
            return sum(values) / len(values)


def _cluster_box(
    demux: int,
    clusters: Mapping[int, Tuple[int, ...]],
    boxes: Mapping[int, Box],
) -> Box:
    """The box around the members of a cluster, nested clusters included."""
    members = [
        _cluster_box(m, clusters, boxes) if m in clusters else boxes[m]
        for m in clusters[demux]
    ] + [boxes[m] for m in clusters[demux] if m in clusters]
    left = min(b.left for b in members) - _CLUSTER_PADDING
    right = max(b.right for b in members) + _CLUSTER_PADDING
    top = min(b.top for b in members) - _CLUSTER_PADDING - _LINE_HEIGHT
    bottom = max(b.bottom for b in members) + _CLUSTER_PADDING
    return Box((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)


def _cluster_svg(index: int, label: str, box: Box) -> str:
    return (
        f'<g id="c{index}" class="cluster">'
        f'<rect x="{box.left:.1f}" y="{box.top:.1f}" width="{box.width:.1f}" '
        f'height="{box.height:.1f}" rx="8" class="cluster-frame"/>'
        f'<text x="{box.left + 8:.1f}" y="{box.top + 14:.1f}" '
        f'class="cluster-label">{html.escape(label)}</text></g>'
    )


def _edge_svg(svg_id: str, classes: str, dashed: bool, source: Box, target: Box) -> str:
    x1, y1, x2, y2 = source.x, source.bottom, target.x, target.top
    middle = (y1 + y2) / 2
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    return (
        f'<g id="{svg_id}" class="edge {classes}">'
        f'<path d="M {x1:.1f} {y1:.1f} C {x1:.1f} {middle:.1f}, {x2:.1f} '
        f'{middle:.1f}, {x2:.1f} {y2:.1f}" fill="none" stroke="#555"{dash} '
        'marker-end="url(#arrow)"/></g>'
    )


def _node_svg(node: DrawnNode, box: Box) -> str:
    rounding = {"source": box.height / 2, "sink": box.height / 2, "map": 6}.get(
        node.kind, 0
    )
    peripheries = (
        f'<rect x="{box.left - 3:.1f}" y="{box.top - 3:.1f}" width="{box.width + 6:.1f}" '
        f'height="{box.height + 6:.1f}" rx="{rounding + 3:.1f}" fill="none" stroke="#555"/>'
        if node.kind == "sink"
        else ""
    )
    lines = "".join(
        f'<text x="{box.x:.1f}" y="{box.top + _PADDING / 2 + (i + 0.8) * _LINE_HEIGHT:.1f}" '
        f'text-anchor="middle" class="{"label" if i == 0 else "label-group"}">'
        f"{html.escape(line)}</text>"
        for i, line in enumerate(node.lines)
    )
    return (
        f'<g id="{node.svg_id}" class="node {node.classes}">'
        f"<title>{html.escape(node.lines[0])}</title>"
        f'<rect x="{box.left:.1f}" y="{box.top:.1f}" width="{box.width:.1f}" '
        f'height="{box.height:.1f}" rx="{rounding:.1f}" fill="{node.fill}" stroke="#555"/>'
        f"{peripheries}{lines}</g>"
    )
