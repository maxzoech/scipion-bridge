"""Reports of pipelines for debugging: the graph, annotated by layers, and its renderers.

``Pipeline.report()`` returns the report of a compiled pipeline with the stats
of its stages; ``report_of`` the graph of an IR without running it. Render it
with ``render_text``, ``render_dot``, ``render_svg`` or ``render_html``.
"""

from .model import (
    Annotations,
    GpuMapPanel,
    GpuOccupant,
    GpuSlot,
    Layer,
    NodeKind,
    Panel,
    PipelineReport,
    ReportEdge,
    ReportNode,
    TablePanel,
    Value,
    report_of,
)
from .render import (
    Layout,
    graphviz_available,
    render_dot,
    render_html,
    render_svg,
    render_text,
)

__all__ = [
    # The model
    "PipelineReport",
    "ReportNode",
    "ReportEdge",
    "Layer",
    "Panel",
    "TablePanel",
    "GpuMapPanel",
    "GpuSlot",
    "GpuOccupant",
    "Annotations",
    "Value",
    "NodeKind",
    "report_of",
    # Rendering
    "Layout",
    "graphviz_available",
    "render_text",
    "render_dot",
    "render_svg",
    "render_html",
]
