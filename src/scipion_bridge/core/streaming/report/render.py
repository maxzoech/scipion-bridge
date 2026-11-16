"""Renderers of pipeline reports: text, DOT, SVG and a self-contained HTML page.

A report (``model.py``) is the graph of a
pipeline with layers of annotations; these renderers draw any report without
knowing what its layers mean:

* ``render_text``: the stages and their annotations, for terminals and logs.
* ``render_dot``: the graph in Graphviz' DOT language.
* ``render_svg``: the graph laid out by Graphviz if its ``dot`` program is
  installed, else by the built-in layout (``layout.py``).
* ``render_html``: one HTML file without external resources: the graph, a
  panel listing the annotations of the clicked stage or edge per layer, the
  colouring of the stages by any numeric annotation of a layer (its
  ``metric`` first), the panels of the layers and their warnings.

Graphviz is optional. It is a program rather than a Python package (the
``graphviz`` package on PyPI only calls the program), so it is installed
with the system or conda, e.g. ``conda install graphviz``.
"""

from __future__ import annotations

import html
import shutil
import subprocess
from typing import List, Literal, Mapping, Optional, Sequence

from .model import (
    Annotations,
    GpuMapPanel,
    Layer,
    Panel,
    PipelineReport,
    TablePanel,
    Value,
)

from . import layout as builtin_layout
from .drawing import LOST_COLOR, PLAIN_FILL, SHAPES, Drawing, drawing_of

Layout = Literal["auto", "graphviz", "builtin"]


# -- Text --------------------------------------------------------------------
def render_text(report: PipelineReport, layers: Optional[Sequence[str]] = None) -> str:
    """The stages with the annotations of ``layers`` (default: all), panels and warnings."""
    shown = [layer for layer in report.layers if layers is None or layer.name in layers]
    depth = _depths(report)
    labels = report.group_labels
    lines = [
        f"Pipeline: {len(report.nodes)} stages, {len(report.groups)} compute groups",
        *(
            "  " * (depth[node.id] + 1)
            + node.id
            + ("" if node.group is None else f"  [{labels[node.group]}]")
            + "".join(
                f"  {layer.name}: {_format(layer.nodes[node.id])}"
                for layer in shown
                if node.id in layer.nodes
            )
            for node in report.nodes
        ),
    ]
    groups = [
        f"  {labels[group]} ({layer.name}): {_format(values)}"
        for layer in shown
        for group, values in layer.groups.items()
    ]
    panels = [
        line for layer in shown for panel in layer.panels for line in _panel_text(panel)
    ]
    warnings = [f"  - {warning}" for layer in shown for warning in layer.warnings]
    for title, section in [
        ("Groups", groups),
        ("Panels", panels),
        ("Warnings", warnings),
    ]:
        match section:
            case []:
                pass
            case _:
                lines += ["", f"{title}:", *section]
    return "\n".join(lines)


def _depths(report: PipelineReport) -> Mapping[str, int]:
    parents = {node.id: node.parent for node in report.nodes}

    def depth(node_id: str) -> int:
        match parents[node_id]:
            case None:
                return 0
            case parent:
                return depth(parent) + 1

    return {node.id: depth(node.id) for node in report.nodes}


def _format(values: Annotations) -> str:
    return " ".join(f"{key}={_value_text(value)}" for key, value in values.items())


def _value_text(value: Value) -> str:
    match value:
        case bool() | None:
            return str(value)
        case float():
            return f"{value:.3g}"
        case tuple():
            return ",".join(_value_text(item) for item in value)
        case _:
            return str(value)


def _panel_text(panel: Panel) -> List[str]:
    match panel:
        case TablePanel(title=title, columns=columns, rows=rows):
            cells = [list(columns), *([_value_text(v) for v in row] for row in rows)]
            widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
            return [
                f"  {title}",
                *(
                    "    " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths))
                    for row in cells
                ),
            ]
        case GpuMapPanel(title=title, gpus=gpus):
            return [
                f"  {title}",
                *(
                    f"    node {slot.node} {slot.name}: "
                    + (
                        ", ".join(f"{o.group} {o.share:.0%}" for o in slot.occupants)
                        or "free"
                    )
                    for slot in gpus
                ),
            ]


# -- DOT and SVG -------------------------------------------------------------
def render_dot(report: PipelineReport) -> str:
    """The graph of ``report`` in DOT, with the SVG ids of ``drawing.py``."""
    drawing = drawing_of(report)
    lines = [
        "digraph pipeline {",
        '  graph [rankdir=TB, bgcolor="transparent", fontname="Helvetica", '
        "fontsize=11, nodesep=0.3, ranksep=0.45];",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        f'fontsize=11, fillcolor="{PLAIN_FILL}", color="#555555"];',
        '  edge [color="#555555", arrowsize=0.7];',
        *_dot_cluster(drawing, None, "  "),
        *(
            f"  {drawing.nodes[e.source].svg_id} -> {drawing.nodes[e.target].svg_id} "
            f'[id="{e.svg_id}"'
            + ("" if not e.classes else f', class="{e.classes}"')
            + (", style=dashed" if e.dashed else "")
            + (', color="#999999"' if e.classes == "virtual" else "")
            + (f', color="{LOST_COLOR}"' if "lost" in e.classes.split() else "")
            + "];"
            for e in drawing.edges
        ),
        "}",
    ]
    return "\n".join(lines)


def _dot_cluster(drawing: Drawing, cluster: Optional[int], indent: str) -> List[str]:
    """The nodes directly in ``cluster`` (None: outside of clusters), and its sub-clusters."""
    lines: List[str] = []
    for i, node in enumerate(drawing.nodes):
        if node.cluster != cluster:
            continue
        label = "\\n".join(_dot_escape(line) for line in node.lines)
        style = "rounded,filled" if node.kind == "map" else "filled"
        lines.append(
            f'{indent}{node.svg_id} [id="{node.svg_id}", class="{node.classes}", '
            f'label="{label}", shape={SHAPES[node.kind]}, style="{style}", '
            f'fillcolor="{node.fill}"'
            + (", peripheries=2" if node.kind == "sink" else "")
            + "];",
        )
        if i in drawing.clusters:
            lines += [
                f"{indent}subgraph cluster_{i} {{",
                f'{indent}  id="c{i}"; class="cluster"; '
                f'label="{_dot_escape(node.lines[0])}"; labeljust=l; '
                'style="rounded,dashed"; color="#999999";',
                *_dot_cluster(drawing, i, indent + "  "),
                f"{indent}}}",
            ]
    return lines


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def graphviz_available() -> bool:
    """Whether Graphviz' ``dot`` program is installed."""
    return shutil.which("dot") is not None


def render_svg(report: PipelineReport, layout: Layout = "auto") -> str:
    """The graph of ``report`` as SVG.

    Args:
        layout: ``graphviz`` lays it out with Graphviz' ``dot``, ``builtin``
            with the built-in layout, ``auto`` with Graphviz if installed.

    Raises:
        RuntimeError: If ``layout`` is ``graphviz`` and Graphviz is not
            installed.
    """
    match (layout, graphviz_available()):
        case ("graphviz", False):
            raise RuntimeError(
                "Graphviz' dot program is not installed; install Graphviz "
                "(e.g. conda install graphviz) or use layout='builtin'.",
            )
        case ("graphviz", True) | ("auto", True):
            svg = subprocess.run(
                ["dot", "-Tsvg"],
                input=render_dot(report),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            # Without the XML declaration and doctype, to inline it in HTML.
            start = svg.index("<svg")
            return svg[start:]
        case _:
            return builtin_layout.render_svg(drawing_of(report))


def layout_name(layout: Layout) -> str:
    """The engine ``render_svg`` uses for ``layout``."""
    match (layout, graphviz_available()):
        case ("builtin", _) | ("auto", False):
            return "built-in layout"
        case _:
            return "Graphviz"


# -- HTML --------------------------------------------------------------------
def _numeric_keys(layer: Layer) -> List[str]:
    """The keys of the numeric annotations of the nodes of ``layer``, its
    ``metric`` first, then in the order they first appear."""
    keys = dict.fromkeys(
        key
        for annotations in layer.nodes.values()
        for key, value in annotations.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    first = [layer.metric] if layer.metric in keys else []
    return [*first, *(key for key in keys if key != layer.metric)]


def render_html(
    report: PipelineReport,
    layout: Layout = "auto",
    title: str = "Pipeline report",
) -> str:
    """A self-contained HTML page of ``report``; see the module docstring."""
    drawing = drawing_of(report)
    metrics = [(layer, key) for layer in report.layers for key in _numeric_keys(layer)]
    data = report.to_json().replace("</", "<\\/")
    return _PAGE.format(
        title=html.escape(title),
        style=_STYLE.replace("LOST_COLOR", LOST_COLOR),
        summary=html.escape(
            f"{len(report.nodes)} stages · {len(report.groups)} compute groups · "
            f"{len(report.layers)} layers",
        ),
        legend="".join(
            f'<span class="swatch" style="background:{fill}"></span>'
            f"{html.escape(group)}"
            for group, fill in drawing.group_fills.items()
        ),
        layers="".join(
            f'<label><input type="checkbox" id="layer-{html.escape(layer.name)}" '
            f"checked> {html.escape(layer.title)}</label>"
            for layer in report.layers
        )
        or '<span class="muted">no layers</span>',
        metrics="".join(
            f'<option value="{index}" data-layer="{html.escape(layer.name)}" '
            f'data-key="{html.escape(key)}">'
            f"{html.escape(layer.title)}: {html.escape(key)}</option>"
            for index, (layer, key) in enumerate(metrics, start=1)
        ),
        svg=render_svg(report, layout),
        panels="".join(
            _panel_html(panel, drawing.group_fills, layer)
            for layer in report.layers
            for panel in layer.panels
        ),
        warnings="".join(f"<li>{html.escape(w)}</li>" for w in report.warnings)
        or '<li class="muted">none</li>',
        engine=html.escape(layout_name(layout)),
        data=data,
        script=_SCRIPT,
    )


def _panel_html(panel: Panel, fills: Mapping[str, str], layer: Layer) -> str:
    match panel:
        case TablePanel(title=title, columns=columns, rows=rows):
            head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
            body = "".join(
                "<tr>"
                + "".join(f"<td>{html.escape(_value_text(v))}</td>" for v in row)
                + "</tr>"
                for row in rows
            )
            return (
                f"<section><h3>{html.escape(title)} "
                f'<span class="muted">{html.escape(layer.title)}</span></h3>'
                f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"
            )
        case GpuMapPanel(title=title, gpus=gpus):
            cells = "".join(
                f'<div class="gpu"><div class="gpu-name">node {slot.node} · {slot.name}</div>'
                '<div class="gpu-bar">'
                + "".join(
                    f'<div class="gpu-share" style="width:{o.share:.0%};'
                    f'background:{fills.get(o.group, PLAIN_FILL)}" '
                    f'title="{html.escape(o.group)} {o.share:.0%}">'
                    f"{html.escape(o.group)}</div>"
                    for o in slot.occupants
                )
                + "</div></div>"
                for slot in gpus
            )
            return (
                f"<section><h3>{html.escape(title)} "
                f'<span class="muted">{html.escape(layer.title)}</span></h3>'
                f'<div class="gpus">{cells}</div></section>'
            )


_STYLE = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7;
        --panel: #f5f5f7; --accent: #0a64d6; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161618; --fg: #ececf0; --muted: #9a9aa2; --line: #3a3a3f;
          --panel: #222226; --accent: #5aa2ff; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 14px 16px; border-bottom: 1px solid var(--line);
         display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: center; }
header h1 { font-size: 17px; margin: 0; }
.muted { color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
          margin-right: 4px; vertical-align: -1px; border: 1px solid #0003; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: 60vh; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
#graph { overflow: auto; padding: 16px; border-right: 1px solid var(--line); }
#graph svg { display: block; margin: 0 auto; max-width: none; height: auto; }
#graph.fit svg { max-width: 100%; }
#graph .node { cursor: pointer; }
#graph .node text { fill: #1d1d1f; }
#graph .edge path, #graph .edge polygon { stroke: var(--muted); }
#graph .edge polygon, #graph .arrowhead { fill: var(--muted); }
#graph .edge { cursor: pointer; }
#graph .edge.lost path, #graph .edge.lost polygon { stroke: LOST_COLOR; }
#graph .edge.lost polygon { fill: LOST_COLOR; }
#graph .cluster text, #graph .cluster-label { fill: var(--muted); }
#graph .cluster-frame, #graph .cluster polygon, #graph .cluster path {
  fill: none; stroke: var(--line); stroke-dasharray: 5 4; }
#graph .selected polygon, #graph .selected path, #graph .selected ellipse,
#graph .selected rect { stroke: var(--accent); stroke-width: 3; }
aside { padding: 14px 16px; background: var(--panel); overflow: auto; }
aside h2 { font-size: 15px; margin: 0 0 8px; word-break: break-all; }
aside h3 { font-size: 13px; margin: 14px 0 4px; color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--line);
         vertical-align: top; word-break: break-word; }
th { color: var(--muted); font-weight: 600; }
.panels { padding: 8px 16px 16px; }
.panels h3 { font-size: 14px; margin: 16px 0 6px; }
.gpus { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.gpu { border: 1px solid var(--line); border-radius: 6px; padding: 6px; }
.gpu-name { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.gpu-bar { display: flex; height: 22px; background: var(--panel); border-radius: 4px; overflow: hidden; }
.gpu-share { font-size: 11px; color: #1d1d1f; padding: 3px 4px; overflow: hidden;
             white-space: nowrap; border-right: 1px solid #0003; }
footer { padding: 8px 16px 16px; font-size: 12px; color: var(--muted); }
button, select { font: inherit; }
"""

_SCRIPT = """
const report = JSON.parse(document.getElementById('report-data').textContent);
const detail = document.getElementById('detail');
const enabled = () => report.layers.filter(
  l => document.getElementById('layer-' + l.name).checked);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[c]);
const fmt = v => Array.isArray(v) ? v.map(fmt).join(', ')
  : typeof v === 'number' && !Number.isInteger(v) ? v.toPrecision(3) : String(v);
const table = values => '<table>' + Object.entries(values).map(
  ([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(fmt(v))}</td></tr>`).join('') + '</table>';
let selected = null;
function select(element, html) {
  if (selected) selected.classList.remove('selected');
  selected = element;
  element.classList.add('selected');
  detail.innerHTML = html;
}
function showNode(i) {
  const n = report.nodes[i];
  let html = `<h2>${esc(n.title)}</h2>` + table({id: n.id, kind: n.kind,
    group: n.group_name, group_id: n.group, 'group_by': n.parent, stage_fp: n.stage_fp,
    fused: n.fused.length ? n.fused.join(' → ') : null});
  for (const l of enabled()) {
    if (n.id in l.nodes) html += `<h3>${esc(l.title)}</h3>` + table(l.nodes[n.id]);
    if (n.group && n.group in l.groups)
      html += `<h3>${esc(l.title)}: ${esc(n.group)}</h3>` + table(l.groups[n.group]);
  }
  select(document.getElementById('s' + i), html);
}
function showEdge(i) {
  const e = report.edges[i];
  let html = `<h2>${esc(e.source)} → ${esc(e.target)}</h2>` + table({port: e.port});
  for (const l of enabled())
    for (const a of l.edges)
      if (a.source === e.source && a.target === e.target)
        html += `<h3>${esc(l.title)}</h3>` + table(a.annotations);
  select(document.getElementById('e' + i), html);
}
report.nodes.forEach((n, i) => {
  const el = document.getElementById('s' + i);
  if (el) el.addEventListener('click', () => showNode(i));
});
report.edges.forEach((e, i) => {
  const el = document.getElementById('e' + i);
  if (el) el.addEventListener('click', () => showEdge(i));
});
const shapeOf = el => el.querySelector('polygon, path, ellipse, rect');
function colour() {
  const option = document.getElementById('metric').selectedOptions[0];
  const key = option.dataset.key;
  const layer = report.layers.find(l => l.name === option.dataset.layer);
  const values = report.nodes.map(n => layer && n.id in layer.nodes
    ? layer.nodes[n.id][key] : undefined);
  const known = values.filter(v => typeof v === 'number');
  const lo = Math.min(...known), hi = Math.max(...known);
  report.nodes.forEach((n, i) => {
    const el = document.getElementById('s' + i);
    if (!el) return;
    const shape = shapeOf(el), v = values[i];
    if (!layer) { shape.style.fill = ''; return; }
    if (typeof v !== 'number') { shape.style.fill = '#d9d9d9'; return; }
    const t = hi > lo ? (v - lo) / (hi - lo) : 1;
    shape.style.fill = `hsl(${50 - 50 * t}, 95%, ${88 - 30 * t}%)`;
  });
  document.getElementById('scale').textContent = layer && known.length
    ? `${key}: ${fmt(lo)} … ${fmt(hi)}` : '';
}
document.getElementById('metric').addEventListener('change', colour);
document.querySelectorAll('header input[type=checkbox]').forEach(
  c => c.addEventListener('change', () => selected && selected.dispatchEvent(new Event('click'))));
document.getElementById('fit').addEventListener('click', () =>
  document.getElementById('graph').classList.toggle('fit'));
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<header>
  <div><h1>{title}</h1><div class="muted">{summary}</div></div>
  <div class="legend">{legend}</div>
  <div class="controls">
    <span class="muted">Layers:</span> {layers}
    <label>Colour by <select id="metric"><option value="">compute group</option>{metrics}</select></label>
    <span id="scale" class="muted"></span>
    <button id="fit" type="button">Fit width</button>
  </div>
</header>
<main>
  <div id="graph">{svg}</div>
  <aside id="detail"><p class="muted">Click a stage or an edge to see its annotations.</p></aside>
</main>
<div class="panels">{panels}<h3>Warnings</h3><ul>{warnings}</ul></div>
<footer>Laid out with {engine}.</footer>
<script type="application/json" id="report-data">{data}</script>
<script>{script}</script>
</body>
</html>
"""
