"""A debug viewer for the lineage of runs.

Serves a page that browses the lineage runs recorded in their run directories
(``--run-dir``, see ``run_store.py``). Like TensorBoard's ``--logdir``, it
takes a directory and lists every run below it::

    python -m scipion_bridge.core.streaming.report.lineage_view runs/

and open http://127.0.0.1:8765. A run directory is a directory with a
``manifest.json`` or a ``lineage/`` directory; the directory given may be one
itself. The list is read again on every visit, so runs started later appear,
and runs that wrote records in the last minute are marked as running. For the
run selected, the page shows:

- the run: its protocols with their fingerprints, parameters and inputs;
- every stage, with the items it made and how many of their elements are
  rows of input files (a stage whose items hold fewer input rows than its
  parents' created new rows: views or ``astype`` would keep them);
- the outputs of the run, and for every item its parents, children, groups
  of elements (e.g. the members of every class) and ancestry;
- for every element, where it came from (an input file and row, or the item
  a stage created it in) and every item holding it;
- the input files, with the rows the run read from them;
- the graph of the pipeline (its report, as ``--report`` draws it), with the
  lineage of an item or an element over it: how many of its ancestors, of
  its descendants or of the items holding an element every stage made, and
  how many links between them every edge carries. A link from a stage to the
  next stage that records items is drawn along the stages between them that
  pass items on (``group_by`` routers, the sources of their children, sinks).

It reads the run once and answers from ``LineageGraph``, which keeps the
records columnar: a run of a million particles loads in seconds and every
page is answered in a fraction of one. Lists are paged. ``Reload`` reads the
records a running pipeline wrote since.

``LineageView`` builds the JSON of every page and can be used without the
server.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from ..lineage import ElementSegment, ItemId
from ..lineage_graph import LineageGraph, element_arrays
from ..run_store import FileRunStore, RunStore
from .drawing import drawing_of
from .model import PipelineReport, template_label
from .render import render_svg

# Items, elements or files a list shows per page, unless asked otherwise.
PAGE = 100
_ZERO = bytes(16)


class LineageView:
    """The pages of the viewer, as JSON-ready dictionaries."""

    def __init__(self, store: RunStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> Dict[str, Any]:
        """Read the records the run wrote so far."""
        graph = LineageGraph.of(self._store)
        manifest = self._store.read_manifest()
        report = self._store.read_report()
        ops = None if report is None else _OpsGraph(PipelineReport.from_json(report))
        with self._lock:
            self.graph, self.manifest, self.ops = graph, manifest, ops
        # The first pages would wait for the indices otherwise.
        threading.Thread(target=graph.warm, daemon=True).start()
        return {"records": len(graph)}

    def summary(self) -> Dict[str, Any]:
        """The run and its stages."""
        graph = self.graph
        return {
            "manifest": _manifest_summary(self.manifest),
            "records": len(graph),
            "outputs": len(graph.leaves()),
            "files": graph.files(),
            "stages": [
                {
                    "label": s.stage.label,
                    "kind": s.stage.kind,
                    "stage_fp": s.stage.stage_fp,
                    "items": s.items,
                    "elements": s.elements,
                    "input_elements": s.input_elements,
                    "inherited": s.inherited,
                }
                for s in graph.stage_summaries()
            ],
        }

    def outputs(self, offset: int = 0, limit: int = PAGE) -> Dict[str, Any]:
        """The items no stage made anything from, by stage."""
        leaves = self.graph.leaves()
        return {
            "total": len(leaves),
            "offset": offset,
            "items": [self._item_row(lid) for lid in leaves[offset : offset + limit]],
        }

    def stage(self, label: str, offset: int = 0, limit: int = PAGE) -> Dict[str, Any]:
        """The items a stage made."""
        items = self.graph.items_of(label)
        return {
            "label": label,
            "total": len(items),
            "offset": offset,
            "items": [self._item_row(lid) for lid in items[offset : offset + limit]],
        }

    def item(self, item_id: ItemId) -> Dict[str, Any]:
        """An item: its stage, parents, children, element groups and files."""
        graph = self.graph
        record = graph.record(item_id)
        children = graph.children(item_id)
        elements, inputs = graph.element_count(item_id)
        return {
            **self._item_row(item_id),
            "inherited": record.inherited,
            "parents": [self._link(p) for p in record.parents[:PAGE]],
            "parent_count": len(record.parents),
            "children": [self._link(c) for c in children[:PAGE]],
            "child_count": len(children),
            "files": [str(f.path) for f in record.files[:PAGE]],
            "file_count": len(record.files),
            "elements": elements,
            "input_elements": inputs,
            "groups": self._groups(graph.elements(item_id)),
        }

    def elements(
        self,
        item_id: ItemId,
        group: int,
        offset: int = 0,
        limit: int = PAGE,
    ) -> Dict[str, Any]:
        """The elements of a group of an item (see ``item``), paged."""
        segments = _grouped(self.graph.elements(item_id))[group][1]
        namespaces, positions = element_arrays(segments)
        page = slice(offset, offset + limit)
        return {
            "total": len(positions),
            "offset": offset,
            "elements": [
                self._element_row(namespace.tobytes(), int(position))
                for namespace, position in zip(namespaces[page], positions[page])
            ],
        }

    def ancestry(self, item_id: ItemId) -> Dict[str, Any]:
        """What an item was made from: items per stage, sources and files."""
        graph = self.graph
        ancestry = graph.ancestry(item_id)
        per_stage: Dict[str, int] = {}
        for lid in ancestry.item_ids:
            stage = graph.stage_of(lid)
            per_stage[stage] = per_stage.get(stage, 0) + 1
        return {
            "item": self._link(item_id),
            "items": len(ancestry.item_ids),
            "stages": [
                {
                    "label": stage.label,
                    "stage_fp": stage.stage_fp,
                    "items": per_stage.get(stage.label, 0),
                }
                for stage in ancestry.stages
            ],
            "sources": [self._link(s.item_id) for s in ancestry.sources[:PAGE]],
            "source_count": len(ancestry.sources),
            "files": [str(f.path) for f in ancestry.files[:PAGE]],
            "file_count": len(ancestry.files),
            "missing": [lid.hex() for lid in ancestry.missing[:PAGE]],
        }

    def element(self, namespace: bytes, position: int) -> Dict[str, Any]:
        """An element: where it came from, and the items holding it."""
        graph = self.graph
        holders = graph.holders(namespace, position)
        leaves = set(graph.leaves())
        try:
            origin = graph.trace_element(namespace, position)
        except KeyError:
            origin = None
        per_stage: Dict[str, List[Dict[str, Any]]] = {}
        for lid in holders:
            per_stage.setdefault(graph.stage_of(lid), []).append(self._link(lid))
        return {
            **self._element_row(namespace, position),
            "created_by": (
                None
                if origin is None or origin.created_by is None
                else self._link(origin.created_by.item_id)
            ),
            "holders": len(holders),
            "outputs": [self._link(lid) for lid in holders if lid in leaves][:PAGE],
            "stages": [
                {"label": label, "items": items[:PAGE], "count": len(items)}
                for label, items in per_stage.items()
            ],
        }

    def ops_graph(self) -> Dict[str, Any]:
        """The graph of the pipeline, with the items every stage made."""
        if self.ops is None:
            return {"svg": None}
        totals = {s.stage.label: s.items for s in self.graph.stage_summaries()}
        return {
            "svg": self.ops.svg,
            "nodes": self.ops.nodes(self.graph.stages, totals),
            "counts": self.ops.node_counts(totals),
        }

    def overlay(self, kind: str, *args: Any) -> Dict[str, Any]:
        """The lineage of an item (``ancestry``, ``descendants``) or of an
        element (``element``) over the graph of the pipeline.

        Raises:
            ValueError: Without a graph, or for another ``kind``.
        """
        if self.ops is None:
            raise ValueError("The run recorded no graph of its pipeline.")
        match kind:
            case "ancestry":
                (item_id,) = args
                items = list(self.graph.ancestry(item_id).item_ids)
                subject: Dict[str, Any] = self._link(item_id)
            case "descendants":
                (item_id,) = args
                items = self.graph.descendants(item_id)
                subject = self._link(item_id)
            case "element":
                namespace, position = args
                items = self.graph.holders(namespace, position)
                subject = self._element_row(namespace, position)
            case _:
                raise ValueError(f"No overlay {kind!r}.")
        stages, links = self.graph.stage_links(items)
        edges, through, unplaced = self.ops.edge_counts(links)
        return {
            "kind": kind,
            "subject": subject,
            "items": len(items),
            "counts": self.ops.node_counts(stages),
            "edges": edges,
            "through": through,
            "unplaced": [[a, b, n] for (a, b), n in unplaced.items()],
        }

    def search(self, text: str) -> Dict[str, Any]:
        """Items by (a prefix of) their hex id, and input files by path."""
        text = text.strip()
        items: List[Dict[str, Any]] = []
        if len(text) >= 6 and all(c in "0123456789abcdef" for c in text.lower()):
            prefix = text.lower()
            items = [
                self._link(lid)
                for lid in self.graph.item_ids()
                if lid.hex().startswith(prefix)
            ][:PAGE]
        files = [
            {"index": index, "path": str(f.path)}
            for index, f in self.graph.find_files(text, PAGE)
        ]
        return {"items": items, "files": files}

    def file(self, index: int, offset: int = 0, limit: int = PAGE) -> Dict[str, Any]:
        """An input file and the rows of it the run read."""
        f = self.graph.file(index)
        rows = self.graph.rows_of(f.namespace)
        return {
            "index": index,
            "path": str(f.path),
            "size": f.size,
            "mtime_ns": f.mtime_ns,
            "namespace": f.namespace.hex(),
            "total": len(rows),
            "offset": offset,
            "rows": [int(r) for r in rows[offset : offset + limit]],
        }

    def _item_row(self, item_id: ItemId) -> Dict[str, Any]:
        elements, inputs = self.graph.element_count(item_id)
        return {
            **self._link(item_id),
            "kind": self.graph.kind_of(item_id),
            "elements": elements,
            "input_elements": inputs,
        }

    def _link(self, item_id: ItemId) -> Dict[str, Any]:
        stage = self.graph.stage_of(item_id) if item_id in self.graph else None
        return {"id": item_id.hex(), "stage": stage}

    def _element_row(self, namespace: bytes, position: int) -> Dict[str, Any]:
        """An element, with the input file it is a row of, if it is one."""
        if position < 0:
            kind = "none" if namespace == _ZERO else "explicit"
            return {"namespace": namespace.hex(), "position": None, "kind": kind}
        index = self.graph.file_index(namespace)
        return {
            "namespace": namespace.hex(),
            "position": position,
            "kind": "created" if index is None else "input",
            "file": None if index is None else str(self.graph.file(index).path),
            "file_index": index,
        }

    def _groups(self, segments: Sequence[ElementSegment]) -> List[Dict[str, Any]]:
        """The element groups of an item: the Sets it holds, by where."""
        result = []
        for index, ((where, path), group) in enumerate(_grouped(segments)):
            namespaces, positions = element_arrays(group)
            inputs = _input_mask(self.graph, namespaces, positions)
            result.append(
                {
                    "index": index,
                    "where": where,
                    "path": path,
                    "elements": int(len(positions)),
                    "input_elements": int(inputs.sum()),
                    "without_id": int(np.sum(positions < 0)),
                },
            )
        return result


# The edges of a path in the drawing (their SVG ids), and the nodes it
# passes through (their indices).
_Path = Tuple[Tuple[str, ...], Tuple[int, ...]]


class _OpsGraph:
    """The drawing of a pipeline report, and where lineage goes on it.

    Stages of ``group_by`` children are nodes of their template. A link
    between the items of two stages is drawn along the path between their
    nodes through nodes that make no records (``group_by`` routers, the
    sources of their children, sinks).
    """

    def __init__(self, report: PipelineReport) -> None:
        self.report = report
        self.svg = render_svg(report, "auto")
        drawing = drawing_of(report)
        self._index = {node.id: i for i, node in enumerate(report.nodes)}
        self._out: Dict[int, List[Tuple[int, str]]] = {}
        for edge in drawing.edges:
            self._out.setdefault(edge.source, []).append((edge.target, edge.svg_id))
        self._pass_through = {
            i
            for i, node in enumerate(report.nodes)
            if node.kind in ("group_by", "sink")
            or (node.kind == "source" and node.parent is not None)
        }
        self._paths: Dict[Tuple[int, int], Optional[_Path]] = {}

    def node_of(self, stage: str) -> Optional[int]:
        """The node of a stage label, if the report has it."""
        return self._index.get(template_label(stage))

    def nodes(
        self,
        stages: Mapping[str, Any],
        totals: Mapping[str, int],
    ) -> List[Dict[str, Any]]:
        """Every node: its id in the SVG, label, title and the stages of the
        run it stands for (several for the template of a ``group_by``)."""
        labels = sorted({*stages, *totals})
        return [
            {
                "svg_id": f"s{i}",
                "label": node.id,
                "title": node.title,
                "kind": node.kind,
                "stage_fp": node.stage_fp,
                "stages": [
                    label for label in labels if template_label(label) == node.id
                ],
            }
            for i, node in enumerate(self.report.nodes)
        ]

    def node_counts(self, stages: Mapping[str, int]) -> Dict[str, int]:
        """Counts by stage, summed per node of the drawing."""
        counts: Dict[str, int] = {}
        for stage, count in stages.items():
            node = self.node_of(stage)
            if node is not None:
                counts[f"s{node}"] = counts.get(f"s{node}", 0) + count
        return counts

    def edge_counts(
        self,
        links: Mapping[Tuple[str, str], int],
    ) -> Tuple[Dict[str, int], List[str], Dict[Tuple[str, str], int]]:
        """Links by stages, summed per edge of the drawing along their paths;
        the nodes the paths pass through; and the links without a path (e.g.
        of stages the report lacks)."""
        edges: Dict[str, int] = {}
        through: Dict[str, None] = {}
        unplaced: Dict[Tuple[str, str], int] = {}
        for (parent, child), count in links.items():
            source, target = self.node_of(parent), self.node_of(child)
            path = (
                None if source is None or target is None else self._path(source, target)
            )
            if path is None:
                unplaced[(parent, child)] = count
                continue
            path_edges, path_nodes = path
            for svg_id in path_edges:
                edges[svg_id] = edges.get(svg_id, 0) + count
            through.update((f"s{node}", None) for node in path_nodes)
        return edges, list(through), unplaced

    def _path(self, source: int, target: int) -> Optional[_Path]:
        """The edges from ``source`` to ``target`` through pass-through nodes,
        and those nodes."""
        key = (source, target)
        if key not in self._paths:
            self._paths[key] = self._search(source, target)
        return self._paths[key]

    def _search(self, source: int, target: int) -> Optional[_Path]:
        start: _Path = ((), ())
        queue: deque[Tuple[int, _Path]] = deque([(source, start)])
        seen = {source}
        while queue:
            node, (edges, nodes) = queue.popleft()
            for following, svg_id in self._out.get(node, []):
                if following == target:
                    return (*edges, svg_id), nodes
                if following in self._pass_through and following not in seen:
                    seen.add(following)
                    queue.append((following, ((*edges, svg_id), (*nodes, following))))
        return None


def _grouped(
    segments: Sequence[ElementSegment],
) -> List[Tuple[Tuple[str, str], List[ElementSegment]]]:
    """``segments`` by the Set they belong to, in order."""
    groups: Dict[Tuple[str, str], List[ElementSegment]] = {}
    for segment in segments:
        groups.setdefault((segment.where, segment.path), []).append(segment)
    return list(groups.items())


def _input_mask(
    graph: LineageGraph,
    namespaces: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Which elements are rows of input files."""
    unique, inverse = np.unique(namespaces, return_inverse=True)
    is_input = np.array(
        [graph.file_index(key.tobytes()) is not None for key in unique],
        dtype=bool,
    )
    return is_input[inverse.ravel()] & (positions >= 0) if unique.size else is_input


def _manifest_summary(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The manifest, with long input lists shortened."""
    inputs = {
        name: {
            "values": (
                value["value"][:20]
                if isinstance(value["value"], list)
                else [value["value"]]
            ),
            "count": len(value["value"]) if isinstance(value["value"], list) else 1,
        }
        for name, value in manifest.get("inputs", {}).items()
    }
    return {
        **{k: v for k, v in manifest.items() if k != "inputs"},
        "inputs": inputs,
    }


# -- The runs of a log directory --------------------------------------------------
def find_runs(logdir: Path) -> Dict[str, Path]:
    """Every run directory in ``logdir``, by its path relative to it (``.``
    for ``logdir`` itself). A run's own directories are not searched."""
    found: Dict[str, Path] = {}
    for root, dirs, files in os.walk(logdir):
        here = Path(root)
        if "manifest.json" in files or "lineage" in dirs:
            found[here.relative_to(logdir).as_posix()] = here
            dirs.clear()
        else:
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
    return dict(sorted(found.items()))


class Runs:
    """The runs of a log directory, and the views of those being browsed.

    A view holds the lineage of its run in memory (about 1 GB for a million
    particles), so only the ``keep`` views used last stay loaded.
    """

    def __init__(self, logdir: Path, keep: int = 2) -> None:
        self.logdir = logdir
        self._keep = keep
        self._views: OrderedDict[str, LineageView] = OrderedDict()
        self._lock = threading.Lock()

    def list(self) -> List[Dict[str, Any]]:
        """Every run: its name, when it was created and last wrote records,
        its protocols, and how many record files it has."""
        return [_run_info(name, path) for name, path in find_runs(self.logdir).items()]

    def view(self, name: str) -> LineageView:
        """The view of the run ``name``, loaded on first use.

        Raises:
            KeyError: If ``logdir`` holds no run ``name``.
        """
        with self._lock:
            if name not in self._views:
                path = find_runs(self.logdir)[name]
                self._views[name] = LineageView(FileRunStore(path))
                while len(self._views) > self._keep:
                    self._views.popitem(last=False)
            self._views.move_to_end(name)
            return self._views[name]


def _run_info(name: str, path: Path) -> Dict[str, Any]:
    """What the list of runs shows of a run, read without loading it."""
    manifest = FileRunStore(path).read_manifest()
    parts = list((path / "lineage").glob("*.parquet"))
    written = [
        p.stat().st_mtime for p in [path / "manifest.json", *parts] if p.exists()
    ]
    return {
        "name": name,
        "run_id": manifest.get("run_id"),
        "created": manifest.get("created"),
        "protocols": [p["protocol"] for p in manifest.get("protocols", [])],
        "parts": len(parts),
        "updated": max(written, default=None),
    }


# -- The server -------------------------------------------------------------------
def _route_runs(runs: Runs, path: str, query: Mapping[str, str]) -> Any:
    """The JSON answer of an API path: ``/api/runs``, or a path of the run
    ``name`` as ``/api/run/<name>/...`` (see ``_route``)."""
    segments = path.strip("/").split("/")
    match segments:
        case ["api", "runs"]:
            return runs.list()
        case ["api", "run", name, *rest] if rest:
            return _route(runs.view(unquote(name)), "/api/" + "/".join(rest), query)
        case _:
            raise KeyError(path)


def _route(view: LineageView, path: str, query: Mapping[str, str]) -> Any:
    """The JSON answer of an API path."""
    offset = int(query.get("offset", 0))
    limit = min(int(query.get("limit", PAGE)), 10 * PAGE)
    parts = [unquote(p) for p in path.strip("/").split("/")][1:]
    routes: Dict[str, Callable[[], Any]] = {
        "summary": view.summary,
        "reload": view.reload,
        "outputs": lambda: view.outputs(offset, limit),
        "stage": lambda: view.stage(query["label"], offset, limit),
        "item": lambda: view.item(bytes.fromhex(parts[1])),
        "elements": lambda: view.elements(
            bytes.fromhex(parts[1]),
            int(query["group"]),
            offset,
            limit,
        ),
        "ancestry": lambda: view.ancestry(bytes.fromhex(parts[1])),
        "element": lambda: view.element(
            bytes.fromhex(query["namespace"]),
            int(query["position"]),
        ),
        "search": lambda: view.search(query.get("q", "")),
        "graph": view.ops_graph,
        "overlay": lambda: (
            view.overlay(
                "element",
                bytes.fromhex(query["namespace"]),
                int(query["position"]),
            )
            if parts[1] == "element"
            else view.overlay(parts[1], bytes.fromhex(parts[2]))
        ),
        "file": lambda: view.file(int(parts[1]), offset, limit),
    }
    return routes[parts[0]]()


def make_handler(runs: Runs) -> type:
    """The request handler of a server for the runs ``runs``."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE_HTML.encode())
                return
            if not url.path.startswith("/api/"):
                self._send(404, "text/plain", b"Not found")
                return
            try:
                body = json.dumps(_route_runs(runs, url.path, query)).encode()
            except (KeyError, ValueError, IndexError) as error:
                message = json.dumps({"error": f"{type(error).__name__}: {error}"})
                self._send(404, "application/json", message.encode())
                return
            self._send(200, "application/json", body)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def serve(
    logdir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    ready: Optional[Callable[[ThreadingHTTPServer], None]] = None,
) -> None:
    """Serve the viewer of the runs in ``logdir`` until interrupted."""
    runs = Runs(logdir)
    server = ThreadingHTTPServer((host, port), make_handler(runs))
    if ready is not None:
        ready(server)
    count = len(find_runs(logdir))
    print(
        f"Lineage of {count} run(s) in {logdir}: http://{host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Browse the lineage of runs.")
    parser.add_argument(
        "logdir",
        type=Path,
        help="A directory of runs (their --run-dir), or one run directory.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.logdir, args.host, args.port)


PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Lineage</title>
<style>
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7;
        --panel: #f5f5f7; --accent: #0a64d6; --good: #1a7f37; --warn: #b35900; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #161618; --fg: #ececf0; --muted: #9a9aa2;
    --line: #3a3a3f; --panel: #222226; --accent: #5aa2ff; --good: #4ac26b; --warn: #f0a050; }
}
:root[data-theme="dark"] { --bg: #161618; --fg: #ececf0; --muted: #9a9aa2;
  --line: #3a3a3f; --panel: #222226; --accent: #5aa2ff; --good: #4ac26b; --warn: #f0a050; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex;
         flex-wrap: wrap; gap: 8px 20px; align-items: center; }
header h1 { font-size: 17px; margin: 0; }
header form { display: flex; gap: 6px; flex: 1; min-width: 220px; max-width: 520px; }
input, button { font: inherit; color: var(--fg); background: var(--bg);
                border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
input { flex: 1; min-width: 0; }
button { cursor: pointer; }
main { display: grid; grid-template-columns: minmax(280px, 420px) minmax(0, 1fr);
       min-height: calc(100vh - 56px); }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
nav { border-right: 1px solid var(--line); background: var(--panel); padding: 12px 16px;
      overflow: auto; max-height: calc(100vh - 56px); }
section { padding: 12px 16px; overflow: auto; max-height: calc(100vh - 56px); }
h2 { font-size: 15px; margin: 4px 0 8px; word-break: break-all; }
h3 { font-size: 13px; margin: 16px 0 4px; color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
a { color: var(--accent); text-decoration: none; cursor: pointer; }
a:hover { text-decoration: underline; }
code, .id { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }
.muted { color: var(--muted); }
.input { color: var(--good); }
.created { color: var(--warn); }
.bar { display: inline-block; height: 8px; border-radius: 4px; background: var(--good);
       vertical-align: middle; }
.bar.rest { background: var(--warn); }
.pager { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 2px 12px;
     margin: 0; font-size: 13px; }
dt { color: var(--muted); }
dd { margin: 0; word-break: break-all; }
.error { color: var(--warn); }
.runs { list-style: none; margin: 0 0 12px; padding: 0; }
.runs li { padding: 5px 8px; border-radius: 6px; margin: 1px 0; }
.runs li.current { background: var(--bg); outline: 1px solid var(--accent); }
.runs a { display: block; font-weight: 600; word-break: break-all; }
.runs .meta { font-size: 12px; color: var(--muted); }
.live { display: inline-block; width: 8px; height: 8px; border-radius: 4px;
        background: var(--good); margin-right: 4px; vertical-align: 1px; }
#ops { overflow: auto; border: 1px solid var(--line); border-radius: 8px;
       padding: 8px; background: var(--bg); }
#ops svg { display: block; margin: 0 auto; height: auto; }
#ops.fit svg { max-width: 100%; }
#ops svg > g > polygon:first-child { fill: transparent; stroke: none; }
#ops .node { cursor: pointer; }
#ops .edge path, #ops .edge polygon { stroke: var(--muted); }
#ops .edge polygon { fill: var(--muted); }
#ops .cluster text, #ops .cluster-label { fill: var(--muted); }
#ops .cluster polygon, #ops .cluster path, #ops .cluster-frame {
  fill: none; stroke: var(--line); stroke-dasharray: 5 4; }
#ops.dim .node:not(.lit) { opacity: 0.3; }
#ops.dim .edge:not(.lit) { opacity: 0.25; }
#ops .edge.lit path { stroke: var(--accent); }
#ops .edge.lit polygon { stroke: var(--accent); fill: var(--accent); }
#ops .node.lit polygon, #ops .node.lit rect:not(.badge), #ops .node.lit ellipse,
#ops .node.lit path { stroke: var(--accent); stroke-width: 2.5; }
#ops .badge { fill: var(--accent); }
#ops .node.through polygon, #ops .node.through ellipse, #ops .node.through path,
#ops .node.through rect:not(.badge) { stroke-dasharray: 4 3; stroke-width: 2; }
#ops .badge-text { fill: #fff; font: 600 11px system-ui, sans-serif; }
#ops .selected polygon, #ops .selected rect:not(.badge) { stroke: var(--warn); stroke-width: 3; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 6px 0; }
</style></head>
<body>
<header>
  <h1>Run lineage</h1><span id="run" class="muted"></span>
  <form id="search"><input id="q" placeholder="Item id (hex) or input file path">
    <button>Search</button></form>
  <button id="reload" title="Read the runs, and the records written since">Reload</button>
</header>
<main><nav><div id="runs"></div><div id="nav"></div></nav><section id="page"></section></main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const short = (hex) => hex ? hex.slice(0, 12) + "…" : "";
const num = (n) => Number(n).toLocaleString();
async function get(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}
// The run selected, and the start of the hash of its pages.
let RUN = null, base = "#/";
const api = (path) => get(`/api/run/${encodeURIComponent(RUN)}/${path}`);
const itemLink = (it) =>
  `<a href="${base}item/${it.id}" class="id" title="${esc(it.id)}">${short(it.id)}</a>`;
const stageLink = (label) =>
  `<a href="${base}stage/${encodeURIComponent(label)}">${esc(label)}</a>`;
const elementLink = (e) => e.position === null
  ? `<span class="muted">${e.kind === "none" ? "no id" : "hashed " + short(e.namespace)}</span>`
  : `<a href="${base}element/${e.namespace}/${e.position}">${e.kind === "input"
      ? esc(e.file.split("/").pop()) + " row " + e.position
      : "created " + short(e.namespace) + " row " + e.position}</a>`;
function share(input, total) {
  if (!total) return "";
  const w = 80, a = Math.round(w * input / total);
  return `<span class="bar" style="width:${a}px"></span><span class="bar rest" style="width:${w - a}px"></span>`;
}
function pager(data, route) {
  const {total, offset} = data, size = (data.items || data.elements || data.rows).length;
  if (total <= size && offset === 0) return "";
  const prev = offset > 0 ? `<a href="${route(Math.max(0, offset - 100))}">← previous</a>` : "";
  const next = offset + size < total ? `<a href="${route(offset + 100)}">next →</a>` : "";
  return `<div class="pager">${prev}<span class="muted">${num(offset + 1)}–${num(offset + size)} of ${num(total)}</span>${next}</div>`;
}
function itemsTable(items) {
  return `<table><tr><th>Item</th><th>Stage</th><th class="num">Elements</th><th>Input rows</th></tr>` +
    items.map((it) => `<tr><td>${itemLink(it)}</td><td>${esc(it.stage)}</td>
      <td class="num">${num(it.elements)}</td><td>${share(it.input_elements, it.elements)}</td></tr>`).join("") +
    `</table>`;
}

async function showNav() {
  const s = await api("summary");
  const m = s.manifest;
  $("run").textContent = [RUN, (m.created || "").replace("T", " ").slice(0, 19)]
    .filter(Boolean).join(" · ");
  $("nav").innerHTML = `
    <h2>Run ${esc(RUN)}</h2>
    <p><a href="${base}graph">Graph of the pipeline →</a></p>
    <dl><dt>Records</dt><dd>${num(s.records)}</dd>
      <dt>Outputs</dt><dd><a href="${base}outputs">${num(s.outputs)}</a></dd>
      <dt>Input files</dt><dd>${num(s.files)}</dd>
      <dt>Library</dt><dd>${esc(m.library_version)}</dd></dl>
    <h3>Protocols</h3>
    <table>${(m.protocols || []).map((p) => `<tr><td>${esc(p.protocol)}</td>
      <td class="id muted" title="${esc(p.fingerprint)}">${short(p.fingerprint)}</td></tr>`).join("")}</table>
    <h3>Parameters</h3>
    <dl>${Object.entries(m.parameters || {}).map(([k, v]) =>
      `<dt>${esc(k)}</dt><dd>${esc(JSON.stringify(v))}</dd>`).join("")}</dl>
    <h3>Inputs</h3>
    <dl>${Object.entries(m.inputs || {}).map(([k, v]) =>
      `<dt>${esc(k)}</dt><dd>${v.values.map(esc).join("<br>")}${v.count > v.values.length
        ? `<br><span class="muted">… ${num(v.count)} in all</span>` : ""}</dd>`).join("")}</dl>
    <h3>Stages</h3>
    <table><tr><th>Stage</th><th class="num">Items</th><th>Input rows</th></tr>
    ${s.stages.map((st) => `<tr><td>${stageLink(st.label)}<br>
      <span class="muted">${st.inherited === st.items && st.items ? "keeps the elements of its items" : ""}</span></td>
      <td class="num">${num(st.items)}</td>
      <td title="${num(st.input_elements)} of ${num(st.elements)} elements are input rows">
        ${share(st.input_elements, st.elements)}</td></tr>`).join("")}
    </table>`;
}

const views = {
  async outputs([offset = 0]) {
    const d = await api(`outputs?offset=${offset}`);
    return `<h2>Outputs</h2><p class="muted">Items no stage made anything from: the
      outputs of the run, and items a stage absorbed without emitting.</p>
      ${pager(d, (o) => `${base}outputs/${o}`)}${itemsTable(d.items)}`;
  },
  async stage([label, offset = 0]) {
    const d = await api(`stage?label=${encodeURIComponent(label)}&offset=${offset}`);
    return `<h2>${esc(d.label)}</h2>
      ${pager(d, (o) => `${base}stage/${encodeURIComponent(label)}/${o}`)}${itemsTable(d.items)}`;
  },
  async item([id]) {
    const d = await api(`item/${id}`);
    const list = (items, count) => items.map(itemLink).join(" ") +
      (count > items.length ? ` <span class="muted">… ${num(count)} in all</span>` : "");
    return `<h2>Item <span class="id">${esc(d.id)}</span></h2>
      <dl><dt>Stage</dt><dd>${stageLink(d.stage)} (${esc(d.kind)})</dd>
        <dt>Elements</dt><dd>${num(d.elements)}, ${num(d.input_elements)} input rows
          ${d.inherited ? '<span class="muted">(the elements of its parent)</span>' : ""}</dd>
        <dt>Parents</dt><dd>${list(d.parents, d.parent_count) || '<span class="muted">none</span>'}</dd>
        <dt>Children</dt><dd>${list(d.children, d.child_count) || '<span class="muted">none</span>'}</dd>
        ${d.file_count ? `<dt>Files</dt><dd>${d.files.map(esc).join("<br>")}${d.file_count > d.files.length
          ? `<br>… ${num(d.file_count)} in all` : ""}</dd>` : ""}</dl>
      <p><a href="${base}ancestry/${d.id}">Ancestry: what it was made from →</a><br>
        On the graph: <a href="${base}graph/ancestry/${d.id}">ancestry</a> ·
        <a href="${base}graph/descendants/${d.id}">descendants</a></p>
      <h3>Elements</h3>
      <table><tr><th>Where</th><th class="num">Elements</th><th>Input rows</th></tr>
      ${d.groups.map((g) => `<tr><td><a href="${base}elements/${d.id}/${g.index}">${esc(g.where || "(item)")}${
        g.path ? "." + esc(g.path) : ""}</a></td><td class="num">${num(g.elements)}</td>
        <td>${share(g.input_elements, g.elements)}</td></tr>`).join("")}</table>`;
  },
  async elements([id, group, offset = 0]) {
    const [item, d] = await Promise.all([api(`item/${id}`),
      api(`elements/${id}?group=${group}&offset=${offset}`)]);
    const g = item.groups[group];
    return `<h2>${esc(g.where || "(item)")}${g.path ? "." + esc(g.path) : ""}</h2>
      <p>of item ${itemLink(item)} in ${esc(item.stage)}</p>
      ${pager(d, (o) => `${base}elements/${id}/${group}/${o}`)}
      <table><tr><th class="num">#</th><th>Element</th></tr>
      ${d.elements.map((e, i) => `<tr><td class="num">${num(d.offset + i)}</td>
        <td>${elementLink(e)}</td></tr>`).join("")}</table>`;
  },
  async ancestry([id]) {
    const d = await api(`ancestry/${id}`);
    return `<h2>Ancestry of ${itemLink(d.item)}</h2>
      <p><a href="${base}graph/ancestry/${id}">Show it on the graph →</a></p>
      <p>${num(d.items)} items, ${num(d.source_count)} source items, ${num(d.file_count)} input files
      ${d.missing.length ? `<span class="error">, ${d.missing.length} parents without a record</span>` : ""}</p>
      <h3>Stages, nearest first</h3>
      <table><tr><th>Stage</th><th class="num">Items</th><th>Fingerprint</th></tr>
      ${d.stages.map((s) => `<tr><td>${stageLink(s.label)}</td><td class="num">${num(s.items)}</td>
        <td class="id muted" title="${esc(s.stage_fp)}">${short(s.stage_fp)}</td></tr>`).join("")}</table>
      <h3>Source items</h3><p>${d.sources.map(itemLink).join(" ")}${d.source_count > d.sources.length
        ? ` <span class="muted">… ${num(d.source_count)} in all</span>` : ""}</p>
      <h3>Input files</h3><p>${d.files.map(esc).join("<br>")}${d.file_count > d.files.length
        ? `<br><span class="muted">… ${num(d.file_count)} in all</span>` : ""}</p>`;
  },
  async element([namespace, position]) {
    const d = await api(`element?namespace=${namespace}&position=${position}`);
    const origin = d.kind === "input"
      ? `<span class="input">row ${num(d.position)} of input file</span>
         <a href="${base}file/${d.file_index}">${esc(d.file)}</a>`
      : d.created_by ? `<span class="created">created</span> in ${itemLink(d.created_by)} by ${esc(d.created_by.stage)}`
      : '<span class="muted">unknown</span>';
    return `<h2>Element ${elementLink(d)}</h2>
      <p><a href="${base}graph/element/${d.namespace}/${d.position}">Show the items holding it on the graph →</a></p>
      <dl><dt>Origin</dt><dd>${origin}</dd><dt>Namespace</dt><dd class="id">${esc(d.namespace)}</dd>
        <dt>Held by</dt><dd>${num(d.holders)} items</dd>
        <dt>Outputs</dt><dd>${d.outputs.map(itemLink).join(" ") || '<span class="muted">none</span>'}</dd></dl>
      <h3>Items holding it, by stage</h3>
      <table>${d.stages.map((s) => `<tr><td>${stageLink(s.label)}</td>
        <td>${s.items.map(itemLink).join(" ")}${s.count > s.items.length ? ` … ${num(s.count)}` : ""}</td></tr>`).join("")}</table>`;
  },
  async file([index, offset = 0]) {
    const d = await api(`file/${index}?offset=${offset}`);
    return `<h2>${esc(d.path)}</h2>
      <dl><dt>Size</dt><dd>${num(d.size)} bytes</dd><dt>Namespace</dt><dd class="id">${esc(d.namespace)}</dd>
        <dt>Rows read</dt><dd>${num(d.total)}</dd></dl>
      ${pager(d, (o) => `${base}file/${index}/${o}`)}
      <table><tr><th>Row</th></tr>${d.rows.map((r) => `<tr><td>
        <a href="${base}element/${d.namespace}/${r}">row ${num(r)}</a></td></tr>`).join("")}</table>`;
  },
  async graph([kind, ...args]) {
    const g = await opsGraph();
    if (!g.svg) return `<h2>Graph</h2><p class="muted">The run recorded no graph of its
      pipeline: runs of RayPipelineRunner with --run-dir record it.</p>`;
    let overlay = null, heading = "Graph of the pipeline", note =
      "Badges count the items every stage made. Open an item or an element to show its lineage.";
    if (kind === "element") {
      overlay = await api(`overlay/element?namespace=${args[0]}&position=${args[1]}`);
      heading = `Items holding ${elementLink(overlay.subject)}`;
    } else if (kind) {
      overlay = await api(`overlay/${kind}/${args[0]}`);
      heading = `${kind === "ancestry" ? "Ancestry" : "Descendants"} of ${itemLink(overlay.subject)}`;
    }
    if (overlay) note = `${num(overlay.items)} items. Badges count them per stage; highlighted
      edges carry links between them (their number in the tooltip).` +
      (overlay.unplaced.length ? ` <span class="error">${overlay.unplaced.length} links between
      stages the graph lacks.</span>` : "");
    setTimeout(() => paintGraph(g, overlay), 0);
    return `<h2>${heading}</h2><p class="muted">${note}</p>
      <div class="legend"><label><input type="checkbox" id="fit" checked> Fit to width</label>
      ${overlay ? `<a href="${base}graph">Without lineage</a>` : ""}</div>
      <div id="ops" class="fit">${g.svg}</div><div id="node-info"></div>`;
  },
  async search([q]) {
    const d = await api(`search?q=${encodeURIComponent(q)}`);
    return `<h2>Search: ${esc(q)}</h2>
      <h3>Items</h3><p>${d.items.map((it) => itemLink(it) + " " + esc(it.stage)).join("<br>") || '<span class="muted">none</span>'}</p>
      <h3>Input files</h3><p>${d.files.map((f) => `<a href="${base}file/${f.index}">${esc(f.path)}</a>`).join("<br>") || '<span class="muted">none</span>'}</p>`;
  },
};

let opsCache = null, opsRun = null;
async function opsGraph() {
  if (!opsCache || opsRun !== RUN) { opsCache = await api("graph"); opsRun = RUN; }
  return opsCache;
}
const SVG = "http://www.w3.org/2000/svg";
function badge(node, text) {
  const box = node.getBBox();
  const label = document.createElementNS(SVG, "text");
  label.setAttribute("class", "badge-text");
  label.setAttribute("text-anchor", "middle");
  label.textContent = text;
  node.appendChild(label);
  const width = Math.max(18, label.getComputedTextLength() + 10);
  const x = box.x + box.width - width / 2 + 4, y = box.y - 2;
  label.setAttribute("x", x); label.setAttribute("y", y + 4);
  const pill = document.createElementNS(SVG, "rect");
  pill.setAttribute("class", "badge");
  pill.setAttribute("x", x - width / 2); pill.setAttribute("y", y - 9);
  pill.setAttribute("width", width); pill.setAttribute("height", 17);
  pill.setAttribute("rx", 8.5);
  node.insertBefore(pill, label);
}
function paintGraph(g, overlay) {
  const ops = $("ops");
  if (!ops) return;
  $("fit").addEventListener("change", (e) => ops.classList.toggle("fit", e.target.checked));
  const counts = overlay ? overlay.counts : g.counts;
  ops.classList.toggle("dim", Boolean(overlay));
  // Room above the top nodes for their badges.
  const svg = ops.querySelector("svg");
  const box = svg.viewBox.baseVal;
  if (box && box.height) {
    box.y -= 16; box.height += 16;
    const height = parseFloat(svg.getAttribute("height"));
    if (height) svg.setAttribute("height", `${height + 16}pt`);
  }
  (overlay ? overlay.through : []).forEach((id) => {
    const node = document.getElementById(id);
    if (node) node.classList.add("lit", "through");
  });
  const nodes = Object.fromEntries(g.nodes.map((n) => [n.svg_id, n]));
  ops.querySelectorAll("g.node").forEach((node) => {
    const count = counts[node.id];
    if (count) { node.classList.add("lit"); badge(node, num(count)); }
    node.addEventListener("click", () => showNode(nodes[node.id], count, node));
  });
  const edges = overlay ? overlay.edges : {};
  const widest = Math.max(1, ...Object.values(edges));
  ops.querySelectorAll("g.edge").forEach((edge) => {
    const count = edges[edge.id];
    if (!count) return;
    edge.classList.add("lit");
    const width = 1.5 + 4 * Math.log1p(count) / Math.log1p(widest);
    edge.querySelectorAll("path").forEach((p) => p.style.strokeWidth = width);
    const title = document.createElementNS(SVG, "title");
    title.textContent = `${num(count)} links`;
    edge.appendChild(title);
  });
}
function showNode(node, count, element) {
  document.querySelectorAll("#ops .selected").forEach((n) => n.classList.remove("selected"));
  element.classList.add("selected");
  $("node-info").innerHTML = `<h3>${esc(node.title)}</h3>
    <dl><dt>Kind</dt><dd>${esc(node.kind)}</dd><dt>Items</dt><dd>${num(count || 0)}</dd>
      ${node.stage_fp ? `<dt>Fingerprint</dt><dd class="id">${esc(node.stage_fp)}</dd>` : ""}
      <dt>Stages</dt><dd>${node.stages.map(stageLink).join("<br>") ||
        '<span class="muted">none recorded (it passes items on)</span>'}</dd></dl>`;
}

let runs = [];
function ago(seconds) {
  const s = Date.now() / 1000 - seconds;
  return s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago`
    : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} days ago`;
}
async function showRuns() {
  runs = await get("/api/runs");
  $("runs").innerHTML = `<h2>Runs</h2>` + (runs.length ? `<ul class="runs">${runs.map((r) => `
    <li class="${r.name === RUN ? "current" : ""}">
      <a href="#/${encodeURIComponent(r.name)}/outputs">${
        r.updated && Date.now() / 1000 - r.updated < 60 ? '<span class="live" title="Wrote records in the last minute"></span>' : ""
      }${esc(r.name)}</a>
      <div class="meta">${runMeta(r)}</div>
      <div class="meta">${r.protocols.map(esc).join(" → ")}</div></li>`).join("")}</ul>`
    : '<p class="muted">No runs found: run directories hold a manifest.json or a lineage/ directory.</p>');
}
const created = (r) => (r.created || "").replace("T", " ").slice(0, 19);
const runMeta = (r) => [
  created(r),
  r.updated ? "updated " + ago(r.updated) : "",
  `${num(r.parts)} record file${r.parts === 1 ? "" : "s"}`,
].filter(Boolean).map(esc).join(" · ");
function runsPage() {
  return `<h2>Runs</h2><p class="muted">Pick a run in the list.</p>
    <table><tr><th>Run</th><th>Created</th><th>Protocols</th><th class="num">Record files</th></tr>
    ${runs.map((r) => `<tr><td><a href="#/${encodeURIComponent(r.name)}/outputs">${esc(r.name)}</a></td>
      <td>${esc(created(r))}</td>
      <td>${r.protocols.map(esc).join(" → ")}</td><td class="num">${num(r.parts)}</td></tr>`).join("")}
    </table>`;
}
async function route() {
  const [run, name, ...args] = location.hash.slice(2).split("/").map(decodeURIComponent);
  if (!runs.some((r) => r.name === run)) {
    await showRuns();
    if (runs.length === 1) { location.hash = `#/${encodeURIComponent(runs[0].name)}/outputs`; return; }
    if (!runs.some((r) => r.name === run)) {
      RUN = null; $("nav").innerHTML = ""; $("run").textContent = "";
      $("page").innerHTML = runsPage();
      return;
    }
  }
  if (run !== RUN) {
    RUN = run; base = `#/${encodeURIComponent(run)}/`;
    $("nav").innerHTML = '<p class="muted">Loading the run…</p>';
    await showRuns();
    showNav().catch((error) => { $("nav").innerHTML = `<p class="error">${esc(error.message)}</p>`; });
  }
  const view = views[name] || views.outputs;
  $("page").innerHTML = '<p class="muted">Loading…</p>';
  try { $("page").innerHTML = await view(args); }
  catch (error) { $("page").innerHTML = `<p class="error">${esc(error.message)}</p>`; }
}
$("search").addEventListener("submit", (event) => {
  event.preventDefault();
  location.hash = base + "search/" + encodeURIComponent($("q").value);
});
$("reload").addEventListener("click", async () => {
  await showRuns();
  if (RUN === null) { $("page").innerHTML = runsPage(); return; }
  await api("reload"); opsCache = null; await showNav(); await route();
});
window.addEventListener("hashchange", route);
// New runs, and runs that are writing, show up like in TensorBoard.
setInterval(() => showRuns().catch(() => {}), 30000);
route();
</script></body></html>
"""


if __name__ == "__main__":
    main()
