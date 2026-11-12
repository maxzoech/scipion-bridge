"""Stats layers and the renderers of pipeline reports, without Ray."""

import re
import sys
from typing import Any, Dict

import pytest

import scipion_bridge as B
from scipion_bridge import Protocol
from scipion_bridge.backend.ray import RayPipelineRunner
from scipion_bridge.core.environment.compute import ComputeAssignment, ComputeResources
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    CompiledPipeline,
    IRDemux,
    IRMap,
    IRSink,
    IRSource,
    Layer,
    Pipeline,
    StageStats,
    TablePanel,
    GpuMapPanel,
    GpuSlot,
    GpuOccupant,
    report_of,
)
from scipion_bridge.core.streaming.backend import stats_layer
from scipion_bridge.core.streaming.report import (
    graphviz_available,
    render_dot,
    render_html,
    render_svg,
    render_text,
)
from scipion_bridge.core.streaming.report.drawing import drawing_of
from scipion_bridge.core.streaming.report.layout import boxes_of, layers_of
from scipion_bridge.single_particle import Particle


def _forward(x):
    return x


def _ir():
    """source -> map(align) -> group_by(source -> refine) -> sink."""
    source = IRSource(name="particles")
    forward = IRMap(
        func=_forward,
        compute=ComputeAssignment(ComputeResources(gpus=1), "id-1", name="Align"),
    )
    template_source = IRSource(name="group")
    template_map = IRMap(func=_forward, name="refine")
    template_source.add_downstream(template_map)
    demux = IRDemux(template=template_map, source_name="group")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(forward)
    forward.add_downstream(demux)
    demux.add_downstream(sink)
    return [sink]


def _report():
    report = report_of(_ir())
    first, second = report.nodes[0].id, report.nodes[1].id
    return report.with_layer(
        Layer(
            name="placement",
            title="Placement",
            nodes={second: {"phase": 1, "replicas": 2}},
            edges={(first, second): {"boundary": "spills"}},
            groups={"id-1": {"gpus": 1.0}},
            metric="phase",
            panels=(
                TablePanel("Groups", ("group", "replicas"), (("Align", 2),)),
                GpuMapPanel("Phase 1", (GpuSlot(0, 0, (GpuOccupant("Align", 0.5),)),)),
            ),
            warnings=("Align waits for FLUSH",),
        ),
    )


# -- The stats layer ---------------------------------------------------------
def test_stats_annotate_the_stages():
    report = report_of(_ir())
    layer = stats_layer(report, {"1:map(_forward)": StageStats(items_in=3)})

    assert layer.nodes["1:map(_forward)"]["items_in"] == 3
    assert layer.metric == "process_s"


def test_stats_of_group_by_children_add_up_in_the_template():
    report = report_of(_ir())
    layer = stats_layer(
        report,
        {
            "2:group_by[a]/1:refine": StageStats(items_in=2, buffered_peak=4),
            "2:group_by[b]/1:refine": StageStats(items_in=1, buffered_peak=7),
            # Stages the report does not have: the sinks of the children.
            "2:group_by[a]/2:sink(_DemuxResultWriter)": StageStats(items_in=2),
        },
    )

    assert layer.nodes["2:group_by/1:refine"] == {
        **layer.nodes["2:group_by/1:refine"],
        "items_in": 3,
        "buffered_peak": 7,
        "children": 2,
    }
    assert set(layer.nodes) == {"2:group_by/1:refine"}


class _CompiledWithStats(CompiledPipeline):
    def send(self, source_name: str, value: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def stats(self) -> Dict[str, StageStats]:
        return {"1:map(_forward)": StageStats(items_in=5)}


def test_pipeline_reports_its_graph_and_stats():
    report = Pipeline(_CompiledWithStats(), ir_sinks=_ir()).report()

    assert len(report.nodes) == 6
    assert report.layer("stats").nodes["1:map(_forward)"]["items_in"] == 5


# -- Text and DOT ------------------------------------------------------------
def test_text_lists_stages_annotations_panels_and_warnings():
    text = render_text(_report())

    assert "  1:map(_forward)  [Align]  placement: phase=1 replicas=2" in text
    assert "    2:group_by/1:refine" in text
    assert "Align (placement): gpus=1" in text
    assert "node 0 GPU 0: Align 50%" in text
    assert "  - Align waits for FLUSH" in text


def test_text_shows_only_the_chosen_layers():
    assert "placement:" not in render_text(_report(), layers=[])


def test_dot_has_the_ids_classes_and_clusters_of_the_html():
    dot = render_dot(_report())

    assert 's1 [id="s1", class="kind-map group-0"' in dot
    assert "subgraph cluster_2 {" in dot
    # The phase boundary is dashed; the edges through the template are virtual.
    assert 's0 -> s1 [id="e0", class="boundary", style=dashed];' in dot
    assert 's2 -> s3 [id="v0", class="virtual"' in dot
    assert 's4 -> s5 [id="v1", class="virtual"' in dot


def test_group_labels_name_protocols_and_tell_instances_apart():
    a = IRMap(
        func=_forward,
        compute=ComputeAssignment(ComputeResources(), "aaaaaa1", name="P"),
    )
    b = IRMap(
        func=_forward,
        compute=ComputeAssignment(ComputeResources(), "bbbbbb2", name="P"),
    )
    c = IRMap(func=_forward, compute=ComputeAssignment(ComputeResources(), "cccccc3"))
    source = IRSource(name="x")
    source.add_downstream(a)
    a.add_downstream(b)
    b.add_downstream(c)

    assert report_of([c]).group_labels == {
        "aaaaaa1": "P #aaaaaa",
        "bbbbbb2": "P #bbbbbb",
        "cccccc3": "cccccc3",
    }


# -- SVG and HTML ------------------------------------------------------------
def _svg_ids(svg: str) -> set:
    return set(re.findall(r'<g id="([sev]\d+)"', svg))


@pytest.mark.parametrize(
    "layout",
    [
        "builtin",
        pytest.param(
            "graphviz",
            marks=pytest.mark.skipif(not graphviz_available(), reason="no Graphviz"),
        ),
    ],
)
def test_html_is_self_contained_with_every_stage_and_edge(layout):
    report = _report()
    page = render_html(report, layout)

    without_namespaces = page.replace('"http://www.w3.org/', "")
    assert "http" not in re.sub(r"https?://www\.w3\.org/\S*", "", without_namespaces)
    assert {f"s{i}" for i in range(len(report.nodes))} <= _svg_ids(page)
    assert {f"e{i}" for i in range(len(report.edges))} <= _svg_ids(page)
    assert '<script type="application/json" id="report-data">' in page
    assert "Phase 1" in page and "Align waits for FLUSH" in page


@pytest.mark.skipif(not graphviz_available(), reason="no Graphviz")
def test_both_layouts_draw_the_same_ids():
    report = _report()

    assert _svg_ids(render_svg(report, "graphviz")) == _svg_ids(
        render_svg(report, "builtin")
    )


def test_graphviz_layout_without_graphviz_raises(monkeypatch):
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="Graphviz"):
        render_svg(_report(), "graphviz")


def test_builtin_layout_has_no_overlaps_and_edges_go_down():
    drawing = drawing_of(_report())
    boxes = boxes_of(drawing)

    for layer in layers_of(drawing):
        ordered = sorted((boxes[n] for n in layer), key=lambda box: box.left)
        assert all(a.right <= b.left for a, b in zip(ordered, ordered[1:]))
    assert all(
        boxes[edge.source].bottom < boxes[edge.target].top for edge in drawing.edges
    )


# -- The runner --------------------------------------------------------------
@B.resources(gpus=1)
class ExplainedProtocol(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input(label="Particles", optional=False)

    def outputs(self):
        return {"output": B.Set[Particle]}

    def _infer(self, particles):
        return {"output": particles}

    def steps(self):
        return self.particles.map(self._infer)


def test_explain_writes_the_report_without_starting_ray(tmp_path, monkeypatch, capsys):
    path = tmp_path / "report.html"
    monkeypatch.setattr(sys, "argv", ["prog", "--explain", str(path)])

    RayPipelineRunner(ExplainedProtocol()).launch_as_terminal_application()

    assert "[ExplainedProtocol]" in capsys.readouterr().out
    assert "ExplainedProtocol" in path.read_text()
    assert path.with_suffix(".dot").read_text().startswith("digraph pipeline {")
