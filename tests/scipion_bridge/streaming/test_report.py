import pytest

from scipion_bridge.core.environment.compute import ComputeAssignment, ComputeResources
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    GpuMapPanel,
    GpuOccupant,
    GpuSlot,
    IRDemux,
    IRMap,
    IRSink,
    IRSource,
    Layer,
    PipelineReport,
    Source,
    TablePanel,
    lower,
    report_of,
)


def _forward(x):
    return x


def _graph():
    """source -> map(gpu protocol) -> group_by(map) -> sink, built as IR."""
    source = IRSource(name="particles")
    forward = IRMap(
        func=_forward,
        compute=ComputeAssignment(ComputeResources(gpus=1), "align"),
        stage_fp="fp-forward",
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


def test_report_lists_the_stages_with_the_labels_of_the_backends():
    report = report_of(_graph())

    assert [(n.id, n.kind, n.parent) for n in report.nodes] == [
        ("0:source(particles)", "source", None),
        ("1:map(_forward)", "map", None),
        ("2:group_by", "group_by", None),
        ("2:group_by/0:source(group)", "source", "2:group_by"),
        ("2:group_by/1:refine", "map", "2:group_by"),
        ("3:sink(CallbackSinkWriter)", "sink", None),
    ]
    forward = report.nodes[1]
    assert (forward.group, forward.stage_fp) == ("align", "fp-forward")
    assert report.groups == ("align",)


def test_report_edges_keep_their_ports():
    source = Source("a")
    merged = source.map(lambda x: x).merge(Source("b").map(lambda x: x))
    report = report_of(lower([merged.sink(lambda item: None)]))

    into_merge = [edge for edge in report.edges if edge.target.endswith(":merge")]
    assert sorted(edge.port for edge in into_merge) == [0, 1]


def test_report_includes_template_edges():
    report = report_of(_graph())

    assert ("2:group_by/0:source(group)", "2:group_by/1:refine") in {
        (edge.source, edge.target) for edge in report.edges
    }


def _layer(report: PipelineReport) -> Layer:
    first, second = report.nodes[0].id, report.nodes[1].id
    return Layer(
        name="placement",
        title="Placement",
        nodes={second: {"phase": 0, "slots": (0, 1)}},
        edges={(first, second): {"boundary": "spills"}},
        groups={"align": {"replicas": 2}},
        metric="phase",
        panels=(
            TablePanel("Groups", ("group", "replicas"), (("align", 2),)),
            GpuMapPanel("Phase 0", (GpuSlot(0, 0, (GpuOccupant("align", 0.5),)),)),
        ),
        warnings=("align shares GPU 0",),
    )


def test_layers_round_trip_through_json():
    report = report_of(_graph())
    report = report.with_layer(_layer(report))

    assert PipelineReport.from_json(report.to_json()) == report
    assert report.warnings == ("align shares GPU 0",)
    assert report.layer("placement").metric == "phase"


def test_json_is_stable():
    report = report_of(_graph())
    report = report.with_layer(_layer(report))

    assert report.to_json() == PipelineReport.from_json(report.to_json()).to_json()


@pytest.mark.parametrize(
    "layer",
    [
        Layer("x", "X", nodes={"9:missing": {}}),
        Layer(
            "x", "X", edges={("0:source(particles)", "3:sink(CallbackSinkWriter)"): {}}
        ),
        Layer("x", "X", groups={"unknown": {}}),
    ],
)
def test_layers_annotate_only_what_the_graph_has(layer):
    with pytest.raises(ValueError, match="does not have"):
        report_of(_graph()).with_layer(layer)


def test_layer_names_are_unique():
    report = report_of(_graph()).with_layer(Layer("x", "X"))

    with pytest.raises(ValueError, match="already"):
        report.with_layer(Layer("x", "Y"))


def test_unknown_layer_raises():
    with pytest.raises(KeyError):
        report_of(_graph()).layer("cost")
