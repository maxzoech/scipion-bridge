"""The provenance samples of stages and the provenance layer of reports (in-process).

The samples are recorded from real Sets whose ``pixels`` reference rows of a
source; the layer is built from samples as the backends collect them.
"""

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    IRDemux,
    IRMap,
    IRSink,
    IRSource,
    Layer,
    Source,
    lower,
    report_of,
)
from scipion_bridge.core.streaming.report import render_dot, render_svg
from scipion_bridge.core.streaming.provenance import (
    SAMPLES,
    ProvenanceSampler,
    StageSamples,
    prefetch_all,
    sample_of,
    sets_of,
)
from scipion_bridge.core.streaming.report.drawing import LOST_COLOR
from scipion_bridge.core.streaming.report.provenance import provenance_layer
from scipion_bridge.core.struct import references
from scipion_bridge.core.struct.source import ArraySource, Reference


class Image(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(4, 4))
    defocus: float


def _particles(n=8, read=True):
    """A Set whose pixels reference rows of a source."""
    s = B.Set[Image](capacity=n)
    s["defocus"] = np.arange(n, dtype=np.float64)
    rows = np.random.rand(n, 4, 4).astype(np.float32)
    references.set_reference(s, ("pixels",), Reference(ArraySource(rows), range(n)))
    if read:
        references.prefetch(s)
    return s


def _rebuilt(s, pixels=None):
    """A new Set with the values of ``s``: its pixels are owned."""
    out = B.Set[Image](capacity=len(s))
    out["defocus"] = s["defocus"]
    out["pixels"] = np.array(s["pixels"]) if pixels is None else pixels
    return out


def _keep(s):
    return s


def _discard(item):
    pass


def _samples(fn, items):
    sampler = ProvenanceSampler()
    for item in items:
        sampler.receive(item)
        sampler.emit(item, [fn(item)])
    return sampler.samples


# Telling copies apart hashes columns, only with __debug__ (no ``python -O``).
needs_debug = pytest.mark.skipif(
    not __debug__, reason="copies are checked with __debug__"
)


def _states(sample):
    return {(s.where, c.path): c.state for s in sample for c in s.columns}


# -- Sampling ------------------------------------------------------------------
def test_sets_are_found_in_items_with_where_they_are():
    s = _particles()

    found = [where for where, _ in sets_of({"out": (s, 1.0), "n": 3, "more": [s]})]

    assert found == ["['out'][0]", "['more'][0]"]


def test_the_first_items_are_sampled_without_reading_references():
    items = [{"out": _particles(read=False)} for _ in range(SAMPLES + 2)]

    samples = _samples(lambda item: item, items)

    assert len(samples.inputs) == len(samples.outputs) == SAMPLES
    assert _states(samples.outputs[0]) == {
        ("['out']", ("pixels",)): "reference",
        ("['out']", ("defocus",)): "owned",
    }
    assert all(
        references.column_states(i["out"])[("pixels",)].state == "reference"
        for i in items
    )


def test_items_emitted_on_flush_are_sampled_as_outputs():
    sampler = ProvenanceSampler()
    for _ in range(SAMPLES):
        sampler.receive(_particles())
        sampler.emit(None, [])
    sampler.emit(None, [_particles(), _particles()])

    assert (len(sampler.samples.inputs), len(sampler.samples.outputs)) == (SAMPLES, 2)


@needs_debug
def test_copied_references_are_found_unchanged():
    s = _particles()
    changed = np.zeros((8, 4, 4), dtype=np.float32)

    (copied,) = _samples(_rebuilt, [s]).outputs[0]
    (computed,) = _samples(lambda x: _rebuilt(x, changed), [s]).outputs[0]

    assert copied.unchanged == (("pixels",),)
    assert computed.unchanged == ()


def test_the_elements_of_collections_are_sampled():
    classes = B.Collection[Image](
        size=2,
        items=[
            Image(pixels=np.ones((4, 4), dtype=np.float32), defocus=float(i))
            for i in range(2)
        ],
    )

    sample = sample_of({"clusters": classes, "particles": _particles()})

    assert [s.where for s in sample] == [
        "['clusters'][0]",
        "['clusters'][1]",
        "['particles']",
    ]
    assert {c.state for c in sample[0].columns} == {"owned"}


def test_prefetch_all_reads_the_sets_inside_items():
    lazy = _particles(read=False)
    classes = B.Collection[Image](size=1)
    item = {"out": [lazy, (lazy[1:3],)], "n": 3, "classes": classes}

    prefetch_all(item)

    assert references.column_states(lazy)[("pixels",)].state == "cached"


# -- The layer -----------------------------------------------------------------
def _chain():
    """source -> map(keep) -> map(rebuild) -> sink, and its node ids."""
    sink = Source("particles").map(_keep).map(_rebuilt).sink(_discard)
    report = report_of(lower([sink]))
    return report, [node.id for node in report.nodes]


def _chain_samples(ids):
    source, keep, rebuild, sink = ids
    items = [_particles() for _ in range(SAMPLES)]
    kept = _samples(_keep, items)
    rebuilt = _samples(_rebuilt, items)
    return {
        source: kept,
        keep: kept,
        rebuild: rebuilt,
        sink: StageSamples(inputs=rebuilt.outputs),
    }


@needs_debug
def test_the_layer_shows_where_references_are_lost():
    report, ids = _chain()
    source, keep, rebuild, sink = ids

    layer = provenance_layer(report, _chain_samples(ids))

    # 64 B of pixels and 8 B of defocus per row.
    assert layer.nodes[keep]["reference_fraction"] == pytest.approx(64 / 72)
    assert layer.nodes[keep]["computed_here"] == ()
    assert layer.nodes[rebuild]["reference_fraction"] == 0.0
    assert layer.nodes[rebuild]["computed_here"] == ("pixels",)
    assert layer.nodes[rebuild]["unchanged"] == ("pixels",)
    assert layer.edges[(keep, rebuild)]["lost"] is True
    assert "lost" not in layer.edges[(source, keep)]
    (warning,) = layer.warnings
    assert "pixels" in warning and "unchanged" in warning


def test_key_paths_show_the_state_of_referenced_paths_at_every_stage():
    report, ids = _chain()

    layer = provenance_layer(report, _chain_samples(ids))

    (panel,) = layer.panels
    assert panel.columns == ("stage", "pixels", "other paths")
    assert [row[1:] for row in panel.rows] == [
        ("ref*", 1),
        ("ref*", 1),
        ("C", 1),
        ("own", 1),
    ]


def test_reading_a_reference_is_shown_where_it_happens():
    report, ids = _chain()
    source, keep, _, _ = ids
    sampler = ProvenanceSampler()
    for item in [_particles(read=False) for _ in range(SAMPLES)]:
        sampler.receive(item)
        references.prefetch(item)
        sampler.emit(item, [item])

    layer = provenance_layer(report, {keep: sampler.samples})

    assert layer.nodes[keep]["read_here"] == ("pixels",)


def test_spills_at_phase_boundaries_are_estimated():
    report, ids = _chain()
    source, keep, rebuild, _ = ids
    report = report.with_layer(
        Layer(
            name="placement",
            title="Placement",
            edges={(keep, rebuild): {"boundary": "spills"}},
        ),
    )

    layer = provenance_layer(report, _chain_samples(ids))

    spill = next(p for p in layer.panels if p.title == "Spill estimate")
    ((_, total, spilled, saved),) = spill.rows
    # The pixels (64 B per row) stay in their source; the defocus is written.
    assert (total, spilled, saved) == ("576 B", "64 B", "89%")
    assert layer.edges[(keep, rebuild)]["spill_bytes"] == 64


def test_samples_of_group_by_children_are_merged_into_the_template():
    template_source = IRSource(name="group")
    template_map = IRMap(func=_rebuilt, name="rebuild")
    template_source.add_downstream(template_map)
    source = IRSource(name="particles")
    demux = IRDemux(template=template_map, source_name="group")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(demux)
    demux.add_downstream(sink)
    report = report_of([sink])
    node = next(n for n in report.nodes if n.title.endswith("rebuild"))
    assert node.parent is not None
    rest = node.id[len(node.parent) :]

    layer = provenance_layer(
        report,
        {
            f"{node.parent}[{key}]{rest}": _samples(_rebuilt, [_particles()])
            for key in ("a", "b")
        },
    )

    assert layer.nodes[node.id]["computed_here"] == ("pixels",)


def test_lost_edges_are_drawn_red():
    report, ids = _chain()
    report = report.with_layer(provenance_layer(report, _chain_samples(ids)))

    assert 'class="lost"' in render_dot(report)
    assert f'stroke="{LOST_COLOR}"' in render_svg(report)
