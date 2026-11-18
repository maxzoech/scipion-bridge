"""The lineage viewer: its pages, and its server (in-process)."""

import json
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge import single_particle as spa
from scipion_bridge.core.streaming import (
    Commit,
    ElementSegment,
    FileRunStore,
    LineageRecord,
    Source,
    StageInfo,
    lower,
)
from scipion_bridge.core.streaming.lineage import element_segments
from scipion_bridge.core.streaming.report import lineage_view, preview, report_of
from scipion_bridge.core.streaming.report.lineage_view import LineageView
from scipion_bridge.core.struct import concat, identity
from scipion_bridge.core.struct.id_column import IdColumn
from scipion_bridge.core.struct.source import FileId
from scipion_bridge.core.utils.hashing import item_id

S1, S2, M1, O1 = (bytes([k]) * 16 for k in range(1, 5))
A = FileId(Path("/data/a.mrcs"), 100, 7)
B_FILE = FileId(Path("/data/b.mrcs"), 100, 8)


class Value(B.Struct):
    value: float


def _values(column):
    s = B.Set[Value](capacity=len(column))
    s["value"] = np.zeros(len(column))
    identity.set_ids(s, column)
    return s


@pytest.fixture
def store(tmp_path):
    """Two source items, a map creating rows, and an output whose class holds
    rows of both stacks and created rows."""
    store = FileRunStore(tmp_path)
    created = item_id(M1, b"")
    members = concat(
        [
            _values(IdColumn.of(A.namespace, range(1, 3))),
            _values(IdColumn.of(B_FILE.namespace, range(0, 1))),
            _values(IdColumn.of(created, range(0, 2))),
        ],
    )
    store.write_manifest(
        {
            "run_id": "abc",
            "protocols": [{"protocol": "P", "fingerprint": "f"}],
            "parameters": {"scale": 3},
            "inputs": {"particles": {"value": ["a.xmd", "b.xmd"], "count": 2}},
        },
    )
    store.write_stages(
        "p",
        [
            StageInfo("0:source(x)", "IRSource", "fp0"),
            StageInfo("1:map(f)", "IRMap", "fp1"),
            StageInfo("2:map(g)", "IRMap", "fp2"),
        ],
    )
    store.append_lineage(
        [
            LineageRecord(
                S1,
                "0:source(x)",
                "source",
                (),
                (ElementSegment("", "", 4, A.namespace, 0, 4),),
                (A,),
            ),
            LineageRecord(
                S2,
                "0:source(x)",
                "source",
                (),
                (ElementSegment("", "", 2, B_FILE.namespace, 0, 2),),
                (B_FILE,),
            ),
            LineageRecord(
                M1,
                "1:map(f)",
                "map",
                (S1,),
                (ElementSegment("", "", 2, created, 0, 2),),
            ),
            LineageRecord(
                O1,
                "2:map(g)",
                "map",
                (M1, S2),
                element_segments({"classes": [members]}),
            ),
        ],
    )
    store.append_commits([Commit(O1, "3:sink(W)", 5)])
    return store


def test_the_summary_shows_the_run_and_its_stages(store):
    summary = LineageView(store).summary()

    assert summary["records"] == 4 and summary["outputs"] == 1
    assert summary["dead_ends"] == 0
    assert summary["manifest"]["inputs"]["particles"]["count"] == 2
    stages = {s["label"]: s for s in summary["stages"]}
    assert list(stages) == ["0:source(x)", "1:map(f)", "2:map(g)"]
    assert (stages["2:map(g)"]["elements"], stages["2:map(g)"]["input_elements"]) == (
        5,
        3,
    )
    assert stages["1:map(f)"]["stage_fp"] == "fp1"


def test_items_of_a_branch_leading_nowhere_are_dead_ends_not_outputs(store):
    unused = bytes([9]) * 16
    store.append_lineage(
        [
            LineageRecord(
                unused,
                "1:map(f)",
                "map",
                (S2,),
                (ElementSegment("", "", 2, item_id(unused, b""), 0, 2),),
            ),
        ],
    )

    view = LineageView(store)

    assert [o["id"] for o in view.outputs()["items"]] == [O1.hex()]
    assert [d["id"] for d in view.dead_end_page()["items"]] == [unused.hex()]


def test_the_items_sinks_wrote_are_marked(store):
    view = LineageView(store)

    assert view.summary()["sinks"] == [
        {"label": "3:sink(W)", "items": 1, "elements": 5}
    ]
    (output,) = view.outputs()["items"]
    assert output["written"] == ["3:sink(W)"]
    assert view.stage("1:map(f)")["items"][0]["written"] == []


class Average(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(8, 8))
    count: int


def test_checkpoints_list_their_items_and_preview_them(store):
    label = "3:checkpoint(classes)"
    average = Average(pixels=np.arange(64, dtype=np.float32).reshape(8, 8), count=5)
    store.begin_checkpoint(label, replay=True)
    store.store_item(
        label,
        O1,
        {"average": average, "members": _values(IdColumn.of(A.namespace, range(3)))},
    )
    store.append_commits([Commit(O1, label, 3)])

    view = LineageView(store)

    (checkpoint,) = view.summary()["checkpoints"]
    assert (checkpoint["label"], checkpoint["items"], checkpoint["complete"]) == (
        label,
        1,
        False,
    )
    (row,) = view.checkpoint(label)["items"]
    assert row["id"] == O1.hex() and row["kept"] == [label]
    assert view.item(O1)["kept"] == [label]
    preview = view.kept_item(label, O1)["preview"]
    fields = {f["name"]: f for f in preview["entries"]["average"]["fields"]}
    assert fields["count"]["value"] == [5]
    assert fields["pixels"]["thumbnail"].startswith("data:image/png;base64,")
    assert preview["entries"]["members"]["length"] == 3
    with pytest.raises(KeyError):
        view.kept_item(label, M1)


def test_unset_fields_are_previewed_as_unset():
    with_average = spa.Class2D(
        class_id=3,
        representative=spa.Particle(pixels=np.ones((8, 8), np.float32)),
    )
    without_average = spa.Class2D(class_id=4)
    classes = B.Collection[spa.Class2D](2, [with_average, without_average])

    entries = preview.preview_of(classes)["entries"]

    first = {f["name"]: f for f in entries["0"]["fields"]}
    assert first["representative/pixels"]["thumbnail"].startswith("data:image/png")
    assert first["representative/ctf/defocus_u"]["unset"]
    assert first["class_id"]["value"] == [3]
    second = {f["name"]: f for f in entries["1"]["fields"]}
    assert second["representative/pixels"]["unset"]


def test_thumbnails_are_valid_pngs():
    data = preview.png(np.arange(12, dtype=np.uint8).reshape(3, 4))

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (4, 3)


def test_an_item_shows_its_links_and_element_groups(store):
    view = LineageView(store)

    item = view.item(O1)

    assert [p["id"] for p in item["parents"]] == [M1.hex(), S2.hex()]
    assert item["children"] == []
    (group,) = item["groups"]
    assert (group["where"], group["elements"], group["input_elements"]) == (
        "['classes'][0]",
        5,
        3,
    )
    page = view.elements(O1, 0, offset=1, limit=3)
    assert page["total"] == 5
    assert [(e["kind"], e["position"]) for e in page["elements"]] == [
        ("input", 2),
        ("input", 0),
        ("created", 0),
    ]
    assert page["elements"][1]["file"] == str(B_FILE.path)


def test_an_element_shows_its_origin_and_holders(store):
    view = LineageView(store)

    row = view.element(A.namespace, 2)
    created = view.element(item_id(M1, b""), 1)

    assert row["kind"] == "input" and row["file"] == str(A.path)
    assert {s["label"] for s in row["stages"]} == {"0:source(x)", "2:map(g)"}
    assert [o["id"] for o in row["outputs"]] == [O1.hex()]
    assert created["created_by"]["id"] == M1.hex()
    # O1 holds rows of several files in one segment (a namespace table).
    assert view.element(B_FILE.namespace, 0)["holders"] == 2


def test_ancestry_search_and_files(store):
    view = LineageView(store)

    ancestry = view.ancestry(O1)
    assert ancestry["items"] == 4 and ancestry["source_count"] == 2
    assert {s["label"]: s["items"] for s in ancestry["stages"]}["0:source(x)"] == 2
    assert view.search(O1.hex()[:8])["items"][0]["id"] == O1.hex()
    (found,) = view.search("b.mrcs")["files"]
    assert view.file(found["index"])["rows"] == [0, 1]


@pytest.fixture
def server(store):
    ready = threading.Event()
    servers = []

    def started(server):
        servers.append(server)
        ready.set()

    thread = threading.Thread(
        target=lineage_view.serve,
        args=(store.path, "127.0.0.1", 0, started),
        daemon=True,
    )
    thread.start()
    ready.wait(timeout=30)
    (server,) = servers
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.headers["Content-Type"], response.read()


def test_the_server_serves_the_page_and_the_api(server, store):
    url = server

    content_type, page = _get(url + "/")
    assert content_type.startswith("text/html") and b"Run lineage" in page
    _, body = _get(f"{url}/api/runs")
    # A run served on its own is named after its directory: browsers drop
    # "." from URLs.
    (name,) = [run["name"] for run in json.loads(body)]
    assert name == store.path.name
    run = f"{url}/api/run/{name}"
    _, body = _get(f"{run}/item/{O1.hex()}")
    assert json.loads(body)["stage"] == "2:map(g)"
    _, body = _get(f"{run}/element?namespace={A.namespace.hex()}&position=1")
    assert json.loads(body)["kind"] == "input"
    for missing in [f"{run}/item/{'00' * 16}", f"{url}/api/run/other/summary"]:
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(missing)
        assert error.value.code == 404


# -- The graph of the pipeline -----------------------------------------------------
def _f(x):
    return x


def _child(stream):
    return stream.map(_f).chunk(2)


@pytest.fixture
def graph_store(tmp_path):
    """The run of source -> map -> group_by(map -> chunk) -> map -> sink, with
    a chain of items through every stage that records them."""
    sink = Source("x").map(_f).group_by(0, _child, workers=None).unkey().map(_f)
    report = report_of(lower([sink.sink(print)]))
    store = FileRunStore(tmp_path)
    store.write_report(report.to_json())
    chain = [
        ("0:source(x)", "source"),
        ("1:map(_f)", "map"),
        ("2:group_by[a]/1:map(_f)", "map"),
        ("2:group_by[a]/2:chunk(2)", "accumulate"),
        ("3:map(_f)", "map"),
    ]
    ids = [bytes([k + 1]) * 16 for k in range(len(chain))]
    store.append_lineage(
        [
            LineageRecord(
                lid,
                stage,
                kind,
                () if k == 0 else (ids[k - 1],),
                (ElementSegment("", "", 2, A.namespace, 0, 2),),
                (A,) if k == 0 else (),
            )
            for k, (lid, (stage, kind)) in enumerate(zip(ids, chain))
        ],
    )
    return store, report, ids


def test_the_graph_has_the_items_of_every_node(graph_store):
    store, report, _ = graph_store

    graph = LineageView(store).ops_graph()

    assert graph["svg"].lstrip().startswith("<svg")
    nodes = {n["label"]: n for n in graph["nodes"]}
    child_map = nodes["2:group_by/1:map(_f)"]
    assert child_map["stages"] == ["2:group_by[a]/1:map(_f)"]
    assert graph["counts"][child_map["svg_id"]] == 1
    assert nodes["2:group_by"]["svg_id"] not in graph["counts"]


def test_the_ancestry_of_an_output_follows_the_edges_through_the_group_by(
    graph_store,
):
    store, report, ids = graph_store
    edges = {(e.source, e.target): f"e{i}" for i, e in enumerate(report.edges)}

    overlay = LineageView(store).overlay("ancestry", ids[-1])

    assert overlay["items"] == 5 and overlay["unplaced"] == []
    lit = set(overlay["edges"])
    assert {
        edges[("0:source(x)", "1:map(_f)")],
        edges[("1:map(_f)", "2:group_by")],
        edges[("2:group_by/0:source(group_by.input)", "2:group_by/1:map(_f)")],
        edges[("2:group_by/1:map(_f)", "2:group_by/2:chunk(2)")],
    } < lit
    # Into the template from the router, and out of it into the next stage.
    assert sum(svg_id.startswith("v") for svg_id in lit) == 2
    # Through the router and the source of the child, which record nothing.
    nodes = {n.id: f"s{i}" for i, n in enumerate(report.nodes)}
    assert set(overlay["through"]) == {
        nodes["2:group_by"],
        nodes["2:group_by/0:source(group_by.input)"],
    }
    (into_sink,) = [svg_id for (_, target), svg_id in edges.items() if "sink" in target]
    assert into_sink not in lit


def test_descendants_and_holders_on_the_graph(graph_store):
    store, _, ids = graph_store
    view = LineageView(store)

    descendants = view.overlay("descendants", ids[2])
    holders = view.overlay("element", A.namespace, 1)

    assert descendants["items"] == 3
    assert holders["items"] == 5 and len(holders["counts"]) == 5
    with pytest.raises(ValueError):
        view.overlay("sideways", ids[0])


def test_a_run_without_a_report_has_no_graph(store):
    view = LineageView(store)

    assert view.ops_graph() == {"svg": None}
    with pytest.raises(ValueError, match="no graph"):
        view.overlay("ancestry", O1)


# -- Directories of runs -----------------------------------------------------------
def _run(path, run_id, protocols=("P",)):
    store = FileRunStore(path)
    store.write_manifest(
        {
            "run_id": run_id,
            "created": "2026-11-17T12:00:00",
            "protocols": [{"protocol": p, "fingerprint": "f"} for p in protocols],
        },
    )
    store.append_lineage(
        [LineageRecord(bytes([1]) * 16, "0:source(x)", "source", (), ())],
    )
    return store


def test_runs_are_found_below_a_directory_but_not_inside_runs(tmp_path):
    _run(tmp_path / "a", "ra")
    _run(tmp_path / "group" / "b", "rb")
    _run(tmp_path / "a" / "nested", "rn")
    (tmp_path / ".hidden" / "lineage").mkdir(parents=True)
    (tmp_path / "empty").mkdir()

    assert list(lineage_view.find_runs(tmp_path)) == ["a", "group/b"]
    assert list(lineage_view.find_runs(tmp_path / "a")) == ["a"]


def test_the_list_of_runs_reads_their_manifests(tmp_path):
    _run(tmp_path / "a", "ra", protocols=("P", "Q"))

    (run,) = lineage_view.Runs(tmp_path).list()

    assert (run["name"], run["run_id"], run["protocols"], run["parts"]) == (
        "a",
        "ra",
        ["P", "Q"],
        1,
    )
    assert run["updated"] is not None


def test_only_the_runs_used_last_stay_loaded(tmp_path):
    for name in "abc":
        _run(tmp_path / name, name)
    runs = lineage_view.Runs(tmp_path, keep=2)

    first = runs.view("a")
    assert runs.view("a") is first
    runs.view("b")
    runs.view("c")

    assert runs.view("a") is not first
    with pytest.raises(KeyError):
        runs.view("d")
