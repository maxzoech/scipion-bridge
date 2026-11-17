"""The lineage viewer: its pages, and its server (in-process)."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming import (
    ElementSegment,
    FileRunStore,
    LineageRecord,
    Source,
    StageInfo,
    lower,
)
from scipion_bridge.core.streaming.lineage import element_segments
from scipion_bridge.core.streaming.report import lineage_view, report_of
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
    return store


def test_the_summary_shows_the_run_and_its_stages(store):
    summary = LineageView(store).summary()

    assert summary["records"] == 4 and summary["outputs"] == 1
    assert summary["manifest"]["inputs"]["particles"]["count"] == 2
    stages = {s["label"]: s for s in summary["stages"]}
    assert list(stages) == ["0:source(x)", "1:map(f)", "2:map(g)"]
    assert (stages["2:map(g)"]["elements"], stages["2:map(g)"]["input_elements"]) == (
        5,
        3,
    )
    assert stages["1:map(f)"]["stage_fp"] == "fp1"


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


def test_the_server_serves_the_page_and_the_api(server):
    url = server

    content_type, page = _get(url + "/")
    assert content_type.startswith("text/html") and b"Run lineage" in page
    _, body = _get(f"{url}/api/runs")
    assert [run["name"] for run in json.loads(body)] == ["."]
    run = f"{url}/api/run/."
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
    assert list(lineage_view.find_runs(tmp_path / "a")) == ["."]


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
