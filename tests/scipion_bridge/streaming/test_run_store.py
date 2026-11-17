"""The run store and the lineage graph of a run (in-process)."""

from pathlib import Path

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming import (
    ElementSegment,
    FileRunStore,
    LineageGraph,
    LineageRecord,
    StageInfo,
)
from scipion_bridge.core.streaming.lineage import element_segments, input_files
from scipion_bridge.core.struct import concat, identity, references
from scipion_bridge.core.struct.id_column import IdColumn
from scipion_bridge.core.struct.source import ArraySource, FileId, Reference
from scipion_bridge.core.utils.hashing import item_id

F = b"F" * 16
G = b"G" * 16
S1, S2, M1, A1 = (bytes([k]) * 16 for k in range(1, 5))


class Value(B.Struct):
    value: float


def _values(n, column=None):
    s = B.Set[Value](capacity=n)
    s["value"] = np.arange(n, dtype=np.float64)
    if column is not None:
        identity.set_ids(s, column)
    return s


def _file(name):
    return FileId(Path(f"/data/{name}.mrcs"), 100, 7)


# -- Element segments ------------------------------------------------------------
def test_element_segments_keep_every_kind_of_id():
    explicit = np.arange(32, dtype=np.uint8).reshape(2, 16)
    s = concat(
        [
            _values(40, IdColumn.of(F, range(4, 44))),
            _values(40, IdColumn.of(G, np.arange(40)[::-1].copy())),
            _values(2, IdColumn.explicit_ids(explicit)),
            _values(3, IdColumn.of(F, range(3))),
            _values(2, IdColumn.of(G, range(2))),
            _values(1),
        ],
    )

    ranged, positions, hashed, mixed, unassigned = element_segments(s)

    assert (ranged.count, ranged.namespace, ranged.start, ranged.stop) == (40, F, 4, 44)
    assert positions.position_array().tolist() == list(range(40))[::-1]
    assert hashed.explicit == explicit.tobytes()
    assert mixed.count == 5 and mixed.namespaces is not None
    assert unassigned.count == 1 and unassigned.namespace is None
    assert ranged.holds(F, 43) and not ranged.holds(F, 44)
    assert positions.holds(G, 2) and not positions.holds(F, 2)
    assert mixed.holds(F, 2) and mixed.holds(G, 1) and not mixed.holds(G, 2)
    assert not mixed.holds(b"X" * 16, 0)


def test_input_files_are_the_files_the_rows_reference():
    s = _values(4)
    source = ArraySource(np.zeros((4, 1)), files=(_file("a"),))
    references.set_reference(s, ("value",), Reference(source, range(4)))

    assert input_files((s, _values(2))) == (_file("a"),)


# -- The store -------------------------------------------------------------------
def _records():
    return [
        LineageRecord(
            S1,
            "0:source(x)",
            "source",
            (),
            (ElementSegment("", "", 4, F, 0, 4),),
            (_file("a"),),
        ),
        LineageRecord(
            S2,
            "0:source(x)",
            "source",
            (),
            (ElementSegment("", "", 2, G, 0, 2),),
            (_file("b"),),
        ),
        LineageRecord(
            M1,
            "1:map(f)",
            "map",
            (S1,),
            (ElementSegment("", "", 4, item_id(M1, b""), 0, 4),),
        ),
        LineageRecord(
            A1,
            "2:chunk(6)",
            "accumulate",
            (M1, S2),
            (
                ElementSegment("", "", 4, item_id(M1, b""), 0, 4),
                ElementSegment("", "", 2, G, 0, 2),
            ),
        ),
    ]


def test_records_round_trip_through_the_store(tmp_path):
    store = FileRunStore(tmp_path)
    records = _records()

    store.append_lineage(records[:2])
    store.append_lineage(records[2:])
    store.append_lineage([])

    assert store.read_lineage() == records
    assert len(list((tmp_path / "lineage").glob("*.parquet"))) == 2
    assert list(tmp_path.rglob("*.tmp")) == []


def test_positions_and_explicit_ids_round_trip(tmp_path):
    store = FileRunStore(tmp_path)
    explicit = np.arange(32, dtype=np.uint8).reshape(2, 16)
    s = concat(
        [
            _values(40, IdColumn.of(G, np.arange(40)[::-1].copy())),
            _values(2, IdColumn.explicit_ids(explicit)),
            _values(3, IdColumn.of(F, range(3))),
            _values(2, IdColumn.of(G, range(2))),
            _values(1),
        ],
    )
    record = LineageRecord(S1, "s", "source", (), element_segments(s), inherited=True)

    store.append_lineage([record])

    assert store.read_lineage() == [record]


def test_manifest_and_stages_round_trip(tmp_path):
    store = FileRunStore(tmp_path)

    store.write_manifest({"run_id": "r", "parameters": {"path": Path("/x")}})
    store.write_stages("p1", [StageInfo("0:source(x)", "IRSource", None)])
    store.write_stages("p2", [StageInfo("1:group_by[3]/0:map(f)", "IRMap", "fp")])

    assert store.read_manifest() == {"run_id": "r", "parameters": {"path": "/x"}}
    assert store.read_stages()["1:group_by[3]/0:map(f)"].stage_fp == "fp"
    assert len(store.read_stages()) == 2


def test_an_empty_store_has_nothing(tmp_path):
    store = FileRunStore(tmp_path / "run")

    assert store.read_manifest() == {}
    assert store.read_stages() == {}
    assert store.read_lineage() == []


# -- Queries ----------------------------------------------------------------------
def _graph():
    stages = {
        "1:map(f)": StageInfo("1:map(f)", "IRMap", "fp-f"),
        "2:chunk(6)": StageInfo("2:chunk(6)", "IRAccumulate", None),
    }
    return LineageGraph.from_records(_records(), stages)


def test_the_ancestry_of_an_item_reaches_its_inputs():
    ancestry = _graph().ancestry(A1)

    # Nearest first: the item, its parents, their parents.
    assert ancestry.item_ids[0] == A1
    assert set(ancestry.item_ids[1:3]) == {M1, S2} and ancestry.item_ids[3] == S1
    assert {r.item_id for r in ancestry.sources} == {S1, S2}
    assert set(ancestry.files) == {_file("a"), _file("b")}
    assert {s.label for s in ancestry.stages} == {
        "2:chunk(6)",
        "1:map(f)",
        "0:source(x)",
    }
    assert {s.stage_fp for s in ancestry.stages} == {None, "fp-f"}
    assert ancestry.missing == ()


def test_parents_without_records_are_reported():
    records = [r for r in _records() if r.item_id != S2]

    ancestry = LineageGraph.from_records(records, {}).ancestry(A1)

    assert ancestry.missing == (S2,)


def test_an_element_of_an_input_file_is_its_row():
    origin = _graph().trace_element(_file("b").namespace, 1)

    assert (origin.file, origin.row) == (_file("b"), 1)
    assert origin.created_by is None


def test_an_element_a_stage_created_is_traced_from_its_item():
    origin = _graph().trace_element(item_id(M1, b""), 2)

    assert origin.created_by is not None and origin.created_by.item_id == M1
    assert origin.ancestry is not None
    assert origin.ancestry.files == (_file("a"),)


def test_an_unknown_element_raises():
    with pytest.raises(KeyError):
        _graph().trace_element(b"X" * 16, 0)


V1, O1 = (bytes([k]) * 16 for k in range(5, 7))


def _graph_with_outputs():
    """Two input stacks through a map, a chunk, a map keeping the elements
    (V1), and an output O1 holding members of both stacks (a mixed segment)."""
    a, b = _file("a"), _file("b")
    created = item_id(M1, b"")
    members = concat(
        [
            _values(2, IdColumn.of(a.namespace, range(1, 3))),
            _values(1, IdColumn.of(b.namespace, range(1))),
        ],
    )
    return LineageGraph.from_records(
        [
            LineageRecord(
                S1,
                "0:source(x)",
                "source",
                (),
                (ElementSegment("", "", 4, a.namespace, 0, 4),),
                (a,),
            ),
            LineageRecord(
                S2,
                "0:source(x)",
                "source",
                (),
                (ElementSegment("", "", 2, b.namespace, 0, 2),),
                (b,),
            ),
            LineageRecord(
                M1,
                "1:map(f)",
                "map",
                (S1,),
                (ElementSegment("", "", 4, created, 0, 4),),
            ),
            LineageRecord(
                A1,
                "2:chunk(6)",
                "accumulate",
                (M1, S2),
                (
                    ElementSegment("", "", 4, created, 0, 4),
                    ElementSegment("", "", 2, b.namespace, 0, 2),
                ),
            ),
            LineageRecord(V1, "3:map(g)", "map", (A1,), (), inherited=True),
            LineageRecord(O1, "4:map(h)", "map", (V1,), element_segments(members)),
        ],
        {},
    )


def test_holders_of_an_element_include_the_items_inheriting_it():
    graph = _graph_with_outputs()

    assert set(graph.holders(_file("a").namespace, 2)) == {S1, O1}
    assert set(graph.holders(_file("b").namespace, 0)) == {S2, A1, V1, O1}
    assert graph.holders(_file("b").namespace, 3) == []
    assert graph.holders(b"X" * 16, 0) == []


def test_inherited_records_hold_the_elements_of_their_parent():
    graph = _graph_with_outputs()

    assert graph.record(V1).elements == ()
    assert graph.elements(V1) == graph.record(A1).elements


def test_leaves_children_and_stage_summaries():
    graph = _graph_with_outputs()

    assert graph.leaves() == [O1]
    assert set(graph.children(S2)) == {A1}
    assert graph.items_of("3:map(g)") == [V1]
    summaries = {s.stage.label: s for s in graph.stage_summaries()}
    assert [s.stage.label for s in graph.stage_summaries()] == [
        "0:source(x)",
        "1:map(f)",
        "2:chunk(6)",
        "3:map(g)",
        "4:map(h)",
    ]
    assert (summaries["0:source(x)"].items, summaries["0:source(x)"].elements) == (
        2,
        6,
    )
    assert summaries["3:map(g)"].elements == summaries["2:chunk(6)"].elements == 6
    assert summaries["2:chunk(6)"].input_elements == 2
    assert summaries["4:map(h)"].input_elements == 3
    assert summaries["3:map(g)"].inherited == 1


def test_a_graph_read_from_a_store_answers_like_one_from_records(tmp_path):
    store = FileRunStore(tmp_path)
    graph = _graph_with_outputs()
    store.append_lineage([graph.record(lid) for lid in graph.item_ids()])

    read = LineageGraph.of(store)

    assert read.item_ids() == graph.item_ids()
    assert set(read.holders(_file("b").namespace, 0)) == {S2, A1, V1, O1}
    assert set(read.ancestry(O1).files) == {_file("a"), _file("b")}


# Documented in docs/source/user_guide/known_issues.rst.
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Namespaces other than input files (Scipion item ids) are not "
        "described in lineage records; when this passes, the gap is closed."
    ),
)
def test_an_element_with_a_scipion_item_id_is_an_input_element():
    scipion = item_id(b"scipion", b"/project/particles.sqlite", b"")
    source = LineageRecord(
        S1,
        "0:source(x)",
        "source",
        (),
        (
            ElementSegment(
                "", "", 3, scipion, None, None, np.array([7, 9, 12], "<i8").tobytes()
            ),
        ),
    )

    origin = LineageGraph.from_records([source], {}).trace_element(scipion, 9)

    assert origin.created_by is None
    assert origin.row == 9
