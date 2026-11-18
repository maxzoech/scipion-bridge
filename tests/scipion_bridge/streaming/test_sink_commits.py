"""Commits of sinks and what makes a sink idempotent (in-process).

The Ray tests (a rerun skips what was committed) are in test_ray_lineage.py.
"""

import asyncio

import numpy as np
import pytest
import tensorstore as ts

import scipion_bridge as B
from scipion_bridge.backend.ray import ray_protocol_runner
from scipion_bridge.backend.ray.sinks import TensorStoreSinkWriter
from scipion_bridge.core.streaming import (
    Commit,
    CommittingSinkWriter,
    ElementSegment,
    FileRunStore,
    LineageGraph,
    LineageRecord,
    RunStoreMismatchError,
    StageInfo,
    StoreSinkWriter,
)
from scipion_bridge.core.streaming.lineage import SeenElements

F = b"F" * 16
G = b"G" * 16
A, B_ID, C = (bytes([k]) * 16 for k in range(1, 4))


class Value(B.Struct):
    value: float


def _values(start, n):
    s = B.Set[Value](capacity=n)
    s["value"] = np.arange(start, start + n, dtype=np.float64)
    return s


# -- The commit log ----------------------------------------------------------------
def test_commits_round_trip_in_their_order_per_sink(tmp_path):
    store = FileRunStore(tmp_path)

    store.append_commits([Commit(A, "3:sink", 2, {"rows": {"x": 2}})])
    store.append_commits([Commit(B_ID, "4:sink", 1), Commit(C, "3:sink", 5)])

    assert store.read_commits("3:sink") == [
        Commit(A, "3:sink", 2, {"rows": {"x": 2}}),
        Commit(C, "3:sink", 5),
    ]
    assert len(store.read_commits()) == 3
    assert FileRunStore(tmp_path / "new").read_commits() == []


def test_stages_that_changed_are_refused():
    def store_with(stages):
        store = FileRunStore("unused")
        store.read_stages = lambda: {s.label: s for s in stages}  # type: ignore
        return store

    store = store_with([StageInfo("1:map(f)", "IRMap", "fp1")])

    store.check_stages(
        [StageInfo("1:map(f)", "IRMap", "fp1"), StageInfo("2:sink", "IRSink", None)],
    )
    with pytest.raises(RunStoreMismatchError, match="1:map"):
        store.check_stages([StageInfo("1:map(f)", "IRMap", "fp2")])


def test_a_run_continues_only_a_run_of_the_same_protocols_and_inputs():
    manifest = {
        "run_id": "a",
        "protocols": [{"protocol": "P", "fingerprint": "f"}],
        "parameters": {"scale": 3},
        "inputs": {"x": {"value": ("a.star",), "count": 1}},
        "chunking": {"chunk_size": 4},
    }
    stored = {
        **manifest,
        "run_id": "b",
        "inputs": {"x": {"value": ["a.star"], "count": 1}},
    }

    ray_protocol_runner._check_continues({}, manifest)
    ray_protocol_runner._check_continues(stored, manifest)
    with pytest.raises(RunStoreMismatchError, match="parameters"):
        ray_protocol_runner._check_continues(
            {**stored, "parameters": {"scale": 4}},
            manifest,
        )


def test_a_graph_keeps_one_record_of_items_recorded_twice(tmp_path):
    store = FileRunStore(tmp_path)
    records = [
        LineageRecord(
            A, "0:source(x)", "source", (), (ElementSegment("", "", 2, F, 0, 2),)
        ),
        LineageRecord(B_ID, "1:map(f)", "map", (A,), (), inherited=True),
    ]
    # A run continuing the first one records its items again.
    store.append_lineage(records)
    store.append_lineage(records)

    graph = LineageGraph.of(store)

    assert graph.item_ids() == [A, B_ID]
    assert graph.children(A) == (B_ID,)
    assert len(graph.holders(F, 1)) == 2


# -- Elements written twice --------------------------------------------------------
def test_elements_seen_before_are_counted_per_field():
    seen = SeenElements()
    rows = ElementSegment("", "particles", 3, F, 0, 3)
    again = ElementSegment(
        "['classes'][1]",
        "particles",
        2,
        F,
        positions=np.array([2, 7], "<i8").tobytes(),
    )
    mixed = ElementSegment(
        "",
        "particles",
        2,
        positions=np.array([1, 1], "<i8").tobytes(),
        namespaces=F + G,
        indices=np.array([0, 1], "<i4").tobytes(),
    )

    assert seen.add((rows,)) == 0
    # Row 2 of F again; the same rows in another field are other elements.
    assert seen.add((again, ElementSegment("", "", 3, F, 0, 3))) == 1
    # Row 1 of F again, row 1 of G is new.
    assert seen.add((mixed,)) == 1
    assert seen.add((ElementSegment("", "particles", 4),)) == 0


# -- Writers -----------------------------------------------------------------------
def test_the_store_sink_writes_an_item_once_however_often_it_is_written(tmp_path):
    writer = StoreSinkWriter(tmp_path / "outputs")
    assert isinstance(writer, CommittingSinkWriter)

    async def run():
        await writer.recover(None)
        await writer.write_identified(_values(0, 2), A)
        await writer.write_identified(_values(5, 3), A)
        await writer.finalize()

    asyncio.run(run())

    assert [p.name for p in (tmp_path / "outputs").iterdir()] == [f"{A.hex()}.pickle"]
    assert list(writer.read(A)["value"]) == [5, 6, 7]


class Particle(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(4, 4))


class Summary(B.Struct):
    mean: B.Array[np.float32] = B.Array(shape=(4, 4))


def _particles(n, value):
    s = B.Set[Particle](capacity=n)
    s["pixels"] = np.full((n, 4, 4), value, dtype=np.float32)
    return s


async def _array(path):
    store = await ts.open(
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}},
    )
    return await store.read()


def test_tensorstore_rows_after_the_last_commit_are_cut_back(tmp_path):
    path = tmp_path / "out.zarr"

    async def crashed_run():
        writer = TensorStoreSinkWriter(path)
        await writer.recover(None)
        receipt = await writer.write_identified(_particles(2, 1.0), A)
        await writer.write_identified(Summary(mean=np.ones((4, 4), np.float32)), B_ID)
        # Written, but the run crashed before its commit.
        await writer.write_identified(_particles(3, 2.0), C)
        return receipt

    async def continued_run(receipt):
        writer = TensorStoreSinkWriter(path)
        await writer.recover(receipt)
        await writer.write_identified(_particles(3, 2.0), C)
        await writer.finalize()

    receipt = asyncio.run(crashed_run())
    assert receipt == {"rows": {"pixels": 2}, "overwritten": []}
    asyncio.run(continued_run(receipt))

    pixels = asyncio.run(_array(path / "pixels"))
    assert pixels.shape == (5, 4, 4)
    assert list(pixels[:, 0, 0]) == [1, 1, 2, 2, 2]
    # Created after the commit: cut back to no rows.
    assert asyncio.run(_array(path / "mean")).shape == (0, 4)


def test_tensorstore_keeps_overwritten_arrays_that_were_committed(tmp_path):
    path = tmp_path / "out.zarr"

    async def run():
        writer = TensorStoreSinkWriter(path)
        await writer.recover(None)
        return await writer.write_identified(
            Summary(mean=np.full((4, 4), 3, np.float32)),
            A,
        )

    receipt = asyncio.run(run())
    asyncio.run(TensorStoreSinkWriter(path).recover(receipt))

    assert receipt == {"rows": {}, "overwritten": ["mean"]}
    assert asyncio.run(_array(path / "mean"))[0, 0] == 3
