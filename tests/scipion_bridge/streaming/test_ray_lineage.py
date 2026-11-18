"""Item ids and element ids on Ray (see core/streaming/lineage.py).

The policies are tested in-process in test_lineage.py; these tests check
what only exists across actors: ids carried between stages, through map
calls, spills and the children of a group_by, and reproduced by a second
run.
"""

import time

import numpy as np
import pytest
import ray
import tensorstore as ts

import scipion_bridge as B
from scipion_bridge.backend.ray import RayBackend, RayCompiledPipeline
from scipion_bridge.backend.ray.sinks import TensorStoreSinkWriter
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    FileRunStore,
    LineageGraph,
    Pipeline,
    Source,
    StoreSinkWriter,
    UniqueElements,
)
from scipion_bridge.core.struct import identity, references
from scipion_bridge.core.struct.id_column import IdColumn
from scipion_bridge.core.struct.source import ArraySource, FileId, Reference

pytestmark = pytest.mark.usefixtures("ray_cluster")

F = b"F" * 16


class Value(B.Struct):
    value: float


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def get(self):
        return self.items


def _collecting(collector, delay_s=0.0):
    def write(item):
        time.sleep(delay_s)
        ray.get(collector.append.remote(item))

    return CallbackSinkWriter(write)


def _values(start, n):
    """Values ``start..start+n-1`` with the element ids ``(F, row)``."""
    s = B.Set[Value](capacity=n)
    s["value"] = np.arange(start, start + n, dtype=np.float64)
    identity.set_ids(s, IdColumn.of(F, range(start, start + n)))
    return s


def _scaled(values):
    """New rows: a Set the map creates."""
    out = B.Set[Value](capacity=len(values))
    out["value"] = np.asarray(values["value"]) * 2
    return out


def _reversed(values):
    """The same rows, as a view."""
    return values[np.arange(len(values))[::-1]]


def _second(item):
    return item[1]


def _run(sink_node, inputs, **backend_kwargs):
    """Run a pipeline on ``inputs``: the ids its sink received, its stats."""
    pipeline = Pipeline.from_sink(
        sink_node,
        backend=RayBackend(init_ray=False, **backend_kwargs),
    )
    try:
        for value in inputs:
            pipeline.send(x=value)
        pipeline.flush()
        compiled = pipeline.compiled
        assert isinstance(compiled, RayCompiledPipeline)
        (received,) = compiled.received_ids().values()
        return received, pipeline.stats()
    finally:
        pipeline.close()


def _element_ids(items):
    columns = [identity.element_ids(item) for item in items]
    assert all(column is not None for column in columns)
    return [element for column in columns if column for element in column.elements()]


def test_two_runs_give_the_same_ids_and_created_rows_get_ids():
    inputs = [_values(3 * k, 3) for k in range(5)]
    collectors = [Collector.remote(), Collector.remote()]

    runs = [
        _run(
            Source("x").map(_scaled).chunk(4).write_to(_collecting(collector)),
            inputs,
        )[0]
        for collector in collectors
    ]

    first, second = runs
    assert first == second
    assert len(first) == 4 == len(set(first))
    outputs = [ray.get(collector.get.remote()) for collector in collectors]
    elements = [_element_ids(items) for items in outputs]
    assert elements[0] == elements[1]
    assert len(set(elements[0])) == 15
    assert all(element is not None and element[0] != F for element in elements[0])


def test_rows_that_move_keep_their_element_ids():
    collector = Collector.remote()

    _run(
        Source("x").map(_reversed).write_to(_collecting(collector)),
        [_values(0, 3), _values(3, 2)],
    )

    assert _element_ids(ray.get(collector.get.remote())) == [
        (F, row) for row in [2, 1, 0, 4, 3]
    ]


def test_ids_survive_spilled_items():
    inputs = [_values(2 * k, 2) for k in range(8)]
    node = Source("x").map(_scaled)

    spilled, stats = _run(
        node.write_to(_collecting(Collector.remote(), delay_s=0.05)),
        inputs,
        spill_threshold=1,
    )
    in_memory, _ = _run(
        Source("x").map(_scaled).write_to(_collecting(Collector.remote())),
        inputs,
    )

    assert sum(stage.spilled for stage in stats.values()) > 0
    assert spilled == in_memory


def _child(stream):
    return stream.map(_second).chunk(2).map(_scaled)


@pytest.mark.parametrize("workers", [None, 2], ids=["per-key", "shared"])
def test_ids_of_group_by_children_are_unique_and_reproduced(workers):
    inputs = [(k % 3, _values(2 * k, 2)) for k in range(9)]

    runs = [
        _run(
            Source("x")
            .group_by(0, _child, workers=workers)
            .unkey()
            .write_to(_collecting(Collector.remote())),
            inputs,
        )[0]
        for _ in range(2)
    ]

    first, second = runs
    # Children return their results in no particular order across keys.
    assert sorted(first) == sorted(second)
    assert len(first) == 9 == len(set(first))


# -- Lineage records in a run store ------------------------------------------------
@pytest.fixture
def stack(tmp_path):
    """An input file, whose rows the inputs of the tests reference."""
    path = tmp_path / "stack.mrcs"
    path.write_bytes(b"rows")
    return FileId.of(path)


def _stack_values(stack, start, n):
    """Rows ``start..`` of ``stack``: element ids and a reference to its rows."""
    s = B.Set[Value](capacity=n)
    source = ArraySource(np.arange(64, dtype=np.float64)[:, None], files=(stack,))
    references.set_reference(s, ("value",), Reference(source, range(start, start + n)))
    identity.set_ids(s, IdColumn.of(stack.namespace, range(start, start + n)))
    return s


def test_the_lineage_of_an_output_reaches_its_input_rows(tmp_path, stack):
    store = FileRunStore(tmp_path / "run")
    collector = Collector.remote()

    received, _ = _run(
        Source("x").map(_scaled).chunk(4).write_to(_collecting(collector)),
        [_stack_values(stack, 3 * k, 3) for k in range(4)],
        run_store=store,
    )

    graph = LineageGraph.of(store)
    ancestry = graph.ancestry(received[0])
    assert ancestry.files == (stack,)
    assert ancestry.missing == ()
    assert [stage.label.split(":")[1] for stage in ancestry.stages] == [
        "chunk(4)",
        "map(_scaled)",
        "source(x)",
    ]
    # The rows the map created trace back to the rows of the stack.
    output, *_ = ray.get(collector.get.remote())
    namespace, position = _element_ids([output])[0]
    origin = graph.trace_element(namespace, position)
    assert origin.created_by is not None and origin.created_by.kind == "map"
    assert origin.ancestry is not None and origin.ancestry.files == (stack,)
    (source,) = origin.ancestry.sources
    assert [s.position_array().tolist() for s in source.elements] == [[0, 1, 2]]
    assert graph.trace_element(stack.namespace, 5).row == 5


def test_the_children_of_a_group_by_record_their_stages_and_items(tmp_path, stack):
    store = FileRunStore(tmp_path / "run")

    received, _ = _run(
        Source("x")
        .group_by(0, _child, workers=None)
        .unkey()
        .write_to(_collecting(Collector.remote())),
        [(k % 2, _stack_values(stack, 2 * k, 2)) for k in range(4)],
        run_store=store,
    )

    graph = LineageGraph.of(store)
    for lid in received:
        ancestry = graph.ancestry(lid)
        assert ancestry.files == (stack,) and ancestry.missing == ()
        assert any("group_by[" in stage.label for stage in ancestry.stages)
        assert all(stage.kind for stage in ancestry.stages)


@ray.remote
class Gate:
    """Holds the writes of a sink until it is released."""

    def __init__(self):
        self.released = False

    def is_released(self):
        return self.released

    def release(self):
        self.released = True


def _gated(gate, collector):
    def write(item):
        while not ray.get(gate.is_released.remote()):
            time.sleep(0.05)
        ray.get(collector.append.remote(item))

    return CallbackSinkWriter(write)


def test_lineage_is_written_while_the_pipeline_runs(tmp_path, stack):
    store = FileRunStore(tmp_path / "run")
    gate = Gate.remote()
    pipeline = Pipeline.from_sink(
        Source("x").map(_scaled).write_to(_gated(gate, Collector.remote())),
        backend=RayBackend(init_ray=False, run_store=store, lineage_batch=2),
    )
    try:
        for k in range(5):
            pipeline.send(x=_stack_values(stack, 2 * k, 2))
        # The sink holds the items: nothing has been flushed yet.
        deadline = time.monotonic() + 30
        while store.read_lineage_table().num_rows < 4 and time.monotonic() < deadline:
            time.sleep(0.2)
        assert store.read_lineage_table().num_rows >= 4

        ray.get(gate.release.remote())
        pipeline.flush()

        # Every record is written once FLUSH has passed: 5 sources, 5 maps.
        assert store.read_lineage_table().num_rows == 10
        stats = pipeline.stats()
        assert stats["0:source(x)"].lineage_s > 0
        assert stats["1:map(_scaled)"].lineage_s > 0
    finally:
        pipeline.close()


# -- Commits of sinks --------------------------------------------------------------
def _sink_stats(stats):
    (sink,) = [s for label, s in stats.items() if "sink" in label]
    return sink


def test_a_rerun_in_the_same_run_store_writes_nothing_again(tmp_path):
    store = FileRunStore(tmp_path / "run")
    inputs = [_values(3 * k, 3) for k in range(4)]

    def run():
        writer = StoreSinkWriter(tmp_path / "outputs")
        return _run(
            Source("x").map(_scaled).write_to(writer),
            inputs,
            run_store=store,
        )

    first, first_stats = run()
    second, second_stats = run()

    assert len(first) == 4 and second == []
    assert _sink_stats(second_stats).skipped == 4
    assert sorted(c.item_id for c in store.read_commits()) == sorted(first)
    assert {c.elements for c in store.read_commits()} == {3}
    assert len(list((tmp_path / "outputs").iterdir())) == 4


def _fail_if(marker):
    def fail(values):
        if marker.exists() and np.asarray(values["value"]).min() >= 9:
            raise RuntimeError("crash")
        return values

    return fail


def test_a_run_continuing_a_failed_one_writes_every_row_once(tmp_path):
    store = FileRunStore(tmp_path / "run")
    marker = tmp_path / "fail"
    marker.touch()
    inputs = [_values(3 * k, 3) for k in range(5)]

    def run():
        sink = (
            Source("x")
            .map(_fail_if(marker))
            .write_to(TensorStoreSinkWriter(tmp_path / "out.zarr"))
        )
        return _run(sink, inputs, run_store=store, lineage_batch=1)

    with pytest.raises(Exception, match="crash"):
        run()
    marker.unlink()
    written, stats = run()

    values = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(tmp_path / "out.zarr/value")},
        },
    ).result()
    assert sorted(np.asarray(values.read().result()).ravel()) == list(range(15))
    assert len(written) + _sink_stats(stats).skipped == 5
    assert len({c.item_id for c in store.read_commits()}) == 5


def test_a_sink_can_refuse_elements_written_twice():
    twice = [_values(0, 3), _values(2, 3)]

    with pytest.raises(Exception, match="1 elements a second time"):
        _run(
            Source("x").write_to(
                _collecting(Collector.remote()),
                unique_elements=UniqueElements.RAISE,
            ),
            twice,
        )
