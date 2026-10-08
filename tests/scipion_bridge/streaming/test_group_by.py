"""group_by / unkey: routing items into a pipeline per key."""

import time

import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ir import (
    IRAccumulate,
    IRDemux,
    IRMap,
    IRSource,
    Keyed,
)
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import KeyedOp, Source, _make_key_extractor
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


def _ids(s):
    return [int(x) for x in s["id"].flatten()]


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, x):
        self.items.append(x)

    def get(self):
        return self.items


def _collecting_writer(collector):
    return CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))


def _second(item):
    return item[1]


def _chunk_ids(stream):
    """Per-key pipeline: chunk the Sets of a key into pairs."""
    return stream.map(_second).chunk(2).map(_ids)


def _compile(node, **backend_kwargs):
    (sink,) = lower([node])
    return RayBackend(init_ray=False, **backend_kwargs).compile([sink])


def _by_key(results):
    grouped = {}
    for result in results:
        assert isinstance(result, Keyed)
        grouped.setdefault(result.key, []).append(result.value)
    return grouped


def _child_labels(pipeline):
    return [label for label in pipeline.stats() if label.startswith("group_by[")]


# -- Lowering ----------------------------------------------------------------


def test_group_by_lowers_to_demux_with_template():
    sink_node = Source("x").group_by(0, _chunk_ids).unkey().sink(print)

    (sink,) = lower([sink_node])

    (demux,) = sink.upstream
    assert isinstance(demux, IRDemux)
    assert [type(up) for up in demux.upstream] == [IRSource]
    template = demux.template
    assert isinstance(template, IRMap)
    (chunk,) = template.upstream
    assert isinstance(chunk, IRAccumulate)
    (select,) = chunk.upstream
    (entry,) = select.upstream
    assert isinstance(entry, IRSource)
    assert entry.name == demux.source_name
    assert entry.downstream == [select]


def test_unkey_adds_no_node():
    keyed = Source("x").group_by(0, _chunk_ids)

    assert isinstance(keyed, KeyedOp)
    assert keyed.unkey() is keyed


def test_template_is_not_part_of_the_enclosing_graph():
    sink_node = Source("x").group_by(0, _chunk_ids).unkey().sink(print)

    (sink,) = lower([sink_node])

    (demux,) = sink.upstream
    (source,) = demux.upstream
    assert isinstance(source, IRSource)
    assert source.name == "x"
    assert source.downstream == [demux]


def test_pipeline_consuming_another_source_raises():
    keyed = Source("x").group_by(
        0,
        lambda stream: stream.combine_latest(Source("y")),
    )

    with pytest.raises(ValueError, match="may only consume its input"):
        lower([keyed.unkey().sink(print)])


def test_pipeline_with_branch_off_its_result_raises():
    def checkpointed(stream):
        return stream.checkpoint(CallbackSinkWriter(print)).map(_second)

    with pytest.raises(ValueError, match="must lead to the stream it returns"):
        Source("x").group_by(0, checkpointed)


def test_pipeline_returning_a_sink_raises():
    with pytest.raises(TypeError, match="must return a stream"):
        Source("x").group_by(0, lambda stream: stream.sink(print))


def test_non_positive_max_keys_raises():
    with pytest.raises(ValueError, match="max_keys must be positive"):
        Source("x").group_by(0, _chunk_ids, max_keys=0)


# -- Key extractor -----------------------------------------------------------


def test_key_extractor_by_index():
    assert _make_key_extractor(1)(("a", "b")) == "b"


def test_key_extractor_by_field_name():
    assert _make_key_extractor("class_id")({"class_id": 4}) == 4


def test_key_extractor_by_function():
    assert _make_key_extractor(len)([1, 2, 3]) == 3


def test_key_extractor_rejects_other_keys():
    with pytest.raises(TypeError, match="index, a field name or a function"):
        _make_key_extractor(1.5)


# -- Ray ---------------------------------------------------------------------


def test_chunks_items_per_key():
    collector = Collector.remote()
    pipeline = _compile(
        Source("x")
        .group_by(0, _chunk_ids)
        .unkey()
        .write_to(_collecting_writer(collector)),
    )

    try:
        for key, ids in [("a", [1]), ("b", [10]), ("a", [2]), ("b", [11, 12])]:
            pipeline.send("x", (key, _make_set(ids)))
        pipeline.flush()

        assert _by_key(ray.get(collector.get.remote())) == {
            "a": [[1, 2]],
            "b": [[10, 11], [12]],
        }
    finally:
        pipeline.close()


def test_children_are_created_lazily_per_key():
    pipeline = _compile(Source("x").group_by(0, _chunk_ids).unkey().sink(print))

    try:
        assert _child_labels(pipeline) == []

        pipeline.send("x", (3, _make_set([1])))
        pipeline.send("x", (3, _make_set([2])))
        pipeline.flush()
        assert _child_labels(pipeline) == [
            "group_by[3]:0:source(group_by.input)",
            "group_by[3]:1:map(_second)",
            "group_by[3]:2:chunk(2)",
            "group_by[3]:3:map(_ids)",
            "group_by[3]:4:sink(_DemuxResultWriter)",
        ]

        pipeline.send("x", (4, _make_set([3])))
        pipeline.flush()
        assert len(_child_labels(pipeline)) == 10
    finally:
        pipeline.close()


def test_stats_count_items_of_children():
    pipeline = _compile(Source("x").group_by(0, _chunk_ids).unkey().sink(print))

    try:
        for key in ["a", "a", "b"]:
            pipeline.send("x", (key, _make_set([1])))
        pipeline.flush()

        stats = pipeline.stats()
        assert stats["1:group_by"].items_in == 3
        assert stats["1:group_by"].items_out == 2
        assert stats["group_by[a]:2:chunk(2)"].items_in == 2
        assert stats["group_by[b]:2:chunk(2)"].items_in == 1
    finally:
        pipeline.close()


def test_exceeding_max_keys_fails_the_pipeline():
    pipeline = _compile(
        Source("x").group_by(0, _chunk_ids, max_keys=2).unkey().sink(print),
    )

    try:
        pipeline.send("x", ("a", _make_set([1])))
        pipeline.send("x", ("b", _make_set([2])))

        with pytest.raises(ray.exceptions.RayTaskError, match="max_keys=2"):
            for _ in range(500):
                pipeline.send("x", ("c", _make_set([3])))
                time.sleep(0.01)
    finally:
        pipeline.close()


def test_flush_drains_children_before_forwarding():
    collector = Collector.remote()
    # The remainders of chunk(3) are only emitted when the children flush.
    pipeline = _compile(
        Source("x")
        .group_by(0, lambda stream: stream.map(_second).chunk(3).map(_ids))
        .unkey()
        .write_to(_collecting_writer(collector)),
    )

    try:
        for index in range(8):
            pipeline.send("x", (index % 4, _make_set([index])))
        pipeline.flush()

        assert _by_key(ray.get(collector.get.remote())) == {
            key: [[key, key + 4]] for key in range(4)
        }
    finally:
        pipeline.close()


def _reject(s):
    raise ValueError("child failed")


def test_failing_child_after_collect_fails_the_next_send():
    pipeline = _compile(
        Source("x")
        .group_by(0, lambda stream: stream.map(_second).collect(1).map(_reject))
        .unkey()
        .sink(print),
    )

    try:
        pipeline.send("x", ("a", _make_set([1])))

        # Only items of another key keep arriving, so the failed child never
        # receives another item.
        with pytest.raises(ray.exceptions.RayTaskError, match="child failed"):
            for index in range(500):
                pipeline.send("x", (f"other-{index % 2}", _make_set([index])))
                time.sleep(0.01)
    finally:
        pipeline.close()


def _named_child_actors(key):
    return [
        name for name in ray.util.list_named_actors() if f"group_by[{key}]:" in name
    ]


def test_close_terminates_children():
    pipeline = _compile(Source("x").group_by(0, _chunk_ids).unkey().sink(print))
    pipeline.send("x", ("close-test", _make_set([1])))
    pipeline.flush()
    # Source, chunk and sink; the maps run as tasks of their upstream stage.
    assert len(_named_child_actors("close-test")) == 3

    pipeline.close()

    deadline = time.monotonic() + 30
    while _named_child_actors("close-test") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _named_child_actors("close-test") == []


def test_tuple_keys_and_nested_group_by():
    collector = Collector.remote()

    def by_parity(stream):
        return stream.group_by(lambda item: item[0][1] % 2, _chunk_ids).unkey()

    pipeline = _compile(
        Source("x")
        .group_by(lambda item: item[0][0], by_parity)
        .unkey()
        .write_to(_collecting_writer(collector)),
    )

    try:
        for index in range(4):
            pipeline.send("x", (("a", index), _make_set([index])))
        pipeline.flush()

        results = ray.get(collector.get.remote())
        assert sorted(results) == [
            Keyed("a", Keyed(0, [0, 2])),
            Keyed("a", Keyed(1, [1, 3])),
        ]
        assert "group_by[a]:group_by[0]:2:chunk(2)" in pipeline.stats()
    finally:
        pipeline.close()


class Params:
    groups: B.Field[int] = B.Field(default=1)
    offset: B.Field[int] = B.Field(default=0)

    def key(self, item):
        return item % self.groups.value

    def shift(self, item):
        return item + self.offset.value


def test_key_function_and_children_see_parameters():
    collector = Collector.remote()
    params = Params()
    pipeline = _compile(
        Source("x")
        .group_by(params.key, lambda stream: stream.map(params.shift))
        .unkey()
        .write_to(_collecting_writer(collector)),
        parameters={"groups": 2, "offset": 100},
    )

    try:
        for item in range(4):
            pipeline.send("x", item)
        pipeline.flush()

        assert _by_key(ray.get(collector.get.remote())) == {
            0: [100, 102],
            1: [101, 103],
        }
    finally:
        pipeline.close()


def _train(sample):
    return sum(_ids(sample))


def _predict(pair):
    chunk, model = pair
    return [model + i for i in _ids(chunk)]


def _refine(stream):
    """The shape of the RefineClassesAlignPCA sub-pipeline."""
    particles = stream.map(_second)
    model = particles.collect(2).map(_train)
    return particles.chunk(2).combine_latest(model).map(_predict)


def test_pipeline_using_its_input_twice():
    collector = Collector.remote()
    pipeline = _compile(
        Source("x")
        .group_by(0, _refine)
        .unkey()
        .write_to(_collecting_writer(collector)),
    )

    try:
        for key, ids in [(0, [1]), (1, [10]), (0, [2, 3]), (1, [20]), (0, [4])]:
            pipeline.send("x", (key, _make_set(ids)))
        pipeline.flush()

        # Model of key 0: 1 + 2; model of key 1: 10 + 20.
        assert _by_key(ray.get(collector.get.remote())) == {
            0: [[4, 5], [6, 7]],
            1: [[40, 50]],
        }
    finally:
        pipeline.close()
