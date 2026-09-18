import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming import Source, Pipeline, FLUSH, FlushSignal
from scipion_bridge.core.streaming.ops import GroupByOp, GroupedOp, GroupedMapOp


class Particle(B.Struct):
    id: int
    class_id: int
    score: float


def test_group_by_with_callable():
    received = []
    source = Source("items")
    sink_node = source.group_by(lambda p: p.class_id).sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    p1 = Particle(id=1, class_id=10, score=0.8)
    p2 = Particle(id=2, class_id=20, score=0.9)

    pipeline.send(items=p1)
    pipeline.send(items=p2)

    assert len(received) == 2
    assert received[0] == (10, p1)
    assert received[1] == (20, p2)
    # Ensure full item is preserved
    assert received[0][1].id == 1
    assert received[0][1].score == 0.8


def test_group_by_with_string_attribute():
    received = []
    source = Source("items")
    sink_node = source.group_by("class_id").sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    p1 = Particle(id=1, class_id=5, score=0.1)
    p2 = Particle(id=2, class_id=8, score=0.2)

    pipeline.send(items=p1)
    pipeline.send(items=p2)

    assert len(received) == 2
    assert received[0] == (5, p1)
    assert received[1] == (8, p2)


def test_group_by_with_dict_and_string_key():
    received = []
    source = Source("items")
    sink_node = source.group_by("group").sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    d1 = {"group": "A", "val": 100}
    d2 = {"group": "B", "val": 200}

    pipeline.send(items=d1)
    pipeline.send(items=d2)

    assert len(received) == 2
    assert received[0] == ("A", d1)
    assert received[1] == ("B", d2)


def test_group_by_with_integer_index():
    received = []
    source = Source("tuples")
    sink_node = source.group_by(0).sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    t1 = ("class_1", 123)
    t2 = ("class_2", 456)

    pipeline.send(tuples=t1)
    pipeline.send(tuples=t2)

    assert len(received) == 2
    assert received[0] == ("class_1", t1)
    assert received[1] == ("class_2", t2)


def test_grouped_op_map():
    received = []
    source = Source("items")
    sink_node = (
        source
        .group_by("class_id")
        .map(lambda p: p.score * 10)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    p1 = Particle(id=1, class_id=3, score=0.5)
    p2 = Particle(id=2, class_id=3, score=0.7)

    pipeline.send(items=p1)
    pipeline.send(items=p2)

    assert len(received) == 2
    assert received[0] == (3, 5.0)
    assert received[1] == (3, 7.0)


def test_grouped_op_chained_map():
    received = []
    source = Source("items")
    sink_node = (
        source
        .group_by(lambda p: p.class_id)
        .map(lambda p: p.score)
        .map(lambda s: f"Score: {s:.1f}")
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    p = Particle(id=1, class_id=42, score=0.9)
    pipeline.send(items=p)

    assert len(received) == 1
    assert received[0] == (42, "Score: 0.9")


def test_group_by_flush_signal():
    received = []
    source = Source("items")
    sink_node = (
        source
        .group_by("class_id")
        .map(lambda p: p.score)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    p = Particle(id=1, class_id=1, score=0.5)
    pipeline.send(items=p)
    assert len(received) == 1
    assert received[0] == (1, 0.5)

    # Flush propagates through group_by and map without error, and is absorbed at Sink
    pipeline.flush()
    assert len(received) == 1

    # Verify directly that FLUSH signal is preserved by the op transforms
    group_op = GroupByOp("class_id")
    assert group_op._group_func(FLUSH) is FLUSH

    map_op = GroupedMapOp(lambda x: x * 2)
    assert map_op._map_func(FLUSH) is FLUSH


def test_group_by_invalid_key():
    source = Source("items")
    # Invalid key type
    op = source.group_by(3.14)  # type: ignore
    stream = Pipeline.from_sink(op.sink(lambda x: None))

    with pytest.raises(TypeError, match="Unsupported key type"):
        stream.send(items="dummy")


def test_group_by_missing_key():
    source = Source("items")
    op = source.group_by("nonexistent_field")
    stream = Pipeline.from_sink(op.sink(lambda x: None))

    with pytest.raises(KeyError, match="Field or attribute 'nonexistent_field' not found"):
        stream.send(items=Particle(id=1, class_id=1, score=0.5))
