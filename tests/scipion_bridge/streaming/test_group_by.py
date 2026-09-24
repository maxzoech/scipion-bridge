from typing import Tuple, Optional
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming import Source, Pipeline, FLUSH
from scipion_bridge.core.streaming.ops import GroupByOp, GroupedMapOp


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
        source.group_by("class_id")
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
        source.group_by(lambda p: p.class_id)
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
        source.group_by("class_id")
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

    with pytest.raises(
        KeyError, match="Field or attribute 'nonexistent_field' not found"
    ):
        stream.send(items=Particle(id=1, class_id=1, score=0.5))


def test_reduce_ema_warmup():
    def compute_ema(
        state: Tuple[int, float],
        x: float,
    ) -> Tuple[Tuple[int, float], Optional[float]]:
        count, current_ema = state
        new_count = count + 1
        alpha = 0.1
        if new_count == 1:
            new_ema = x
        else:
            new_ema = alpha * x + (1.0 - alpha) * current_ema

        if new_count <= 1000:
            return (new_count, new_ema), None
        return (new_count, new_ema), new_ema

    received = []
    source = Source("values")
    sink_node = source.reduce(
        compute_ema,
        start=(0, 0.0),
    ).sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    for i in range(1, 1001):
        pipeline.send(values=float(i))

    assert len(received) == 0

    for i in range(1001, 1501):
        pipeline.send(values=float(i))

    assert len(received) == 500

    pipeline.flush()
    assert len(received) == 500


def test_sub_pipeline_online_reduce():
    def compute_ema(
        state: Tuple[int, float],
        p: Particle,
    ) -> Tuple[Tuple[int, float], Optional[float]]:
        count, current_ema = state
        new_count = count + 1
        alpha = 0.2
        if new_count == 1:
            new_ema = p.score
        else:
            new_ema = alpha * p.score + (1.0 - alpha) * current_ema

        if new_count <= 2:
            return (new_count, new_ema), None
        return (new_count, new_ema), new_ema

    received = []
    source = Source("particles")
    sink_node = (
        source.group_by(
            "class_id",
            pipeline=lambda sub: sub.reduce(compute_ema, start=(0, 0.0)),
        )
        .map(lambda ema_score: f"EMA: {ema_score:.2f}")
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    # Class 1: 2 items (warmup)
    pipeline.send(particles=Particle(id=1, class_id=1, score=10.0))
    pipeline.send(particles=Particle(id=2, class_id=1, score=20.0))
    assert len(received) == 0

    # Class 2: 2 items (warmup)
    pipeline.send(particles=Particle(id=3, class_id=2, score=100.0))
    pipeline.send(particles=Particle(id=4, class_id=2, score=200.0))
    assert len(received) == 0

    # Class 1: 3rd item (exceeds warmup, emits immediately)
    pipeline.send(particles=Particle(id=5, class_id=1, score=30.0))
    assert len(received) == 1
    assert received[0][0] == 1
    assert received[0][1].startswith("EMA: ")

    # Class 2: 3rd item (exceeds warmup, emits immediately)
    pipeline.send(particles=Particle(id=6, class_id=2, score=300.0))
    assert len(received) == 2
    assert received[1][0] == 2
    assert received[1][1].startswith("EMA: ")

    pipeline.flush()
    assert len(received) == 2


def test_sub_pipeline_chunk_and_flush():
    received = []
    source = Source("particles")
    sink_node = source.group_by(
        "class_id",
        pipeline=lambda sub: sub.chunk(2),
    ).sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    # Send 2 items for class 1, 1 item for class 2
    pipeline.send(particles=Particle(id=1, class_id=1, score=1.0))
    pipeline.send(particles=Particle(id=2, class_id=2, score=2.0))
    pipeline.send(particles=Particle(id=3, class_id=1, score=3.0))

    # Class 1 reached size 2 (items 1 and 3) -> emits immediately
    assert len(received) == 1
    k, chunk = received[0]
    assert k == 1
    assert len(chunk) == 2
    assert chunk[0].id == 1
    assert chunk[1].id == 3

    # Send 1 more for class 1 (id 4) -> buffered, len(received) still 1
    pipeline.send(particles=Particle(id=4, class_id=1, score=4.0))
    assert len(received) == 1

    # Send 1 more for class 1 (id 5) -> Class 1 has [id 4, id 5], emits second chunk
    pipeline.send(particles=Particle(id=5, class_id=1, score=5.0))
    assert len(received) == 2
    assert received[1][0] == 1
    assert len(received[1][1]) == 2
    assert received[1][1][0].id == 4
    assert received[1][1][1].id == 5

    # Send 1 more for class 1 (id 6) -> 1 leftover in class 1
    pipeline.send(particles=Particle(id=6, class_id=1, score=6.0))
    # Send 1 more for class 2 (id 7) -> Class 2 has [id 2, id 7], emits chunk for class 2
    pipeline.send(particles=Particle(id=7, class_id=2, score=7.0))
    assert len(received) == 3
    assert received[2][0] == 2
    assert len(received[2][1]) == 2
    assert received[2][1][0].id == 2
    assert received[2][1][1].id == 7

    # Send 1 more for class 2 (id 8) -> 1 leftover in class 2
    pipeline.send(particles=Particle(id=8, class_id=2, score=8.0))
    assert len(received) == 3

    # Flush emits leftovers for class 1 (id 6) and class 2 (id 8)
    pipeline.flush()
    assert len(received) == 5
    flushed_keys = [item[0] for item in received[3:]]
    assert 1 in flushed_keys
    assert 2 in flushed_keys
