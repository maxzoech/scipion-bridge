from typing import Dict, Tuple, Any
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming import (
    Source,
    Pipeline,
    FLUSH,
    FlushSignal,
    ReduceOp,
    AccumulateOp,
    KeyedChunkOp,
    KeyedReduceOp,
)


class Particle(B.Struct):
    id: int
    class_id: int
    score: float


class ClassModel(B.Struct):
    class_id: int
    count: int
    mean_score: float


# ---------------------------------------------------------------------------
# ReduceOp Tests
# ---------------------------------------------------------------------------

def test_reduce_op_with_start():
    received = []
    source = Source("numbers")
    sink_node = source.reduce(lambda acc, x: acc + x, start=10).sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(numbers=1)
    pipeline.send(numbers=2)
    pipeline.send(numbers=3)

    # In a reduction, nothing is emitted until flush
    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 1
    assert received[0] == 16  # 10 + 1 + 2 + 3


def test_reduce_op_without_start():
    received = []
    source = Source("words")
    sink_node = source.reduce(lambda acc, x: f"{acc}-{x}").sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(words="a")
    pipeline.send(words="b")
    pipeline.send(words="c")

    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 1
    assert received[0] == "a-b-c"


def test_reduce_op_empty_stream():
    received_with_start = []
    source1 = Source("items")
    sink1 = source1.reduce(lambda a, b: a + b, start=0).sink(lambda x: received_with_start.append(x))
    pipeline1 = Pipeline.from_sink(sink1)
    pipeline1.flush()
    # If start was provided, empty stream emits start value
    assert received_with_start == [0]

    received_no_start = []
    source2 = Source("items")
    sink2 = source2.reduce(lambda a, b: a + b).sink(lambda x: received_no_start.append(x))
    pipeline2 = Pipeline.from_sink(sink2)
    pipeline2.flush()
    # If no start was provided, empty stream emits nothing
    assert received_no_start == []


# ---------------------------------------------------------------------------
# AccumulateOp Tests
# ---------------------------------------------------------------------------

def test_accumulate_op():
    received = []
    source = Source("numbers")
    sink_node = source.accumulate(lambda acc, x: (acc + x, acc + x), start=0).sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(numbers=1)
    pipeline.send(numbers=2)
    pipeline.send(numbers=3)

    # Accumulate emits running totals on each item
    assert received == [1, 3, 6]

    pipeline.flush()
    # Flush should not duplicate emissions
    assert received == [1, 3, 6]


# ---------------------------------------------------------------------------
# KeyedChunkOp Tests
# ---------------------------------------------------------------------------

def test_keyed_chunk_with_struct_particles():
    received = []
    source = Source("particles")
    sink_node = (
        source
        .group_by(lambda p: p.class_id)
        .chunk(3)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    # Send particles belonging to class 1 and class 2
    pipeline.send(particles=Particle(id=1, class_id=1, score=0.1))
    pipeline.send(particles=Particle(id=2, class_id=2, score=0.2))
    pipeline.send(particles=Particle(id=3, class_id=1, score=0.3))
    assert len(received) == 0  # Neither class has reached 3

    pipeline.send(particles=Particle(id=4, class_id=1, score=0.4))
    # Class 1 has 3 items -> emits immediately!
    assert len(received) == 1
    k, chunk = received[0]
    assert k == 1
    assert isinstance(chunk, B.Set)
    assert len(chunk) == 3
    assert chunk[0].id == 1
    assert chunk[1].id == 3
    assert chunk[2].id == 4

    # Class 2 only has 1 item so far. Now flush.
    pipeline.flush()
    assert len(received) == 2
    k2, chunk2 = received[1]
    assert k2 == 2
    assert isinstance(chunk2, B.Set)
    assert len(chunk2) == 1
    assert chunk2[0].id == 2


def test_keyed_chunk_with_set_input():
    received = []
    source = Source("batches")
    sink_node = (
        source
        .flatten()
        .group_by(lambda p: p.class_id)
        .chunk(4)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    set_c1 = B.Set[Particle]([
        Particle(id=1, class_id=10, score=0.1),
        Particle(id=2, class_id=10, score=0.2),
    ])
    set_c2 = B.Set[Particle]([
        Particle(id=10, class_id=20, score=0.5),
    ])

    pipeline.send(batches=set_c1)
    pipeline.send(batches=set_c2)

    # Send more to class 10 to hit threshold of 4
    more_c1 = B.Set[Particle]([
        Particle(id=3, class_id=10, score=0.3),
        Particle(id=4, class_id=10, score=0.4),
        Particle(id=5, class_id=10, score=0.5),
    ])
    pipeline.send(batches=more_c1)

    assert len(received) == 1
    k, chunk = received[0]
    assert k == 10
    assert isinstance(chunk, B.Set)
    assert len(chunk) == 4
    assert [p.id for p in chunk] == [1, 2, 3, 4]

    pipeline.flush()
    # Leftovers: class 10 has 1 leftover (id 5), class 20 has 1 leftover (id 10)
    assert len(received) == 3
    assert received[1][0] == 10
    assert len(received[1][1]) == 1
    assert received[1][1][0].id == 5

    assert received[2][0] == 20
    assert len(received[2][1]) == 1
    assert received[2][1][0].id == 10


def test_keyed_chunk_with_prekeyed_set_batches():
    received = []
    source = Source("keyed_sets")
    # source emits (key, Set[Particle])
    sink_node = (
        source
        .group_by(0)
        .map(lambda x: x[1])
        .chunk(3)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    s1 = B.Set[Particle]([Particle(id=1, class_id=1, score=0.1), Particle(id=2, class_id=1, score=0.2)])
    s2 = B.Set[Particle]([Particle(id=3, class_id=1, score=0.3), Particle(id=4, class_id=1, score=0.4)])
    pipeline.send(keyed_sets=(1, s1))
    assert len(received) == 0

    pipeline.send(keyed_sets=(1, s2))
    assert len(received) == 1
    k, chunk = received[0]
    assert k == 1
    assert len(chunk) == 3

    pipeline.flush()
    assert len(received) == 2
    assert len(received[1][1]) == 1


def test_keyed_chunk_with_scalar_values():
    received = []
    source = Source("pairs")
    sink_node = (
        source
        .group_by(0)  # tuple key is index 0
        .map(lambda x: x[1])  # extract second item
        .chunk(2)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(pairs=("group_a", 100))
    pipeline.send(pairs=("group_b", 200))
    pipeline.send(pairs=("group_a", 101))

    # group_a hits 2
    assert len(received) == 1
    assert received[0] == ("group_a", [100, 101])

    pipeline.flush()
    # group_b leftover
    assert len(received) == 2
    assert received[1] == ("group_b", [200])


# ---------------------------------------------------------------------------
# KeyedReduceOp Tests
# ---------------------------------------------------------------------------

def test_keyed_reduce():
    received = []
    source = Source("items")
    sink_node = (
        source
        .group_by("class_id")
        .reduce_by_key(lambda acc, p: acc + p.score, start=0.0)
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(items=Particle(id=1, class_id=1, score=1.5))
    pipeline.send(items=Particle(id=2, class_id=2, score=10.0))
    pipeline.send(items=Particle(id=3, class_id=1, score=2.5))
    pipeline.send(items=Particle(id=4, class_id=2, score=20.0))

    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 2
    results = dict(received)
    assert pytest.approx(results[1]) == 4.0
    assert pytest.approx(results[2]) == 30.0


# ---------------------------------------------------------------------------
# Full 2D Classification Pipeline Pattern
# ---------------------------------------------------------------------------

def test_2d_classification_pipeline_pattern():
    def compute_class_alignment(particles_chunk: B.Set[Particle]) -> ClassModel:
        cid = particles_chunk[0].class_id
        scores = [p.score for p in particles_chunk]
        return ClassModel(
            class_id=cid,
            count=len(particles_chunk),
            mean_score=float(sum(scores) / len(scores)),
        )

    def accumulate_classes(acc: Dict[int, ClassModel], item: Tuple[int, ClassModel]) -> Dict[int, ClassModel]:
        cid, model = item
        acc[cid] = model
        return acc

    received = []
    source = Source("particles")
    sink_node = (
        source
        .flatten()
        .group_by(lambda p: p.class_id)
        .chunk(2)
        .map(compute_class_alignment)
        .reduce(accumulate_classes, start={})
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    p1 = Particle(id=1, class_id=10, score=2.0)
    p2 = Particle(id=2, class_id=20, score=4.0)
    p3 = Particle(id=3, class_id=10, score=4.0)
    p4 = Particle(id=4, class_id=20, score=6.0)

    pipeline.send(particles=B.Set[Particle]([p1, p2, p3, p4]))
    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 1
    final_models = received[0]
    assert isinstance(final_models, dict)
    assert 10 in final_models
    assert 20 in final_models
    assert final_models[10].count == 2
    assert final_models[10].mean_score == 3.0
    assert final_models[20].count == 2
    assert final_models[20].mean_score == 5.0
