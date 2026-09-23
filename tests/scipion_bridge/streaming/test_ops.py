from typing import Tuple

import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.streaming.ops import (
    Source,
    MapOp,
    ChunkOp,
    MinChunkOp,
    CollectOp,
    CombineOp,
    FlattenOp,
)
from scipion_bridge.core.streaming.node import FLUSH
from scipion_bridge.core.streaming.pipeline import Pipeline


class Metadata(B.Struct):
    foo: int


class Particle(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(256, 256))
    metadata: Metadata


class ParticleEmbeddings(B.Struct):
    particles: B.Set[Particle] = B.Set[Particle](capacity=10)
    latent_code: B.Array[np.float32] = B.Array(shape=(128,))


def test_chunk_single_elements_into_set():
    received = []

    # Pipeline: Source -> chunk(3) -> sink
    source = Source("items")
    sink_node = source.chunk(3).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1]))
    stream.send(items=B.Set[Particle]([p2]))
    assert len(received) == 0  # Not enough items yet

    p3 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 3.0, metadata=Metadata(foo=3)
    )
    stream.send(items=B.Set[Particle]([p3]))

    assert len(received) == 1
    chunk_result = received[0]
    assert isinstance(chunk_result, B.Set)
    assert len(chunk_result) == 3
    assert chunk_result[0].metadata.foo == 1
    assert chunk_result[1].metadata.foo == 2
    assert chunk_result[2].metadata.foo == 3


def test_chunk_rechunk_large_set_into_small_sets():
    received = []

    # Pipeline: Source -> chunk(6) -> chunk(2) -> sink
    source = Source("items")
    sink_node = source.chunk(6).chunk(2).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    # Emit 6 single-element Set containers
    for i in range(1, 7):
        p = Particle(
            pixels=np.zeros([256, 256], dtype=np.float32) + i, metadata=Metadata(foo=i)
        )
        stream.send(items=B.Set[Particle]([p]))

    # Should have split the 6-element set into 3 smaller sets of capacity 2
    assert len(received) == 3

    for chunk_idx, chunk in enumerate(received):
        assert isinstance(chunk, B.Set)
        assert len(chunk) == 2
        assert chunk[0].metadata.foo == chunk_idx * 2 + 1
        assert chunk[1].metadata.foo == chunk_idx * 2 + 2


def test_chunk_large_number_of_elements():
    received = []

    # Pipeline: Source -> chunk(1500) -> sink
    source = Source("items")
    sink_node = source.chunk(1500).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    for i in range(1, 1501):
        p = Particle(
            pixels=np.zeros([256, 256], dtype=np.float32) + i, metadata=Metadata(foo=i)
        )
        stream.send(items=B.Set[Particle]([p]))

    assert len(received) == 1
    assert len(received[0]) == 1500


def test_map_op():
    received = []

    source = Source("numbers")
    sink_node = (
        source.map(lambda x: x + 10)
        .map(lambda x: x * 2)
        .sink(lambda x: received.append(x))
    )

    stream = Pipeline.from_sink(sink_node)
    stream.send(numbers=5)  # 5 -> 15 -> 30
    stream.send(numbers=1)  # 1 -> 11 -> 22

    assert received == [30, 22]


def test_map_op_transform_struct():
    received = []

    def _encode_particles(particle_set: B.Set[Particle]) -> ParticleEmbeddings:
        latent = np.full(128, 0.42, dtype=np.float32)
        return ParticleEmbeddings(particles=particle_set, latent_code=latent)

    source = Source("particles")
    sink_node = source.map(_encode_particles).sink(lambda x: received.append(x))

    stream = Pipeline.from_sink(sink_node)

    batch_size = 3
    particle_set = B.Set[Particle](capacity=batch_size)
    for idx in range(batch_size):
        pixels = np.ones([256, 256], dtype=np.float32) * idx
        particle_set[idx] = Particle(pixels=pixels, metadata=Metadata(foo=idx))

    stream.send(particles=particle_set)

    assert len(received) == 1
    result = received[0]
    assert isinstance(result, ParticleEmbeddings)
    assert isinstance(result.particles, B.Set)
    assert len(result.particles) == batch_size
    assert result.latent_code.shape == (128,)
    assert np.allclose(result.latent_code, 0.42)


def test_chunk_op_flush_partial_set():
    received = []

    source = Source("items")
    sink_node = source.chunk(5).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1]))
    stream.send(items=B.Set[Particle]([p2]))

    assert len(received) == 0  # 2 elements, chunk_size=5 -> buffered

    stream.flush()

    assert len(received) == 1
    flushed_chunk = received[0]
    assert isinstance(flushed_chunk, B.Set)
    assert len(flushed_chunk) == 2
    assert flushed_chunk[0].metadata.foo == 1
    assert flushed_chunk[1].metadata.foo == 2


def test_chunk_op_flush_empty_queue():
    received = []

    source = Source("items")
    sink_node = source.chunk(5).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    stream.flush()

    assert len(received) == 0  # Empty queue -> nothing emitted on flush


def test_min_chunk_buffers_and_emits_complete_set():
    received = []

    source = Source("items")
    sink_node = source.min_chunk(5).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1, p2]))
    assert len(received) == 0  # 2 elements < 5 -> buffered

    p3 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 3.0, metadata=Metadata(foo=3)
    )
    p4 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 4.0, metadata=Metadata(foo=4)
    )
    p5 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 5.0, metadata=Metadata(foo=5)
    )
    p6 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 6.0, metadata=Metadata(foo=6)
    )

    # Send set of 4 (total capacity = 2 + 4 = 6 >= 5) -> emits complete set of 6 without splitting
    stream.send(items=B.Set[Particle]([p3, p4, p5, p6]))

    assert len(received) == 1
    result = received[0]
    assert isinstance(result, B.Set)
    assert len(result) == 6


def test_min_chunk_flush():
    received = []

    source = Source("items")
    sink_node = source.min_chunk(5).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1, p2]))
    assert len(received) == 0  # 2 elements < 5 -> buffered

    stream.flush()

    assert len(received) == 1
    flushed = received[0]
    assert isinstance(flushed, B.Set)
    assert len(flushed) == 2


def test_collect_buffers_emits_once_and_ignores_subsequent_items():
    received = []

    source = Source("items")
    sink_node = source.collect(5).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )
    p3 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 3.0, metadata=Metadata(foo=3)
    )
    p4 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 4.0, metadata=Metadata(foo=4)
    )

    # Send 2 elements (2 < 5) -> buffered
    stream.send(items=B.Set[Particle]([p1, p2]))
    assert len(received) == 0

    # Send 2 elements (total 4 < 5) -> buffered
    stream.send(items=B.Set[Particle]([p3, p4]))
    assert len(received) == 0

    p5 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 5.0, metadata=Metadata(foo=5)
    )
    p6 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 6.0, metadata=Metadata(foo=6)
    )

    # Send 2 elements (total 4 + 2 = 6 >= 5 threshold reached) -> emits concatenated set truncated to count 5
    stream.send(items=B.Set[Particle]([p5, p6]))
    assert len(received) == 1
    result = received[0]
    assert isinstance(result, B.Set)
    assert len(result) == 5

    # Subsequent sends after threshold reached should be ignored
    p7 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 7.0, metadata=Metadata(foo=7)
    )
    stream.send(items=B.Set[Particle]([p7]))
    assert len(received) == 1


def test_collect_single_large_set_truncated_to_count():
    received = []

    source = Source("items")
    sink_node = source.collect(3).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    particles = [
        Particle(
            pixels=np.zeros([256, 256], dtype=np.float32) + float(i),
            metadata=Metadata(foo=i),
        )
        for i in range(1, 10)
    ]

    # Send a single set with 9 elements when count is 3
    stream.send(items=B.Set[Particle](particles))
    assert len(received) == 1
    result = received[0]
    assert isinstance(result, B.Set)
    assert len(result) == 3
    assert result[0].metadata.foo == 1
    assert result[1].metadata.foo == 2
    assert result[2].metadata.foo == 3


def test_collect_flush_before_threshold():
    received = []

    source = Source("items")
    sink_node = source.collect(5).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1, p2]))
    assert len(received) == 0

    stream.flush()

    assert len(received) == 1
    flushed = received[0]
    assert isinstance(flushed, B.Set)
    assert len(flushed) == 2


def test_collect_count_none_buffers_until_flush():
    received = []

    source = Source("items")
    sink_node = source.collect(None).sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )
    p3 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 3.0, metadata=Metadata(foo=3)
    )

    stream.send(items=B.Set[Particle]([p1]))
    stream.send(items=B.Set[Particle]([p2, p3]))

    # count=None means nothing is emitted until flush signal
    assert len(received) == 0

    stream.flush()

    assert len(received) == 1
    batch = received[0]
    assert isinstance(batch, B.Set)
    assert len(batch) == 3
    assert batch[0].metadata.foo == 1
    assert batch[1].metadata.foo == 2
    assert batch[2].metadata.foo == 3


def test_combine_op():
    received = []

    s1 = Source("a")
    s2 = Source("b")
    combined = CombineOp(s1, s2)
    sink_node = combined.sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    stream.send(a=1)
    assert len(received) == 0  # s2 hasn't emitted yet

    stream.send(b="x")
    assert received == [(1, "x")]

    stream.send(a=2)
    assert received == [(1, "x"), (2, "x")]

    stream.send(b="y")
    assert received == [(1, "x"), (2, "x"), (2, "y")]


def test_combine_op_buffers_pre_emission_items():
    received = []

    s1 = Source("a")
    s2 = Source("b")
    combined = CombineOp(s1, s2)
    sink_node = combined.sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    stream.send(a=10)
    stream.send(a=20)
    stream.send(a=30)
    assert len(received) == 0  # s2 hasn't emitted yet

    stream.send(b="trained")
    assert received == [(10, "trained"), (20, "trained"), (30, "trained")]

    stream.send(a=40)
    assert received == [
        (10, "trained"),
        (20, "trained"),
        (30, "trained"),
        (40, "trained"),
    ]


def test_flatten_op_unrolls_set():
    received = []

    source = Source("items")
    sink_node = source.flatten().sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )
    p3 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 3.0, metadata=Metadata(foo=3)
    )

    stream.send(items=B.Set[Particle]([p1, p2]))
    assert len(received) == 2
    assert received[0].metadata.foo == 1
    assert received[1].metadata.foo == 2

    stream.send(items=B.Set[Particle]([p3]))
    assert len(received) == 3
    assert received[2].metadata.foo == 3


def test_flatten_op_empty_set():
    received = []

    source = Source("items")
    sink_node = source.flatten().sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    stream.send(items=B.Set[Particle]([]))
    assert len(received) == 0


def test_flatten_op_unrolls_tuple():
    received = []

    source = Source("items")
    sink_node = source.flatten().sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    stream.send(items=(10, 20, 30))
    assert received == [10, 20, 30]


def test_flatten_op_flush_lifecycle():
    received = []

    source = Source("items")
    sink_node = source.flatten().sink(lambda x: received.append(x))
    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1, p2]))
    assert len(received) == 2

    # Flush should not crash and should not append to normal sink output
    stream.flush()
    assert len(received) == 2

    op = FlattenOp()
    assert op._prepare_unroll(FLUSH) == [FLUSH]


def test_flatten_op_invalid_type_raises():
    op = FlattenOp()
    with pytest.raises(
        TypeError, match="FlattenOp expected an iterable or struct.Set, got int"
    ):
        op._prepare_unroll(42)

    with pytest.raises(
        TypeError, match="FlattenOp expected an iterable or struct.Set, got Particle"
    ):
        p = Particle(
            pixels=np.zeros([256, 256], dtype=np.float32), metadata=Metadata(foo=1)
        )
        op._prepare_unroll(p)


def test_basic_accumulate():
    def _acc_fn(state: B.Set, new_value: B.Set):
        state = B.concat([state, new_value])

        return state, state

    received = []

    source = Source("items")
    sink_node = source.accumulate(
        _acc_fn,
    ).sink(lambda x: received.append(x))

    stream = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 1.0, metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32) + 2.0, metadata=Metadata(foo=2)
    )

    stream.send(items=B.Set[Particle]([p1]))
    stream.send(items=B.Set[Particle]([p2]))

    assert len(received[-1]) == 2


# def test_basic_accumulate_start_value():

#     class Sample(B.Struct):

#         label: int

#     class Group(B.Struct):

#         samples: B.Set[Sample]

#     def _acc_fn(state: B.Set[Group], new_value: Tuple[int, B.Set]):
#         idx, next_set = new_value

#         current_group = state[idx]
#         new_set = B.Set.concat(current_group.samples, next_set)

#         print(new_set)

#         return state, new_value

#     received = []

#     source = Source("items")
#     sink_node = (
#         source.accumulate(
#             _acc_fn,
#             start=B.Set[Group](capacity=20)
#         )
#         .sink(lambda x: received.append(x))
#     )

#     stream = Pipeline.from_sink(sink_node)

#     stream.send(items=(0, B.Set[Sample]([Sample(label=0)])))


if __name__ == "__main__":
    from scipion_bridge.backend.standalone.container import configure_default_env
def test_combine_latest_flush_barrier():
    received = []
    source_a = Source("a")
    source_b = Source("b")
    # ReduceOp only emits its accumulated list upon receiving FlushSignal
    sink_node = (
        source_a.combine_latest(source_b)
        .reduce(lambda acc, x: acc + [x], start=[])
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(a=1)
    pipeline.send(b=10)
    assert len(received) == 0  # ReduceOp does not emit until flush

    # Flush source_a only
    pipeline._backend_sources["a"].emit(FLUSH)
    # Downstream ReduceOp should NOT have flushed yet because source_b is still active
    assert len(received) == 0

    # source_b continues emitting while source_a was already flushed
    pipeline.send(b=20)
    assert len(received) == 0

    # Now flush source_b - barrier is satisfied, exactly one FLUSH triggers ReduceOp emission
    pipeline._backend_sources["b"].emit(FLUSH)
    assert len(received) == 1
    assert received[0] == [(1, 10), (1, 20)]


def test_as_keyed_and_unkey():
    received = []
    source = Source("pairs")
    sink_node = (
        source.as_keyed()
        .reduce_by_key(lambda a, b: a + b)
        .unkey()
        .map(lambda pair: f"{pair[0]}:{pair[1]}")
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(pairs=("k1", 1))
    pipeline.send(pairs=("k2", 10))
    pipeline.send(pairs=("k1", 2))
    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 2
    assert "k1:3" in received
    assert "k2:10" in received


def test_as_keyed_type_validation():
    received = []
    source = Source("bad_items")
    sink_node = source.as_keyed().sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    with pytest.raises(TypeError, match="as_keyed expects \\(key, value\\) pairs"):
        pipeline.send(bad_items="not_a_tuple")


def test_collect_struct_instances():
    received = []
    source = Source("structs")
    sink_node = source.collect().sink(lambda x: received.append(x))
    pipeline = Pipeline.from_sink(sink_node)

    pipeline.send(structs=Metadata(foo=1))
    pipeline.send(structs=Metadata(foo=2))
    pipeline.send(structs=Metadata(foo=3))
    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 1
    collected_set = received[0]
    assert isinstance(collected_set, B.Set)
    assert len(collected_set) == 3
    assert collected_set[0].foo == 1
    assert collected_set[1].foo == 2
    assert collected_set[2].foo == 3


def test_keyed_reduce_with_struct_concat_and_unkey():
    received = []
    source = Source("keyed_sets")
    sink_node = (
        source.as_keyed()
        .reduce_by_key(lambda acc, chunk: B.concat([acc, chunk]))
        .unkey()
        .map(lambda item: (item[0], len(item[1])))
        .sink(lambda x: received.append(x))
    )
    pipeline = Pipeline.from_sink(sink_node)

    p1 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32), metadata=Metadata(foo=1)
    )
    p2 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32), metadata=Metadata(foo=2)
    )
    p3 = Particle(
        pixels=np.zeros([256, 256], dtype=np.float32), metadata=Metadata(foo=3)
    )

    pipeline.send(keyed_sets=(0, B.Set[Particle]([p1, p2])))
    pipeline.send(keyed_sets=(1, B.Set[Particle]([p3])))
    pipeline.send(keyed_sets=(0, B.Set[Particle]([p3])))

    assert len(received) == 0

    pipeline.flush()
    assert len(received) == 2
    results = dict(received)
    assert results[0] == 3  # 2 + 1
    assert results[1] == 1  # 1
