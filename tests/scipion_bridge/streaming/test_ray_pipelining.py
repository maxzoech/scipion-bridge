"""Tests for pipelined (overlapping) stage execution in the Ray backend."""

import asyncio
import random
import tempfile
import threading
import time

import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import (
    DEFAULT_BUFFER_SIZE,
    RayBackend,
    _DEFAULT_MAX_IN_FLIGHT,
)
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter
from scipion_bridge.core.streaming.spill import pickle_spill_store

# Timing assertions: run on one xdist worker, after one another.
pytestmark = [
    pytest.mark.usefixtures("ray_cluster"),
    pytest.mark.xdist_group("timing"),
]


class PipeItem(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[PipeItem](capacity=len(ids))
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


@ray.remote
class Gate:
    """Async actor blocking callers of ``wait`` until ``open`` is called."""

    def __init__(self):
        self._event = asyncio.Event()

    async def wait(self):
        await self._event.wait()

    def open(self):
        self._event.set()


def _collecting_writer(collector):
    return CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))


def test_queue_size_must_be_positive():
    with pytest.raises(ValueError, match="Queue size must be positive"):
        RayBackend(init_ray=False, queue_size=0)


def test_order_is_preserved_across_stages():
    collector = Collector.remote()

    def jittered_increment(x):
        time.sleep(random.uniform(0.0, 0.01))
        return x + 1

    source = Source("numbers")
    stage = source.map(jittered_increment)
    stage.checkpoint(_collecting_writer(collector))
    final = Collector.remote()
    sink_node = stage.map(jittered_increment).write_to(_collecting_writer(final))
    ckpt_sink = stage.downstream[0]

    backend = RayBackend(init_ray=False, queue_size=2)
    with Pipeline.from_sink(sink_node, ckpt_sink, backend=backend) as pipeline:
        for i in range(30):
            pipeline.send(numbers=i)

    assert ray.get(collector.get.remote()) == [i + 1 for i in range(30)]
    assert ray.get(final.get.remote()) == [i + 2 for i in range(30)]


def test_send_does_not_wait_for_downstream_compute():
    collector = Collector.remote()
    map_duration = 0.3
    n_items = 6

    def slow_identity(x):
        time.sleep(map_duration)
        return x

    source = Source("numbers")
    sink_node = source.map(slow_identity).write_to(_collecting_writer(collector))

    backend = RayBackend(init_ray=False, queue_size=n_items)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        t_start = time.perf_counter()
        for i in range(n_items):
            pipeline.send(numbers=i)
        send_duration = time.perf_counter() - t_start

    # Sequential execution would block each send for a full map call.
    assert send_duration < 0.5 * map_duration * n_items
    assert ray.get(collector.get.remote()) == list(range(n_items))


def test_chunk_and_map_overlap_with_producer():
    collector = Collector.remote()
    produce_duration = 0.3
    map_duration = 0.6
    n_sends = 8

    def slow_model(s):
        time.sleep(map_duration)
        return s

    source = Source("items")
    sink_node = source.chunk(4).map(slow_model).write_to(_collecting_writer(collector))

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        t_start = time.perf_counter()
        for i in range(n_sends):
            time.sleep(produce_duration)  # Simulates reading the next input chunk.
            pipeline.send(items=_make_set([2 * i, 2 * i + 1]))
    elapsed = time.perf_counter() - t_start

    n_chunks = n_sends * 2 // 4
    sequential = n_sends * produce_duration + n_chunks * map_duration
    assert elapsed < sequential - map_duration
    assert [_ids(s) for s in ray.get(collector.get.remote())] == [
        list(range(4 * c, 4 * c + 4)) for c in range(n_chunks)
    ]


def _gated_writer(gate, collector):
    def gated_write(x):
        ray.get(gate.wait.remote())
        ray.get(collector.append.remote(x))

    return CallbackSinkWriter(gated_write)


def _stage(stats, suffix):
    (stage,) = [s for label, s in stats.items() if label.endswith(suffix)]
    return stage


def test_buffer_size_bounds_items_in_flight():
    collector = Collector.remote()
    gate = Gate.remote()
    n_items = 20

    source = Source("numbers")
    sink_node = source.map(lambda x: x).write_to(_gated_writer(gate, collector))

    backend = RayBackend(
        init_ray=False,
        queue_size=1,
        max_in_flight=1,
        buffer_size=1,
    )
    pipeline = Pipeline.from_sink(sink_node, backend=backend)

    sent = []

    def produce():
        for i in range(n_items):
            pipeline.send(numbers=i)
            sent.append(i)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    producer.join(timeout=2.0)

    assert producer.is_alive()
    assert len(sent) < n_items

    ray.get(gate.open.remote())
    producer.join(timeout=30.0)
    assert not producer.is_alive()

    pipeline.flush()
    assert ray.get(collector.get.remote()) == list(range(n_items))


def test_unbounded_buffer_lets_send_run_ahead_of_a_blocked_stage():
    collector = Collector.remote()
    gate = Gate.remote()
    n_items = 20

    source = Source("numbers")
    sink_node = source.map(lambda x: x).write_to(_gated_writer(gate, collector))
    backend = RayBackend(init_ray=False, buffer_size=None)
    pipeline = Pipeline.from_sink(sink_node, backend=backend)

    def produce():
        for i in range(n_items):
            pipeline.send(numbers=i)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    producer.join(timeout=10.0)
    assert not producer.is_alive()

    flusher = threading.Thread(target=pipeline.flush, daemon=True)
    flusher.start()
    flusher.join(timeout=0.5)
    assert flusher.is_alive()

    ray.get(gate.open.remote())
    flusher.join(timeout=30.0)
    assert not flusher.is_alive()

    assert ray.get(collector.get.remote()) == list(range(n_items))
    assert _stage(pipeline.stats(), "sink(CallbackSinkWriter)").buffered_peak > 2
    pipeline.close()


@ray.remote
class Progress:
    """Counts the items a sink has written, for the driver to compare with."""

    def __init__(self):
        self.count = 0

    def add(self, count):
        self.count += count

    def get(self):
        return self.count


# Rows of a Set of 1 MB: Ray moves such items through the object store.
_LARGE_ROWS = 2**17


def _slow_counting_writer(progress, duration):
    def write(chunk):
        time.sleep(duration)
        ray.get(progress.add.remote(len(chunk) // _LARGE_ROWS))

    return CallbackSinkWriter(write)


def _lead_over_slow_sink(backend, n_sends, sends_per_chunk):
    """Most sends the driver ran ahead of a slow sink, with the stage stats."""
    progress = Progress.remote()
    sink_node = (
        Source("items")
        .chunk(sends_per_chunk * _LARGE_ROWS)
        .write_to(_slow_counting_writer(progress, 0.05))
    )
    pipeline = Pipeline.from_sink(sink_node, backend=backend)
    try:
        lead = 0
        for sent in range(1, n_sends + 1):
            pipeline.send(items=_make_set(np.arange(_LARGE_ROWS)))
            lead = max(lead, sent - ray.get(progress.get.remote()))
        pipeline.flush()
        return lead, pipeline.stats()
    finally:
        pipeline.close()


# Sends per chunk of the pipelines measuring how far the driver runs ahead.
_SENDS_PER_CHUNK = 2
# Items a stage holds at most with the default settings: its mailbox, the
# item being fetched, its inbox, the item being computed, its outbox, the
# item being emitted and its items in flight.
_STAGE_HOLDS = DEFAULT_BUFFER_SIZE + 2 * 2 + _DEFAULT_MAX_IN_FLIGHT + 3
# Sends the driver can run ahead: source, chunk and sink hold their items,
# the latter two (mostly) in chunks of several sends.
_MAX_LEAD = _STAGE_HOLDS * (1 + 2 * _SENDS_PER_CHUNK)


def test_send_waits_for_a_slow_stage_by_default():
    lead, stats = _lead_over_slow_sink(
        RayBackend(init_ray=False),
        n_sends=2 * _MAX_LEAD,
        sends_per_chunk=_SENDS_PER_CHUNK,
    )

    assert lead <= _MAX_LEAD
    assert all(s.buffered_peak <= DEFAULT_BUFFER_SIZE for s in stats.values())
    assert all(s.spilled == 0 for s in stats.values())


def test_unbounded_buffer_lets_send_run_ahead_of_a_slow_stage():
    lead, stats = _lead_over_slow_sink(
        RayBackend(init_ray=False, buffer_size=None),
        n_sends=2 * _MAX_LEAD,
        sends_per_chunk=_SENDS_PER_CHUNK,
    )

    assert lead > _MAX_LEAD
    # The backlog beyond the spill threshold waits on disk.
    assert _stage(stats, "sink(CallbackSinkWriter)").spilled > 0


def _run_or_fail(pipeline, sends, timeout=60.0):
    """Send every (input, item) and flush, failing if the pipeline deadlocks."""
    errors = []

    def run():
        try:
            for name, item in sends:
                pipeline.send(**{name: item})
            pipeline.flush()
        except Exception as error:
            errors.append(error)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=timeout)
    assert not runner.is_alive(), "The pipeline deadlocked."
    assert errors == []


def _tight_backend():
    """The smallest buffers: every bound is a single item."""
    return RayBackend(init_ray=False, queue_size=1, max_in_flight=1, buffer_size=1)


def _sum_ids(sample):
    return sum(_ids(sample))


def _describe_pair(pair):
    item, model = pair
    return (_ids(item), model)


def test_combine_latest_with_a_sample_of_its_own_input_does_not_deadlock():
    # The second input needs more items of the source than any buffer holds,
    # while the first input receives every item of the source as well.
    collector = Collector.remote()
    n_sample = 8
    items = Source("items")
    model = items.map(lambda s: s).collect(n_sample).map(_sum_ids)
    sink_node = (
        items.combine_latest(model)
        .map(_describe_pair)
        .write_to(_collecting_writer(collector))
    )

    with Pipeline.from_sink(sink_node, backend=_tight_backend()) as pipeline:
        _run_or_fail(pipeline, [("items", _make_set([i])) for i in range(20)])

    assert ray.get(collector.get.remote()) == [
        ([i], sum(range(n_sample))) for i in range(20)
    ]


def test_merge_of_a_branch_and_its_own_source_does_not_deadlock():
    collector = Collector.remote()

    def slow_identity(x):
        time.sleep(0.01)
        return x

    numbers = Source("numbers")
    merged = numbers.map(slow_identity).map(lambda x: x)
    numbers.op(merged)
    sink_node = merged.write_to(_collecting_writer(collector))

    with Pipeline.from_sink(sink_node, backend=_tight_backend()) as pipeline:
        _run_or_fail(pipeline, [("numbers", i) for i in range(20)])

    assert sorted(ray.get(collector.get.remote())) == sorted([*range(20)] * 2)


def test_items_overflowing_a_blocked_stage_are_spilled(tmp_path):
    collector = Collector.remote()
    gate = Gate.remote()
    n_items = 10

    source = Source("numbers")
    sink_node = source.map(lambda x: x).write_to(_gated_writer(gate, collector))
    backend = RayBackend(
        init_ray=False,
        spill_threshold=2,
        spill_store=pickle_spill_store(tmp_path),
    )
    pipeline = Pipeline.from_sink(sink_node, backend=backend)

    for i in range(n_items):
        pipeline.send(numbers=i)
    ray.get(gate.open.remote())
    pipeline.flush()

    assert ray.get(collector.get.remote()) == list(range(n_items))
    assert _stage(pipeline.stats(), "sink(CallbackSinkWriter)").spilled > 0
    assert list(tmp_path.rglob("*.pkl")) == []
    pipeline.close()


def test_close_removes_the_default_spill_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    collector = Collector.remote()
    gate = Gate.remote()

    source = Source("numbers")
    sink_node = source.map(lambda x: x).write_to(_gated_writer(gate, collector))
    backend = RayBackend(init_ray=False, spill_threshold=1)
    pipeline = Pipeline.from_sink(sink_node, backend=backend)

    for i in range(6):
        pipeline.send(numbers=i)
    ray.get(gate.open.remote())
    pipeline.flush()
    assert _stage(pipeline.stats(), "sink(CallbackSinkWriter)").spilled > 0
    assert [path.name[:21] for path in tmp_path.iterdir()] == ["scipion_bridge_spill_"]

    pipeline.close()

    assert list(tmp_path.iterdir()) == []


def _timed_identity(x):
    t_start = time.time()
    time.sleep(0.5)
    return t_start, time.time()


def _most_overlapping(intervals):
    return max(sum(start <= t < end for start, end in intervals) for t, _ in intervals)


def test_max_in_flight_bounds_concurrent_map_calls():
    overlaps = []
    for max_in_flight in (1, 4):
        collector = Collector.remote()
        sink_node = (
            Source("numbers")
            .map(_timed_identity)
            .write_to(_collecting_writer(collector))
        )
        backend = RayBackend(init_ray=False, max_in_flight=max_in_flight)
        with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
            for i in range(4):
                pipeline.send(numbers=i)
        overlaps.append(_most_overlapping(ray.get(collector.get.remote())))

    assert overlaps[0] == 1
    assert overlaps[1] > 1


def test_stage_error_propagates_to_driver():
    def fail_on_three(x):
        if x == 3:
            raise ValueError("boom on three")
        return x

    source = Source("numbers")
    sink_node = source.map(fail_on_three).write_to(CallbackSinkWriter(lambda x: None))

    backend = RayBackend(init_ray=False)
    pipeline = Pipeline.from_sink(sink_node, backend=backend)

    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        for i in range(6):
            pipeline.send(numbers=i)
        pipeline.flush()

    assert "boom on three" in str(exc_info.value)

    # The stage stays failed: a later flush raises instead of hanging.
    with pytest.raises(ray.exceptions.RayTaskError):
        pipeline.flush()


def _reject(s):
    raise ValueError("rejected sample")


def test_stage_error_fails_send_before_flush():
    # After collect emits its single item, the failed stage never receives
    # another item; the error must still reach the driver while it is sending.
    source = Source("items")
    sink_node = source.collect(2).map(_reject).write_to(CallbackSinkWriter(print))

    backend = RayBackend(init_ray=False)
    pipeline = Pipeline.from_sink(sink_node, backend=backend)

    sent = 0
    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        for i in range(500):
            pipeline.send(items=_make_set([i]))
            sent += 1
            time.sleep(0.01)

    assert "rejected sample" in str(exc_info.value)
    assert sent < 500


def test_exception_in_pipeline_block_is_not_hidden_by_flush():
    source = Source("items")
    sink_node = source.collect(1).map(_reject).write_to(CallbackSinkWriter(print))

    backend = RayBackend(init_ray=False)
    with pytest.raises(KeyError, match="driver error"):
        with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
            pipeline.send(items=_make_set([1]))
            raise KeyError("driver error")


def test_accumulator_with_rapid_sends_emits_exact_chunks():
    collector = Collector.remote()

    source = Source("items")
    sink_node = source.chunk(3).write_to(_collecting_writer(collector))

    backend = RayBackend(init_ray=False, queue_size=4)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        for i in range(0, 40, 2):
            pipeline.send(items=_make_set([i, i + 1]))

    chunks = [_ids(s) for s in ray.get(collector.get.remote())]
    assert chunks == [list(range(i, min(i + 3, 40))) for i in range(0, 40, 3)]


def test_stage_stats_count_items_per_stage():
    def identity(s):
        return s

    source = Source("items")
    sink_node = (
        source.chunk(4).map(identity).write_to(CallbackSinkWriter(lambda s: None))
    )

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        for i in range(0, 10, 2):
            pipeline.send(items=_make_set([i, i + 1]))

    stats = pipeline.stats()

    assert [label.split(":", 1)[1] for label in stats] == [
        "source(items)",
        "chunk(4)",
        f"map({identity.__qualname__})",
        "sink(CallbackSinkWriter)",
    ]
    assert [(s.items_in, s.items_out) for s in stats.values()] == [
        (5, 5),
        (5, 3),
        (3, 3),
        (3, 0),
    ]
    assert all(s.process_s >= 0.0 and s.idle_s >= 0.0 for s in stats.values())


def _run_timed(sink_node, n_items):
    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        t_start = time.perf_counter()
        for i in range(n_items):
            pipeline.send(numbers=i)
    return time.perf_counter() - t_start, pipeline.stats()


def test_split_maps_overlap_gpu_and_cpu_work():
    gpu_duration = 0.3
    cpu_duration = 0.2
    n_items = 6

    def forward(x):
        time.sleep(gpu_duration)  # Stands in for the GPU forward pass.
        return x * 10

    def build_metadata(y):
        time.sleep(cpu_duration)  # Stands in for CPU post-processing.
        return y + 1

    fused_collector = Collector.remote()
    fused = (
        Source("numbers")
        .map(lambda x: build_metadata(forward(x)))
        .write_to(_collecting_writer(fused_collector))
    )
    _run_timed(fused, n_items)

    split_collector = Collector.remote()
    split = (
        Source("numbers")
        .map(forward)
        .map(build_metadata)
        .write_to(_collecting_writer(split_collector))
    )
    split_elapsed, stats = _run_timed(split, n_items)

    expected = [i * 10 + 1 for i in range(n_items)]
    assert ray.get(fused_collector.get.remote()) == expected
    assert ray.get(split_collector.get.remote()) == expected

    # Both map stages are busy at the same time: the wall time is below the sum
    # of their processing times. (The fused map overlaps across the items in
    # flight just as well, so on warm workers both pipelines take as long.)
    forward_stats, metadata_stats = [s for label, s in stats.items() if "map(" in label]
    assert split_elapsed < forward_stats.process_s + metadata_stats.process_s
