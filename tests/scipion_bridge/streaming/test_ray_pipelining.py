"""Tests for pipelined (overlapping) stage execution in the Ray backend."""

import asyncio
import random
import threading
import time

import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


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


def test_backpressure_bounds_items_in_flight():
    collector = Collector.remote()
    gate = Gate.remote()
    n_items = 20

    def gated_write(x):
        ray.get(gate.wait.remote())
        ray.get(collector.append.remote(x))

    source = Source("numbers")
    sink_node = source.map(lambda x: x).write_to(CallbackSinkWriter(gated_write))

    backend = RayBackend(init_ray=False, queue_size=1)
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
        "accumulate",
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
    fused_elapsed, _ = _run_timed(fused, n_items)

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
    # of their processing times, and below the fused pipeline's wall time.
    forward_stats, metadata_stats = [s for label, s in stats.items() if "map(" in label]
    assert split_elapsed < forward_stats.process_s + metadata_stats.process_s
    assert split_elapsed < fused_elapsed - n_items * cpu_duration / 2
