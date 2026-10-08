"""Tests for ``Op.map_element``: parallel in-place per-element processing."""

import os
import sys
import threading

import awkward as ak
import cloudpickle
import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming import element_mapper
from scipion_bridge.core.environment.compute import CPUS_ENV_VAR
from scipion_bridge.core.streaming.element_mapper import (
    ElementMapConfig,
    ElementMapper,
)
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter

pytestmark = pytest.mark.usefixtures("ray_cluster")


class Sample(B.Struct):
    image: B.Array[np.float32] = B.Array(shape=(2, 2))
    embedding: B.Array[float] = B.Array(shape=(None,))
    score: float


def _make_samples(n: int) -> B.Set[Sample]:
    samples = B.Set[Sample](capacity=n)
    samples["image"] = np.zeros((n, 2, 2), dtype=np.float32)
    samples["embedding"] = ak.Array([np.zeros(1) for _ in range(n)])
    samples["score"] = np.arange(n, dtype=float).reshape(n, 1)
    return samples


def _increment(x):
    return x + 1


def _close(mapper):
    element_mapper._pools.pop(mapper.stage).close()


def _run(func, col, **config):
    mapper = ElementMapper(func, ElementMapConfig(**config))
    result = mapper(col)
    _close(mapper)
    return result


@pytest.mark.parametrize(
    "make_col",
    [lambda: list(range(20)), lambda: np.arange(20)],
    ids=["list", "ndarray"],
)
def test_updates_collection_in_place_in_order(make_col):
    col = make_col()

    result = _run(_increment, col, workers=4)

    assert result is col
    assert list(col) == list(range(1, 21))


def test_updates_set_rows_in_place():
    samples = _make_samples(10)

    def double_score(sample):
        sample.score = sample.score * 2
        return sample

    _run(double_score, samples)

    assert np.array_equal(samples["score"].ravel(), np.arange(10) * 2.0)


def test_empty_collection_is_forwarded_without_pools():
    mapper = ElementMapper(_increment, ElementMapConfig())

    assert mapper([]) == []
    assert mapper.stage not in element_mapper._pools


def test_first_exception_propagates():
    def fail_on_three(x):
        if x == 3:
            raise ValueError("bad element 3")
        return x

    with pytest.raises(ValueError, match="bad element 3"):
        _run(fail_on_three, list(range(8)), workers=2)


def test_auto_workers_follow_collection_length_capped_at_cpus(monkeypatch):
    monkeypatch.setattr(element_mapper, "_available_cpus", lambda: 4)
    mapper = ElementMapper(_increment, ElementMapConfig())

    mapper(list(range(3)))
    assert element_mapper._pools[mapper.stage].size == 3

    mapper(list(range(10)))
    assert element_mapper._pools[mapper.stage].size == 4

    pools = element_mapper._pools[mapper.stage]
    mapper(list(range(2)))
    assert element_mapper._pools[mapper.stage] is pools
    _close(mapper)


def test_available_cpus_are_all_cores_without_reservation(monkeypatch):
    monkeypatch.delenv(CPUS_ENV_VAR, raising=False)

    assert element_mapper._available_cpus() == len(os.sched_getaffinity(0))


def test_available_cpus_are_the_reserved_cores(monkeypatch):
    monkeypatch.setenv(CPUS_ENV_VAR, "3")

    assert element_mapper._available_cpus() == 3


def test_fixed_workers_do_not_depend_on_the_collection_length():
    mapper = ElementMapper(_increment, ElementMapConfig(workers=2))

    mapper([1, 2, 3, 4, 5])

    assert element_mapper._pools[mapper.stage].size == 2
    _close(mapper)


def test_pickled_mapper_reuses_the_pools_of_the_process():
    # Run as a Ray task, the mapper is pickled with every call.
    mapper = ElementMapper(_increment, ElementMapConfig())
    mapper([1, 2])
    pools = element_mapper._pools[mapper.stage]

    restored = cloudpickle.loads(cloudpickle.dumps(mapper))
    col = [1, 2]
    restored(col)

    assert col == [2, 3]
    assert element_mapper._pools[restored.stage] is pools
    _close(mapper)


@pytest.mark.parametrize(
    "config",
    [
        {"workers": 0},
        {"workers": -1},
        {"workers": True},
        {"workers": "many"},
        {"executor": "gpu"},
        {"start_method": "spawn"},
        {"executor": "process", "start_method": "clone"},
        {"chunksize": 0},
    ],
)
def test_invalid_config_raises(config):
    with pytest.raises(ValueError):
        ElementMapConfig(**config)


@pytest.fixture
def frequent_thread_switches():
    """Switch threads every microsecond so that races in short code paths show up."""
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


@pytest.mark.parametrize("repetition", range(20))
def test_concurrent_row_writes_on_read_only_set(repetition, frequent_thread_switches):
    workers = 8
    samples = ray.get(ray.put(_make_samples(200)))
    assert not samples["score"].flags.writeable

    barrier = threading.Barrier(workers)
    first_call = threading.local()

    def process(sample):
        # Every thread's first element waits until all threads arrived, so the
        # column transitions (copy-on-write, row splitting) happen concurrently.
        if not getattr(first_call, "done", False):
            first_call.done = True
            barrier.wait(timeout=10)

        i = int(sample.score)
        match i % 2:
            case 0:
                sample.score = 10.0 * i
                sample.embedding = np.full(i % 3 + 1, float(i))
                sample.image = np.full((2, 2), i, dtype=np.float32)
                return sample
            case _:
                return Sample(
                    score=10.0 * i,
                    embedding=np.full(i % 3 + 1, float(i)),
                    image=np.full((2, 2), i, dtype=np.float32),
                )

    _run(process, samples, workers=workers, chunksize=1)

    assert np.array_equal(samples["score"].ravel(), np.arange(200) * 10.0)
    assert np.array_equal(samples["image"][:, 0, 0], np.arange(200, dtype=np.float32))
    assert ak.to_list(samples["embedding"]) == [
        [float(i)] * (i % 3 + 1) for i in range(200)
    ]


@pytest.mark.parametrize(
    "func",
    [_increment, lambda x: x + 1],
    ids=["module_function", "lambda"],
)
def test_process_executor_matches_thread_executor(func):
    col = list(range(12))

    _run(func, col, workers=2, executor="process")

    assert col == list(range(1, 13))


def test_process_executor_on_set_elements():
    samples = _make_samples(6)

    _run(_double_score, samples, workers=2, executor="process", start_method="spawn")

    assert np.array_equal(samples["score"].ravel(), np.arange(6) * 2.0)


def _double_score(sample):
    sample.score = sample.score * 2
    return sample


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, x):
        self.items.append(x)

    def get(self):
        return self.items


def test_map_element_stage_in_ray_pipeline():
    collector = Collector.remote()

    def total_score(samples):
        return float(np.sum(samples["score"]))

    source = Source("samples")
    sink_node = (
        source.chunk(4)
        .map_element(_double_score)
        .map(total_score)
        .write_to(CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x))))
    )

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        samples = _make_samples(8)
        pipeline.send(samples=samples[:4])
        pipeline.send(samples=samples[4:])

    assert ray.get(collector.get.remote()) == [
        2.0 * (0 + 1 + 2 + 3),
        2.0 * (4 + 5 + 6 + 7),
    ]
    labels = [label.split(":", 1)[1] for label in pipeline.stats()]
    assert labels[1:3] == ["chunk(4)", f"map_element({_double_score.__qualname__})"]
