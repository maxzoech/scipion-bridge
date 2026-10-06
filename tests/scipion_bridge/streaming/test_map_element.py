"""Tests for ``Op.map_element``: parallel in-place per-element processing."""

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
from scipion_bridge.core.streaming.element_mapper import (
    ElementMapConfig,
    make_element_mapper,
)
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


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


def _run(func, col, **config):
    accumulate, initial_state = make_element_mapper(func, ElementMapConfig(**config))
    state, emitted = accumulate(initial_state(), col)
    state.close()
    return emitted


@pytest.mark.parametrize(
    "make_col",
    [lambda: list(range(20)), lambda: np.arange(20)],
    ids=["list", "ndarray"],
)
def test_updates_collection_in_place_in_order(make_col):
    col = make_col()

    emitted = _run(_increment, col, workers=4)

    assert emitted == [col]
    assert emitted[0] is col
    assert list(col) == list(range(1, 21))


def test_updates_set_rows_in_place():
    samples = _make_samples(10)

    def double_score(sample):
        sample.score = sample.score * 2
        return sample

    _run(double_score, samples)

    assert np.array_equal(samples["score"].ravel(), np.arange(10) * 2.0)


def test_empty_collection_is_forwarded_without_pools():
    accumulate, initial_state = make_element_mapper(_increment, ElementMapConfig())

    state, emitted = accumulate(initial_state(), [])

    assert emitted == [[]]
    assert state.threads is None


def test_first_exception_propagates():
    def fail_on_three(x):
        if x == 3:
            raise ValueError("bad element 3")
        return x

    with pytest.raises(ValueError, match="bad element 3"):
        _run(fail_on_three, list(range(8)), workers=2)


def test_auto_workers_follow_collection_length_capped_at_cpus(monkeypatch):
    monkeypatch.setattr(element_mapper, "_available_cpus", lambda: 4)
    accumulate, initial_state = make_element_mapper(_increment, ElementMapConfig())

    state = initial_state()
    assert state.size == 0

    state, _ = accumulate(state, list(range(3)))
    assert state.size == 3

    state, _ = accumulate(state, list(range(10)))
    assert state.size == 4

    pools = state
    state, _ = accumulate(state, list(range(2)))
    assert state is pools
    state.close()


def test_fixed_workers_create_pools_eagerly():
    accumulate, initial_state = make_element_mapper(
        _increment, ElementMapConfig(workers=2)
    )

    state = initial_state()

    assert state.size == 2
    assert state.threads is not None
    state.close()


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


def test_mapper_functions_pickle_without_pools():
    accumulate, initial_state = make_element_mapper(
        _increment, ElementMapConfig(workers=2)
    )

    restored_accumulate = cloudpickle.loads(cloudpickle.dumps(accumulate))
    restored_initial_state = cloudpickle.loads(cloudpickle.dumps(initial_state))

    col = [1, 2]
    state, _ = restored_accumulate(restored_initial_state(), col)
    assert col == [2, 3]
    state.close()


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
