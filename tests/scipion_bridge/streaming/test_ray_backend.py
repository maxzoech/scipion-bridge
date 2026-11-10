import time

import pytest
import ray
from scipion_bridge.core.streaming import Source
from scipion_bridge.core.streaming import CallbackSinkWriter
from scipion_bridge.core.streaming import Pipeline
from scipion_bridge.backend.ray import RayBackend

pytestmark = pytest.mark.usefixtures("ray_cluster")


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, x):
        self.items.append(x)

    def get(self):
        return self.items


def test_ray_basic_pipeline():
    collector = Collector.remote()
    writer = CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))

    source = Source("numbers")
    sink_node = source.map_batch(lambda x: x * 2).write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(numbers=5)
        pipeline.send(numbers=10)

    assert ray.get(collector.get.remote()) == [10, 20]


def test_ray_checkpoint_and_branching():
    ckpt_collector = Collector.remote()
    final_collector = Collector.remote()

    ckpt_writer = CallbackSinkWriter(lambda x: ray.get(ckpt_collector.append.remote(x)))
    final_writer = CallbackSinkWriter(
        lambda x: ray.get(final_collector.append.remote(x))
    )

    source = Source("input")
    stage1 = source.map_batch(lambda x: x + 10)
    stage1.checkpoint(ckpt_writer)
    stage2 = stage1.map_batch(lambda x: x * 2)
    final_sink = stage2.write_to(final_writer)

    # Note: checkpoint attaches a sink to stage1
    from scipion_bridge.core.streaming import Sink

    ckpt_sink = [n for n in stage1.downstream if isinstance(n, Sink)][0]

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(final_sink, ckpt_sink, backend=backend) as pipeline:
        pipeline.send(input=1)
        pipeline.send(input=2)

    assert ray.get(ckpt_collector.get.remote()) == [11, 12]
    assert ray.get(final_collector.get.remote()) == [22, 24]


def test_ray_unknown_source_raises():
    writer = CallbackSinkWriter(lambda x: None)
    source = Source("registered_source")
    sink_node = source.write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        with pytest.raises(KeyError, match="not registered"):
            pipeline.send(unregistered_source=123)


def _live_named_actors() -> set[str]:
    return set(ray.util.list_named_actors())


def _wait_until_terminated(names: set[str], timeout: float = 10.0) -> set[str]:
    """The actors among ``names`` still alive after ``timeout`` seconds.

    ``ray.kill`` returns before the cluster has removed the actor.
    """
    deadline = time.monotonic() + timeout
    while (alive := names & _live_named_actors()) and time.monotonic() < deadline:
        time.sleep(0.1)
    return alive


def test_leaving_the_context_terminates_the_actors():
    before = _live_named_actors()
    sink_node = (
        Source("numbers")
        .map_batch(lambda x: x * 2)
        .scan(lambda a, b: a + b)
        .sink(print)
    )

    with Pipeline.from_sink(sink_node, backend=RayBackend(init_ray=False)) as pipeline:
        pipeline.send(numbers=5)
        started = _live_named_actors() - before

    assert started
    assert _wait_until_terminated(started) == set()
