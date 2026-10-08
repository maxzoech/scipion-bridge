import pytest
import numpy as np
import ray
import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter

pytestmark = pytest.mark.usefixtures("ray_cluster")


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


@ray.remote
class IdCollector:
    def __init__(self):
        self.ids = []

    def append(self, s):
        self.ids.extend(int(x) for x in s["id"].flatten())

    def get(self):
        return self.ids


def _writer(collector):
    return CallbackSinkWriter(lambda s: ray.get(collector.append.remote(s)))


def test_repeated_input_feeds_every_branch_with_every_item():
    first, second = IdCollector.remote(), IdCollector.remote()

    # Two Source nodes with the same name, like two accesses to self.particles
    sink_a = Source("items").chunk(4).write_to(_writer(first))
    sink_b = Source("items").chunk(3).write_to(_writer(second))

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_a, sink_b, backend=backend) as pipeline:
        pipeline.send(items=_make_set(list(range(0, 5))))
        pipeline.send(items=_make_set(list(range(5, 10))))

    assert ray.get(first.get.remote()) == list(range(10))
    assert ray.get(second.get.remote()) == list(range(10))


def test_actors_are_named_and_pipelines_do_not_clash():
    collector = IdCollector.remote()

    def build():
        return Source("items").map_batch(lambda s: s).write_to(_writer(collector))

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(build(), backend=backend) as p1:
        with Pipeline.from_sink(build(), backend=backend) as p2:
            p1.send(items=_make_set([1]))
            p2.send(items=_make_set([2]))

            labels = list(p1.stats())
            assert labels == [
                "0:source(items)",
                "1:map(test_actors_are_named_and_pipelines_do_not_clash.<locals>."
                "build.<locals>.<lambda>)",
                "2:sink(CallbackSinkWriter)",
            ]

    assert sorted(ray.get(collector.get.remote())) == [1, 2]
