import numpy as np
import pytest
import ray
import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ops import Source, _flatten
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


@ray.remote
class ItemCollector:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def get(self):
        return self.items


@pytest.mark.parametrize("container", [list, tuple])
def test_flatten_emits_every_element(container):
    _, out = _flatten(None, container([1, (2, 3), "x"]))
    assert out == [1, (2, 3), "x"]


def test_flatten_of_empty_list_emits_nothing():
    _, out = _flatten(None, [])
    assert out == []


def test_flatten_rejects_non_sequences():
    with pytest.raises(TypeError, match="FlattenOp expected a list or tuple"):
        _flatten(None, _make_set([1]))


def _split_ids(s):
    return [(int(i), int(i) % 2) for i in s["id"].flatten()]


def test_flatten_in_ray_pipeline():
    collector = ItemCollector.remote()
    writer = CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))

    sink_node = Source("items").map(_split_ids).flatten().write_to(writer)

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(items=_make_set([1, 2, 3]))
        pipeline.send(items=_make_set([4]))

    assert ray.get(collector.get.remote()) == [(1, 1), (2, 0), (3, 1), (4, 0)]
