from typing import Dict, Type

import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming import LoweringContext
from scipion_bridge.core.streaming.ir import IRAccumulate, IROp, Tagged
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import (
    Op,
    Source,
    _combine_latest,
    _combine_latest_flush,
    _combine_latest_initial_state,
)
from scipion_bridge.core.streaming.pipeline import Pipeline
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
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


def _collecting_writer(collector):
    return CallbackSinkWriter(lambda x: ray.get(collector.append.remote(x)))


# -- Accumulator -------------------------------------------------------------


def test_items_before_first_latest_are_buffered_and_released():
    state = _combine_latest_initial_state()

    state, out = _combine_latest(state, Tagged(port=0, item="a"))
    assert out == []
    state, out = _combine_latest(state, Tagged(port=0, item="b"))
    assert out == []

    state, out = _combine_latest(state, Tagged(port=1, item="m1"))
    assert out == [("a", "m1"), ("b", "m1")]

    state, out = _combine_latest(state, Tagged(port=0, item="c"))
    assert out == [("c", "m1")]


def test_latest_is_updated_without_emitting():
    state = _combine_latest_initial_state()
    state, _ = _combine_latest(state, Tagged(port=1, item="m1"))

    state, out = _combine_latest(state, Tagged(port=1, item="m2"))
    assert out == []

    state, out = _combine_latest(state, Tagged(port=0, item="a"))
    assert out == [("a", "m2")]


def test_none_is_a_valid_latest_value():
    state, _ = _combine_latest(_combine_latest_initial_state(), Tagged(1, None))
    _, out = _combine_latest(state, Tagged(port=0, item="a"))
    assert out == [("a", None)]


def test_flush_drops_unpaired_items():
    state, _ = _combine_latest(_combine_latest_initial_state(), Tagged(0, "a"))

    state, out = _combine_latest_flush(state)
    assert out == []
    assert state.pending == []


def test_unknown_port_raises():
    with pytest.raises(ValueError, match="CombineLatestOp has two inputs"):
        _combine_latest(_combine_latest_initial_state(), Tagged(port=2, item="a"))


# -- Graph and lowering ------------------------------------------------------


def test_combine_latest_with_itself_raises():
    source = Source("input")
    with pytest.raises(ValueError, match="two different streams"):
        source.combine_latest(source)


def test_combine_latest_lowering():
    left = Source("left")
    right = Source("right")
    sink_node = left.combine_latest(right).sink(lambda x: None)

    combine_ir = lower([sink_node])[0].upstream[0]
    assert isinstance(combine_ir, IRAccumulate)
    assert combine_ir.tag_inputs
    assert combine_ir.name == "combine_latest"
    # Ports are the positions of the upstream stages: left is 0, right is 1.
    assert [up.name for up in combine_ir.upstream] == ["left", "right"]


def test_same_stream_on_both_inputs_fails_lowering():
    # Two accesses to the same protocol input are one stream.
    sink_node = Source("x").combine_latest(Source("x")).sink(lambda x: None)

    with pytest.raises(ValueError, match="same stream on several inputs"):
        lower([sink_node])


# -- Ray pipeline ------------------------------------------------------------


def _train(s):
    return sum(_ids(s))


def _describe_pair(pair):
    chunk, model = pair
    return (_ids(chunk), model)


@pytest.mark.parametrize(
    ("sample_size", "model"),
    [
        (3, 1 + 2 + 3),
        # The stream is shorter than the sample: collect emits on flush.
        (100, sum(range(1, 11))),
    ],
)
def test_chunks_are_paired_with_model_trained_on_same_stream(sample_size, model):
    collector = Collector.remote()

    model_stream = Source("items").collect(sample_size).map(_train)
    sink_node = (
        Source("items")
        .chunk(4)
        .combine_latest(model_stream)
        .map(_describe_pair)
        .write_to(_collecting_writer(collector))
    )

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        for start in range(1, 11, 2):
            pipeline.send(items=_make_set([start, start + 1]))

    assert ray.get(collector.get.remote()) == [
        ([1, 2, 3, 4], model),
        ([5, 6, 7, 8], model),
        ([9, 10], model),
    ]


class _FlushMarker(Op):
    """Emits "flush" every time the stage is flushed."""

    def lower(self, ctx: LoweringContext) -> IROp:
        return IRAccumulate(
            accumulate_fn=lambda state, item: (state, [item]),
            initial_state_fn=lambda: None,
            flush_fn=lambda state: (state, ["flush"]),
            name="flush_marker",
        )


def test_stage_with_two_inputs_flushes_once():
    collector = Collector.remote()

    merged = Source("a").map(lambda x: x)
    Source("b").op(merged)
    sink_node = merged.op(_FlushMarker()).write_to(_collecting_writer(collector))

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(a=1)
        pipeline.send(b=2)
        pipeline.send(a=3)

    items = ray.get(collector.get.remote())
    assert sorted(items[:-1]) == [1, 2, 3]
    assert items[-1] == "flush"
    assert items.count("flush") == 1


def test_flush_of_failed_two_input_stage_raises():
    def fail(x):
        raise ValueError("left failed")

    sink_node = (
        Source("left").map(fail).combine_latest(Source("right")).sink(lambda x: None)
    )

    backend = RayBackend(init_ray=False)
    pipeline = Pipeline.from_sink(sink_node, backend=backend)
    with pytest.raises(ray.exceptions.RayTaskError, match="left failed"):
        pipeline.send(left=1)
        pipeline.flush()


# -- Chained protocols -------------------------------------------------------


class Produce(B.Protocol):
    particles: B.Input[B.Set[Item]] = B.Input()

    def outputs(self) -> Dict[str, Type]:
        return {"particles": B.Set[Item]}

    def _wrap(self, s):
        return {"particles": s}

    def steps(self) -> B.Op:
        return self.particles.map(self._wrap)


class Classify(B.Protocol):
    """Uses its input twice: to train on a sample and to classify every chunk."""

    particles: B.Input[B.Set[Item]] = B.Input()

    def outputs(self) -> Dict[str, Type]:
        return {"particles": B.Set[Item]}

    def _output(self, pair):
        chunk, _ = pair
        return {"particles": chunk}

    def steps(self) -> B.Op:
        model = self.particles.collect(3).map(_train)
        return self.particles.chunk(2).combine_latest(model).map(self._output)


def test_chained_protocol_feeds_both_inputs_of_combine_latest():
    collector = Collector.remote()

    chain = Produce() | Classify()
    sink_node = chain.get_pipeline().write_to(
        CallbackSinkWriter(
            lambda out: ray.get(collector.append.remote(_ids(out["particles"]))),
        ),
    )

    # The adapter selecting the output of Produce feeds both branches, and the
    # chunk branch stays on port 0 of combine_latest.
    combine_ir = next(
        node
        for node in _walk_ir(LoweringContext().lower_node(sink_node))
        if isinstance(node, IRAccumulate) and node.name == "combine_latest"
    )
    assert [up.name for up in combine_ir.upstream] == ["chunk(2)", "map(_train)"]

    backend = RayBackend(init_ray=False)
    with Pipeline.from_sink(sink_node, backend=backend) as pipeline:
        pipeline.send(particles=_make_set([1, 2, 3]))
        pipeline.send(particles=_make_set([4, 5]))

    assert ray.get(collector.get.remote()) == [[1, 2], [3, 4], [5]]


def _walk_ir(root):
    seen, stack, order = set(), [root], []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        stack.extend(node.upstream)
    return order
