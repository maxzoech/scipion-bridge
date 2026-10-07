"""Backend hooks for compiling and driving nested (child) pipelines."""

import time

import numpy as np
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.ir import (
    IRAccumulate,
    IRMap,
    IRSink,
    IRSource,
    clone_ir,
)
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


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


def _double(x):
    return 2 * x


def _template():
    """Lowered `x.map(_double)`, without a sink, as a group_by template."""
    (exit_ir,) = lower([Source("x").map(_double)])
    return exit_ir


def _walk(root):
    seen, stack, order = set(), [root], []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        stack.extend(node.upstream)
    return order


# -- clone_ir ----------------------------------------------------------------


def test_clone_copies_every_node_and_shares_functions():
    template = _template()

    (clone,) = clone_ir([template])

    assert isinstance(clone, IRMap)
    assert clone is not template
    assert clone.func is template.func
    assert clone.name == template.name
    (source,) = clone.upstream
    assert isinstance(source, IRSource)
    assert source is not template.upstream[0]
    assert source.name == "x"
    assert source.downstream == [clone]


def test_clone_keeps_upstream_order_of_multi_input_stages():
    sink_node = Source("left").combine_latest(Source("right")).sink(print)
    (sink_ir,) = lower([sink_node])

    (clone,) = clone_ir([sink_ir])

    (combine,) = clone.upstream
    assert isinstance(combine, IRAccumulate)
    assert combine.tag_inputs
    assert [up.name for up in combine.upstream] == ["left", "right"]


def test_clone_keeps_shared_upstream_nodes_shared():
    # The template of the protocol uses its input twice (collect and chunk).
    source = Source("x")
    trained = source.collect(3).map(_double)
    sink_node = source.chunk(2).combine_latest(trained).sink(print)
    (sink_ir,) = lower([sink_node])

    (clone,) = clone_ir([sink_ir])

    sources = [node for node in _walk(clone) if isinstance(node, IRSource)]
    assert len(sources) == 1
    assert len(sources[0].downstream) == 2
    assert len(_walk(clone)) == len(_walk(sink_ir))


def test_clone_leaves_template_untouched():
    template = _template()
    (clone,) = clone_ir([template])

    clone.add_downstream(IRSink(writer=CallbackSinkWriter(print)))

    assert template.downstream == []
    assert template.upstream[0].downstream == [template]


# -- compile -----------------------------------------------------------------


def test_clones_of_one_template_run_independently():
    template = _template()
    collectors = [Collector.remote(), Collector.remote()]
    backend = RayBackend(init_ray=False)

    pipelines = []
    for index, collector in enumerate(collectors):
        (clone,) = clone_ir([template])
        sink = IRSink(writer=_collecting_writer(collector))
        clone.add_downstream(sink)
        pipelines.append(backend.compile([sink], name=f"child[{index}]:"))

    try:
        pipelines[0].send("x", 1)
        pipelines[1].send("x", 10)
        pipelines[0].send("x", 2)
        for pipeline in pipelines:
            pipeline.flush()

        assert ray.get(collectors[0].get.remote()) == [2, 4]
        assert ray.get(collectors[1].get.remote()) == [20]
        assert template.downstream == []
    finally:
        for pipeline in pipelines:
            pipeline.close()


def test_compile_name_prefixes_stage_labels():
    (sink,) = lower([Source("x").map(_double).sink(print)])
    pipeline = RayBackend(init_ray=False).compile([sink], name="group_by[3]:")

    try:
        assert list(pipeline.stats()) == [
            "group_by[3]:0:source(x)",
            "group_by[3]:1:map(_double)",
            "group_by[3]:2:sink(CallbackSinkWriter)",
        ]
    finally:
        pipeline.close()


def test_source_handle_pushes_without_send():
    collector = Collector.remote()
    (sink,) = lower([Source("x").map(_double).write_to(_collecting_writer(collector))])
    pipeline = RayBackend(init_ray=False).compile([sink])

    try:
        ray.get(pipeline.source_handle("x").push.remote(21))
        pipeline.flush()

        assert ray.get(collector.get.remote()) == [42]
        with pytest.raises(KeyError, match="not registered"):
            pipeline.source_handle("y")
    finally:
        pipeline.close()


def _reject(s):
    raise ValueError("child failed")


def test_failing_child_stage_aborts_abort_targets():
    backend = RayBackend(init_ray=False)
    (parent_sink,) = lower([Source("p").map(_double).sink(print)])
    parent = backend.compile([parent_sink])

    # After collect, the failed child stage never receives another item, so
    # only the abort reaches the parent.
    (child_sink,) = lower([Source("c").collect(1).map(_reject).sink(print)])
    child = backend.compile(
        [child_sink],
        name="child:",
        abort_targets=[parent.source_handle("p")],
    )

    try:
        child.send("c", _make_set([1]))

        with pytest.raises(ray.exceptions.RayTaskError, match="child failed"):
            for i in range(500):
                parent.send("p", i)
                time.sleep(0.01)
    finally:
        child.close()
        parent.close()
