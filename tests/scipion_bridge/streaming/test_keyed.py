"""Pipelines of a group_by shared by several keys (share_keys), in-process.

The functions of the shared pipeline are called directly; the Ray tests of
group_by run them across actors (see test_group_by.py).
"""

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.streaming.ir import (
    IRAccumulate,
    IRDemux,
    IRMap,
    IRSource,
    Keyed,
    Tagged,
)
from scipion_bridge.core.streaming.keyed import share_keys
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source


class Item(B.Struct):
    id: int


def _make_set(ids):
    s = B.Set[Item](capacity=len(ids))
    s["id"] = np.array(ids, dtype=np.int64).reshape(-1, 1)
    return s


def _ids(s):
    return [int(x) for x in s["id"].flatten()]


def _template(pipeline):
    """The shared copy of the template of ``Source.group_by(0, pipeline)``."""
    (sink,) = lower([Source("x").group_by(0, pipeline).unkey().sink(print)])
    (demux,) = sink.upstream
    assert isinstance(demux, IRDemux)
    assert demux.template is not None
    return demux.template, share_keys(demux.template)


def _nodes(exit_):
    nodes = [exit_]
    for node in nodes:
        nodes.extend(up for up in node.upstream if up not in nodes)
    return nodes[::-1]


def _accumulate(node, items):
    """Run an accumulator over ``items`` and flush it, as its stage does."""
    assert isinstance(node, IRAccumulate)
    state = node.initial_state_fn()
    emitted = []
    for item in items:
        state, emissions = node.accumulate_fn(state, item)
        emitted.extend(emissions)
    match node.flush_fn:
        case None:
            pass
        case flush_fn:
            state, emissions = flush_fn(state)
            emitted.extend(emissions)
    return emitted


def _by_key(results):
    grouped = {}
    for result in results:
        assert isinstance(result, Keyed)
        grouped.setdefault(result.key, []).append(result.value)
    return grouped


def test_shared_copy_keeps_the_shape_and_leaves_the_template_alone():
    template, shared = _template(lambda stream: stream.chunk(2).map(_ids))

    assert [type(node) for node in _nodes(shared)] == [IRSource, IRAccumulate, IRMap]
    assert all(
        copy is not original for copy, original in zip(_nodes(shared), _nodes(template))
    )
    assert isinstance(template, IRMap)
    assert template.func is _ids


def test_maps_keep_the_key():
    _, shared = _template(lambda stream: stream.map(len))

    assert isinstance(shared, IRMap)
    assert shared.func(Keyed("a", [1, 2])) == Keyed("a", 2)


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [
        (
            lambda stream: stream.chunk(2),
            {"a": [[1, 2], [3]], "b": [[10, 11]]},
        ),
        (
            lambda stream: stream.collect(2),
            {"a": [[1, 2]], "b": [[10, 11]]},
        ),
    ],
)
def test_accumulators_keep_a_state_per_key(pipeline, expected):
    _, shared = _template(pipeline)
    items = [("a", 1), ("b", 10), ("a", 2), ("b", 11), ("a", 3)]

    emitted = _accumulate(
        shared,
        [Keyed(key, _make_set([value])) for key, value in items],
    )

    assert {
        key: [_ids(s) for s in sets] for key, sets in _by_key(emitted).items()
    } == expected


def test_flatten_keeps_the_key_of_every_element():
    _, shared = _template(lambda stream: stream.flatten())

    emitted = _accumulate(shared, [Keyed("a", [1, 2]), Keyed("b", [3])])

    assert emitted == [Keyed("a", 1), Keyed("a", 2), Keyed("b", 3)]


def test_combine_latest_pairs_the_items_of_the_same_key():
    _, shared = _template(lambda stream: stream.combine_latest(stream.map(len)))
    combine = next(
        node
        for node in _nodes(shared)
        if isinstance(node, IRAccumulate) and node.tag_inputs
    )

    emitted = _accumulate(
        combine,
        [
            Tagged(0, Keyed("a", "a0")),
            Tagged(1, Keyed("b", "latest-b")),
            Tagged(0, Keyed("b", "b0")),
            Tagged(1, Keyed("a", "latest-a")),
        ],
    )

    assert emitted == [
        Keyed("b", ("b0", "latest-b")),
        Keyed("a", ("a0", "latest-a")),
    ]


@pytest.mark.parametrize(
    "pipeline",
    [
        lambda stream: stream.chunk(2),
        lambda stream: stream.collect(2),
        lambda stream: stream.combine_latest(stream.map(len)),
    ],
)
def test_flush_of_an_initial_state_emits_nothing(pipeline):
    # A shared pipeline only flushes the keys an accumulator has seen; a key
    # without items must not miss an emission.
    template, _ = _template(pipeline)
    accumulators = [node for node in _nodes(template) if isinstance(node, IRAccumulate)]

    assert accumulators
    for node in accumulators:
        assert node.flush_fn is not None
        _, emissions = node.flush_fn(node.initial_state_fn())
        assert emissions == []


def test_accumulators_without_flush_stay_without():
    _, shared = _template(lambda stream: stream.flatten())

    assert isinstance(shared, IRAccumulate)
    assert shared.flush_fn is None


def test_items_without_key_raise():
    _, shared = _template(lambda stream: stream.chunk(2))

    with pytest.raises(TypeError, match="has no key"):
        _accumulate(shared, [_make_set([1])])


def test_nested_group_by_receives_keyed_items():
    _, shared = _template(lambda stream: stream.group_by(0, lambda s: s).unkey())

    assert isinstance(shared, IRDemux)
    assert shared.outer_keyed
