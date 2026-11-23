"""The ``astype`` operation (in-process): validation, lowering and fusion.

The cast itself is tested in ``struct/test_cast.py``; its Ray integration in
``test_ray_compute.py``.
"""

import numpy as np
import pytest

import scipion_bridge as B
from scipion_bridge.core.protocol.fingerprint import _stage_label
from scipion_bridge.core.streaming import (
    IRAccumulate,
    IRMap,
    NoFusion,
    Source,
    lower,
    report_of,
)
from scipion_bridge.core.streaming.ops import AsTypeOp, Cast
from scipion_bridge.core.struct import references


class Image(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(2, 2))
    defocus: float


class ScoredImage(Image):
    score: float


def _images(ids):
    s = B.Set[Image](capacity=len(ids))
    s["defocus"] = np.asarray(ids, dtype=np.float64)
    s["pixels"] = np.zeros((len(ids), 2, 2), dtype=np.float32)
    return s


def _discard(item):
    pass


def _scored(s):
    s["score"] = s["defocus"] * 2
    return s


# -- Validation ----------------------------------------------------------------
@pytest.mark.parametrize("target", [42, B.Set, int, np.float32])
def test_the_target_is_checked_when_the_pipeline_is_built(target):
    with pytest.raises(TypeError, match="Set type"):
        Source("items").astype(target)


def test_collections_are_not_implemented():
    with pytest.raises(NotImplementedError):
        Source("items").astype(B.Collection[Image])


def test_items_other_than_sets_and_structs_are_rejected():
    with pytest.raises(TypeError, match=r"astype\(Set\[ScoredImage\]\) expected a Set"):
        Cast(B.Set[ScoredImage])({"items": _images([1])})


# -- Lowering and fusion -------------------------------------------------------
def test_astype_lowers_to_a_fusable_map_without_resources():
    sink_node = Source("items").astype(B.Set[ScoredImage]).sink(_discard)

    (sink,) = lower([sink_node], fusion=NoFusion())

    (astype,) = sink.upstream
    assert isinstance(astype, IRMap)
    assert astype.fusable and astype.compute is None
    assert astype.name == "astype(Set[ScoredImage])"


def test_astype_after_chunk_runs_inside_the_chunk_stage():
    sink_node = Source("items").chunk(4).astype(B.Set[ScoredImage]).sink(_discard)

    (sink,) = lower([sink_node])

    (chunk,) = sink.upstream
    assert isinstance(chunk, IRAccumulate) and chunk.flush_fn is not None
    assert chunk.name == "chunk(4) → astype(Set[ScoredImage])"
    state = chunk.initial_state_fn()
    state, emitted = chunk.accumulate_fn(state, _images([0, 1, 2]))
    state, more = chunk.accumulate_fn(state, _images([3, 4]))
    _, flushed = chunk.flush_fn(state)
    batches = emitted + more + flushed
    assert [type(b) for b in batches] == [B.Set[ScoredImage]] * 2
    assert [len(b) for b in batches] == [4, 1]
    assert np.asarray(batches[0]["defocus"]).ravel().tolist() == [0, 1, 2, 3]


def test_astype_after_a_source_runs_inside_the_next_map():
    sink_node = Source("items").astype(B.Set[ScoredImage]).map(_scored).sink(_discard)

    report = report_of(lower([sink_node]))

    assert [n.title for n in report.nodes] == [
        "source(items)",
        "astype(Set[ScoredImage]) → map(_scored)",
        "sink(CallbackSinkWriter)",
    ]


def test_the_fused_map_writes_new_fields_into_the_cast_item():
    sink_node = Source("items").astype(B.Set[ScoredImage]).map(_scored).sink(_discard)
    (sink,) = lower([sink_node])
    (fused,) = sink.upstream
    images = _images([1, 2])

    scored = fused.func(images)

    assert np.asarray(scored["score"]).ravel().tolist() == [2.0, 4.0]
    assert ("score",) not in references.column_states(images)


# -- Protocols -----------------------------------------------------------------
@B.resources(gpus=1)
class Typed(B.Protocol):
    items: B.Input[B.Set[Image]] = B.Input()

    @classmethod
    def outputs(cls):
        return {"result": B.Set[ScoredImage]}

    def steps(self):
        return self.items.astype(B.Set[ScoredImage]).map(_scored)


def test_protocols_assign_no_resources_to_astype():
    steps = Typed()._tagged_steps()

    (astype,) = steps.upstream
    assert isinstance(astype, AsTypeOp)
    assert not hasattr(astype, "compute")
    assert steps.compute is not None


def test_the_stage_fingerprint_names_the_target():
    assert _stage_label(AsTypeOp(B.Set[ScoredImage])) == "AsTypeOp(Set[ScoredImage])"
    assert _stage_label(AsTypeOp(B.Set[Image])) != _stage_label(
        AsTypeOp(B.Set[ScoredImage])
    )
