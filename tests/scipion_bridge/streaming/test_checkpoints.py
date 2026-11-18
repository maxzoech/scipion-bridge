"""Checkpoints and taps: what the run store keeps, and when the backend
leaves them out (in-process). The Ray tests are in test_ray_checkpoints.py."""

import numpy as np

import scipion_bridge as B
from scipion_bridge.backend.ray import RayBackend
from scipion_bridge.core.streaming import (
    CheckpointInfo,
    Commit,
    FileRunStore,
    IRCheckpoint,
    IRMap,
)

A, C = (bytes([k]) * 16 for k in (1, 3))


class Value(B.Struct):
    value: float


def test_a_checkpoint_keeps_items_and_is_complete_once_marked(tmp_path):
    store = FileRunStore(tmp_path)
    label = "2:group_by[3]/4:checkpoint(classes)"
    values = B.Set[Value](capacity=2)
    values["value"] = np.array([1.0, 2.0])

    store.begin_checkpoint(label, replay=True)
    store.begin_checkpoint("5:tap", replay=False)
    store.store_item(label, A, values)
    store.append_commits([Commit(A, label, 2)])

    assert store.read_checkpoints() == {
        label: CheckpointInfo(label, True, 1, False),
        "5:tap": CheckpointInfo("5:tap", False, 0, False),
    }
    assert list(store.read_item(label, A)["value"]) == [1.0, 2.0]
    store.complete_checkpoint(label, repeated=4)
    assert store.read_checkpoints()[label] == CheckpointInfo(label, True, 1, True, 4)


def test_checkpoints_are_left_out_without_a_run_store(tmp_path):
    output = IRCheckpoint(protocol_output=True)
    user = IRCheckpoint(name="embeddings")
    with_store = RayBackend(init_ray=False, run_store=FileRunStore(tmp_path))
    without_outputs = RayBackend(
        init_ray=False,
        run_store=FileRunStore(tmp_path),
        checkpoint_outputs=False,
    )

    assert RayBackend(init_ray=False)._leaves_out(user)
    assert not with_store._leaves_out(user)
    assert not with_store._leaves_out(output)
    assert without_outputs._leaves_out(output)
    assert not without_outputs._leaves_out(user)
    assert not RayBackend(init_ray=False)._leaves_out(IRMap())
