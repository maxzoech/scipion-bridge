import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.environment.storage import NumPyStorageProvider
from scipion_bridge.backend.standalone.container import Container


class Particle(B.Struct):
    pixels: B.Array[np.float32] = B.Array(shape=(128, 128))
    voltage: float


def test_numpy_storage_provider():
    provider = NumPyStorageProvider()
    group = provider.create_group()

    arr = group.create_array("pixels", (128, 128), np.float32)
    assert arr.shape == (128, 128)
    assert arr.ndim == 2
    assert arr.dtype == np.float32

    data = np.random.randn(128, 128).astype(np.float32)
    arr[:] = data
    assert np.allclose(arr, data)


def test_struct_with_numpy_storage():
    p = Particle()
    data = np.random.randn(128, 128).astype(np.float32)
    p.pixels = data
    p.voltage = 300.0

    assert p.pixels.shape == (128, 128)
    assert np.allclose(p.pixels, data)
    assert p.voltage == 300.0


def test_container_storage_provider_injection():
    container = Container()
    container.wire(packages=["scipion_bridge"])

    p = Particle()
    p.voltage = 200.0
    assert p.voltage == 200.0


def test_slice_arrow_array():
    import pyarrow as pa
    from scipion_bridge.core.struct.utils.arrow_utils import slice_arrow_array
    from scipion_bridge.core.struct.key_path import KeyPath

    arr = pa.array([10, 20, 30, 40, 50])

    assert slice_arrow_array(arr, ()).to_pylist() == [10, 20, 30, 40, 50]
    assert slice_arrow_array(arr, None).to_pylist() == [10, 20, 30, 40, 50]
    assert slice_arrow_array(arr, 2).to_pylist() == [30]
    assert slice_arrow_array(arr, -1).to_pylist() == [50]
    assert slice_arrow_array(arr, slice(1, 4)).to_pylist() == [20, 30, 40]
    assert slice_arrow_array(arr, (slice(2, 5),)).to_pylist() == [30, 40, 50]

    kp = KeyPath().narrow_slice(slice(1, 3))
    assert slice_arrow_array(arr, kp).to_pylist() == [20, 30]

    with pytest.raises(IndexError):
        slice_arrow_array(arr, 10)


class RowSample(B.Struct):
    image: B.Array[np.float32] = B.Array(shape=(2, 2))
    embedding: B.Array[float] = B.Array(shape=(None,))
    score: float


def _row_samples(n: int) -> "B.Set[RowSample]":
    import awkward as ak

    samples = B.Set[RowSample](capacity=n)
    samples["embedding"] = ak.Array([np.arange(3, dtype=float) + i for i in range(n)])
    samples["score"] = np.arange(n, dtype=float).reshape(n, 1)
    return samples


def test_ragged_row_write_does_not_convert_column_to_python_lists(monkeypatch):
    import awkward as ak

    samples = _row_samples(4)

    def fail(*args, **kwargs):
        raise AssertionError("ak.to_list must not be used for per-row writes")

    monkeypatch.setattr(ak, "to_list", fail)
    samples[1].embedding = np.array([7.0])
    monkeypatch.undo()

    assert ak.to_list(samples["embedding"]) == [
        [0.0, 1.0, 2.0],
        [7.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ]


def test_ragged_row_writes_on_read_only_column():
    import pickle

    import awkward as ak

    samples = pickle.loads(pickle.dumps(_row_samples(3), protocol=5))

    samples[0].embedding = np.zeros(5)
    samples[2].embedding = np.ones(1)

    assert ak.to_list(samples["embedding"]) == [[0.0] * 5, [1.0, 2.0, 3.0], [1.0]]


def test_new_field_is_allocated_at_length_of_sibling_columns():
    samples = B.Set[RowSample]()
    samples["score"] = np.ones((5, 1))

    samples[2].image = np.full((2, 2), 3.0, dtype=np.float32)

    assert samples["image"].shape == (5, 2, 2)
    assert samples["image"][2, 0, 0] == 3.0


def test_struct_owning_storage_pickles_with_lock():
    import pickle

    sample = RowSample(score=2.5, image=np.ones((2, 2), dtype=np.float32))

    restored = pickle.loads(pickle.dumps(sample))

    assert restored.score == 2.5
    restored.score = 3.0  # The restored storage is usable (lock recreated).
    assert restored.score == 3.0


def test_struct_view_pickles_only_its_own_leaves():
    import pickle

    samples = _row_samples(500)
    element = samples[7]

    restored = pickle.loads(pickle.dumps(element))

    # Pickling the parent storage would be at least as large as the whole Set.
    assert len(pickle.dumps(element)) < len(pickle.dumps(samples)) / 10
    assert restored.score == 7.0
    assert np.array_equal(restored.embedding, np.arange(3, dtype=float) + 7)
    assert not restored.storage.is_view


def test_set_row_assignment_skips_uninitialized_fields_of_value():
    samples = _row_samples(3)

    samples[1] = RowSample(score=42.0)

    assert samples["score"].ravel().tolist() == [0.0, 42.0, 2.0]
    assert np.array_equal(samples[1].embedding, np.arange(3, dtype=float) + 1)
