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
