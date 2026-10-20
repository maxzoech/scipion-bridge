from pathlib import Path

import mrcfile
import mrcfile.utils
import numpy as np
import pytest

from scipion_bridge.single_particle.mrc import read_stack, write_stack

N_IMAGES = 10


def _write_image_stack(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path) as mrc:
        mrc.set_data(data)
        mrc.set_image_stack()


def _expected(path: Path) -> np.ndarray:
    with mrcfile.open(path) as mrc:
        assert mrc.data is not None
        return np.asarray(mrc.data, dtype=np.float32)


@pytest.fixture(
    params=["<f4", ">f4", "<i2", ">i2", "<f2", ">f2"],
)
def stack_path(request, tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    values = rng.normal(scale=100.0, size=(N_IMAGES, 6, 5))
    path = tmp_path / "stack.mrcs"
    _write_image_stack(path, values.astype(request.param))
    with mrcfile.open(path, header_only=True) as mrc:
        assert mrc.header is not None
        assert mrcfile.utils.data_dtype_from_header(mrc.header) == np.dtype(
            request.param,
        )
    return path


@pytest.mark.parametrize(
    "selection",
    [
        None,
        slice(0, 4),  # first chunk
        slice(4, 8),  # middle chunk
        slice(8, 10),  # last chunk
        slice(8, 12),  # chunk crossing the end
        slice(12, 16),  # chunk past the end
        slice(None, None, 2),
        slice(1, 7, 3),
        slice(None, None, -1),
        slice(-3, None),
    ],
)
def test_read_stack_matches_mrcfile(stack_path: Path, selection):
    expected = _expected(stack_path)
    if selection is not None:
        expected = expected[selection]

    images = read_stack(stack_path, selection)

    assert images.dtype == np.float32
    assert images.shape == (len(expected), 6, 5)
    assert np.array_equal(images, expected)


def test_read_stack_single_image_file(tmp_path: Path):
    path = tmp_path / "image.mrc"
    data = np.arange(30, dtype=np.float32).reshape(6, 5)
    mrcfile.write(path, data)

    images = read_stack(path)

    assert images.shape == (1, 6, 5)
    assert np.array_equal(images[0], data)


def test_read_stack_rejects_volume(tmp_path: Path):
    path = tmp_path / "volume.mrc"
    mrcfile.write(path, np.zeros((4, 6, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="space group 1 "):
        read_stack(path, slice(0, 2))


def test_read_stack_rejects_volume_stack(tmp_path: Path):
    path = tmp_path / "volumes.mrcs"
    mrcfile.write(path, np.zeros((2, 4, 6, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="space group 401 "):
        read_stack(path)


def test_read_stack_truncated_file(tmp_path: Path):
    path = tmp_path / "stack.mrcs"
    _write_image_stack(path, np.ones((N_IMAGES, 6, 5), dtype=np.float32))
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size - 4)

    with pytest.raises(EOFError):
        read_stack(path, slice(8, 10))


def test_write_stack_marks_image_stack(tmp_path: Path):
    path = tmp_path / "stack.mrcs"
    data = np.arange(N_IMAGES * 30, dtype=np.float32).reshape(N_IMAGES, 6, 5)

    write_stack(path, data, overwrite=False)

    with mrcfile.open(path, header_only=True) as mrc:
        assert mrc.header is not None
        assert int(mrc.header.ispg) == 0
    assert np.array_equal(read_stack(path), data)
