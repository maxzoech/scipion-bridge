"""Reading and writing image stacks in MRC files."""

import io
import os
from pathlib import Path
from typing import Optional

import mrcfile
import mrcfile.utils
import numpy as np
from mrcfile.constants import IMAGE_STACK_SPACEGROUP


def read_stack(path: Path, selection: Optional[slice] = None) -> np.ndarray:
    """Read images (N, H, W) as float32 from an MRC image stack or single image.

    Each run of consecutive images is read with one sequential read. Copying
    from a memory map instead faults the pages in one by one, which reads large
    stacks from spinning disks at a fraction of their sequential speed.

    Args:
        path: The MRC file.
        selection: The images to read. Defaults to all of them.

    Raises:
        ValueError: If the file holds a volume or a stack of volumes.
        EOFError: If the file ends before the images its header declares.
    """
    with mrcfile.open(path, header_only=True) as mrc:
        header = mrc.header
    assert header is not None

    nz, ny, nx = int(header.nz), int(header.ny), int(header.nx)
    if nz > 1 and int(header.ispg) != IMAGE_STACK_SPACEGROUP:
        raise ValueError(
            f"Expected an image stack (space group {IMAGE_STACK_SPACEGROUP}) or a "
            f"single image in MRC file '{path}', got space group {int(header.ispg)} "
            f"with data shape {mrcfile.utils.data_shape_from_header(header)}, "
            f"i.e. a volume or a stack of volumes.",
        )

    dtype = mrcfile.utils.data_dtype_from_header(header)
    data_offset = header.nbytes + int(header.nsymbt)
    image_nbytes = ny * nx * dtype.itemsize
    indices = range(nz)[selection if selection is not None else slice(None)]

    images = np.empty((len(indices), ny, nx), dtype=dtype)
    with open(path, "rb", buffering=0) as f:
        for position, first, count in _runs(indices):
            _read_exact(
                f,
                data_offset + first * image_nbytes,
                images[position : position + count].view(np.uint8).data,
                path,
            )
    return np.asarray(images, dtype=np.float32)


def write_stack(
    path: str | os.PathLike,
    images: np.ndarray,
    *,
    overwrite: bool,
) -> None:
    """Write images (N, H, W) to an MRC file marked as an image stack.

    `mrcfile.write` marks any three-dimensional array as a volume, which
    `read_stack` rejects.
    """
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(images)
        mrc.set_image_stack()


def _runs(indices: range) -> list[tuple[int, int, int]]:
    """Split indices into runs of consecutive images: (position, first, count)."""
    match indices.step:
        case 1:
            return [(0, indices.start, len(indices))] if len(indices) > 0 else []
        case _:
            return [(position, index, 1) for position, index in enumerate(indices)]


def _read_exact(f: io.RawIOBase, offset: int, buffer: memoryview, path: Path) -> None:
    f.seek(offset)
    view = buffer.cast("B")
    filled = 0
    # A single read returns at most ~2 GiB on Linux
    while filled < len(view):
        count = f.readinto(view[filled:])
        if not count:
            raise EOFError(f"MRC file '{path}' ends before the images it declares.")
        filled += count
