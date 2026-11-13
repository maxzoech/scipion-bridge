"""Image stacks as sources of the ``pixels`` of particles (internal).

The particle resolvers do not read the images of a stack: they make the
``pixels`` of the resolved Set reference rows of a ``StackSource``, read
when the column is first read (``core/struct/source.py``). The source only
uses the public API of the stack proxies.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ..core.struct.source import ColumnSource, FileId, RowIndex
from .mrc import check_stack
from .proxies import MRCStackProxy, SpiderStackProxy

StackProxy = Union[MRCStackProxy, SpiderStackProxy]


class StackSource(ColumnSource):
    """The images of an MRC or SPIDER stack, as float32.

    The rows, their bytes in the file and the file's identity are taken from
    the header when the source is created.
    """

    def __init__(self, stack: StackProxy) -> None:
        super().__init__()
        assert not stack.managed, "temporary stacks cannot be referenced"
        self._stack = stack
        self._rows = len(stack)
        self._nbytes_per_row = int(stack.estimated_item_nbytes or 0)
        self._files = (FileId.of(stack.path),)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def nbytes_per_row(self) -> int:
        return self._nbytes_per_row

    @property
    def files(self) -> Tuple[FileId, ...]:
        return self._files

    def describe(self) -> str:
        return self._stack.path.name

    def _read(self, index: RowIndex) -> np.ndarray:
        return read_rows(self._stack, index)


def read_rows(stack: StackProxy, index: RowIndex) -> np.ndarray:
    """The images ``index`` of ``stack``, read sequentially.

    The stack readers read contiguous slices, so the images of a strided
    range or an array of positions are selected from the slice covering them.
    """
    if isinstance(index, range) and index.step == 1:
        return stack.read_images(slice(index.start, index.stop))
    positions = np.asarray(index, dtype=np.int64)
    if len(positions) == 0:
        return stack.read_images(slice(0, 0))
    first = int(positions.min())
    covering = stack.read_images(slice(first, int(positions.max()) + 1))
    return covering[positions - first]


def check_rows(stack: StackProxy, index: RowIndex) -> None:
    """Raise the errors reading images ``index`` of ``stack`` would raise.

    Reads the header only: a reference to the images fails when it is
    created, as reading them eagerly did.
    """
    match stack:
        case MRCStackProxy():
            stop = int(np.max(np.asarray(index))) + 1 if len(index) else 0
            check_stack(stack.path, stop)
        case SpiderStackProxy():
            stack.read_images(slice(0, 0))
