"""Columns whose values are rows of a source, such as a file of images.

A ``Reference`` records that a column of a ``RootEngine`` holds rows of a
``ColumnSource``, e.g. images of an MRC stack, instead of owning its bytes.
The storage reads the rows when the column is first read, and a write turns
the column into an owned one (see ``storage.py``). Code outside the library
does not see references: they are internal to the type system, and a Set
with references behaves exactly like one holding the same values.

Every reference carries a digest of each of its rows (``row_digests``),
taken when its rows are first read. Rereading the rows, e.g. after a spill
dropped them, compares the digests, so changed inputs are detected exactly
instead of silently read.
"""

from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import xxhash
from numpy.typing import NDArray

# The rows of a source a reference holds: a range, or absolute positions.
RowIndex = Union[range, NDArray[np.int64]]

# Bytes of a row digest (xxh3, 128 bits).
DIGEST_BYTES = 16


class StaleReferenceError(RuntimeError):
    """The rows of a source differ from the rows a reference recorded."""


@dataclass(frozen=True)
class FileId:
    """A file as it was when a source was created: its size and mtime."""

    path: Path
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, path: Path) -> FileId:
        stat = os.stat(path)
        return cls(Path(path), stat.st_size, stat.st_mtime_ns)

    def matches(self) -> bool:
        """Whether the file is readable here, unchanged since this id was taken."""
        try:
            return FileId.of(self.path) == self
        except FileNotFoundError:
            return False

    def check(self) -> None:
        """Raise if the file changed since this id was taken.

        Raises:
            StaleReferenceError: If the file is gone, or its size or mtime
                changed.
        """
        try:
            current = FileId.of(self.path)
        except FileNotFoundError as error:
            raise StaleReferenceError(
                f"The input file {self.path} was removed while its rows were "
                "referenced by items of the pipeline.",
            ) from error
        if current != self:
            raise StaleReferenceError(
                f"The input file {self.path} changed while its rows were "
                "referenced by items of the pipeline (size or modification "
                "time differ). Inputs must not change during a run.",
            )


@dataclass(eq=False)
class SourceStats:
    """What a source read in this process, for the cost model and reports."""

    read_bytes: int = 0
    read_s: float = 0.0


class ColumnSource(abc.ABC):
    """Rows of values a column can reference, e.g. the images of a stack.

    A source describes its data and reads it; it never decides when to read
    (see ``ReferencePolicy`` in ``references.py``). Sources must be
    deterministic: the same files and rows give the same values. Digests
    enforce it.

    Attributes:
        stats: Bytes read and time spent reading in this process.
    """

    def __init__(self) -> None:
        self.stats = SourceStats()

    @property
    @abc.abstractmethod
    def rows(self) -> int:
        """The rows of the source."""

    @property
    @abc.abstractmethod
    def nbytes_per_row(self) -> int:
        """The bytes of one row once read."""

    @property
    def files(self) -> Tuple[FileId, ...]:
        """The files the rows are read from, as they were when referenced."""
        return ()

    @property
    def domain(self) -> str:
        """Where the files can be read (see ``core/environment/storage.py``)."""
        return ""

    def describe(self) -> str:
        """A short description for reports, e.g. the file name."""
        return type(self).__name__

    def read(self, index: RowIndex) -> np.ndarray:
        """Read rows ``index`` of the source, after checking its files.

        Raises:
            StaleReferenceError: If a file changed since it was referenced.
        """
        for file in self.files:
            file.check()
        start = time.perf_counter()
        rows = self._read(index)
        self.stats.read_s += time.perf_counter() - start
        self.stats.read_bytes += rows.nbytes
        return rows

    @abc.abstractmethod
    def _read(self, index: RowIndex) -> np.ndarray:
        """Read rows ``index``, in order."""

    def contiguous(self, index: RowIndex) -> bool:
        """Whether reading ``index`` reads one sequential run of the source."""
        if isinstance(index, range):
            return index.step == 1
        return bool(len(index) == 0 or np.all(np.diff(index) == 1))

    def __getstate__(self) -> dict:
        # Statistics belong to the process that read the rows.
        state = self.__dict__.copy()
        state["stats"] = SourceStats()
        return state


class ArraySource(ColumnSource):
    """Rows of an array in memory; for tests, and as the simplest source."""

    def __init__(self, array: np.ndarray, files: Tuple[FileId, ...] = ()) -> None:
        super().__init__()
        self._array = array
        self._files = files

    @property
    def rows(self) -> int:
        return len(self._array)

    @property
    def nbytes_per_row(self) -> int:
        return int(self._array[0].nbytes) if len(self._array) else 0

    @property
    def files(self) -> Tuple[FileId, ...]:
        return self._files

    def _read(self, index: RowIndex) -> np.ndarray:
        return self._array[_as_selector(index)].copy()


def _as_selector(index: RowIndex) -> Union[slice, NDArray[np.int64]]:
    if isinstance(index, range):
        return slice(index.start, index.stop, index.step)
    return index


def row_digests(rows: np.ndarray) -> NDArray[np.uint8]:
    """The xxh3 digest of the bytes of every row, as a ``(rows, 16)`` array."""
    contiguous = np.ascontiguousarray(rows)
    digests = np.empty((len(contiguous), DIGEST_BYTES), dtype=np.uint8)
    for i, row in enumerate(contiguous):
        digests[i] = np.frombuffer(xxhash.xxh3_128_digest(row.data), dtype=np.uint8)
    return digests


@dataclass(frozen=True, eq=False)
class Reference:
    """A column holding rows ``index`` of ``source``.

    Attributes:
        source: The source of the rows.
        index: The rows, in order, as positions in the source.
        digests: The digest of every row, once the rows were read.
    """

    source: ColumnSource
    index: RowIndex
    digests: Optional[NDArray[np.uint8]] = field(default=None)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def nbytes(self) -> int:
        return len(self) * self.source.nbytes_per_row

    def narrow(self, selector: Union[slice, int, NDArray[np.integer]]) -> Reference:
        """The rows ``selector`` of this reference, with their digests.

        ``selector`` is relative to this reference's rows, like the index of
        a ``KeyPath`` component: a slice or an array of positions.
        """
        match selector:
            case slice():
                index: RowIndex = self.index[selector]
                digests = None if self.digests is None else self.digests[selector]
            case np.ndarray():
                positions = selector.astype(np.int64, copy=False)
                index = np.asarray(self.index, dtype=np.int64)[positions]
                digests = None if self.digests is None else self.digests[positions]
            case _:
                raise TypeError(f"Cannot narrow a reference with {selector!r}.")
        return Reference(self.source, index, digests)

    def read(self) -> Tuple[np.ndarray, Reference]:
        """Read the rows, and the reference with their digests.

        Raises:
            StaleReferenceError: If the rows differ from the digests taken
                when they were first read, or a file changed.
        """
        rows = self.source.read(self.index)
        digests = row_digests(rows)
        if self.digests is not None and not np.array_equal(digests, self.digests):
            changed = np.flatnonzero(np.any(digests != self.digests, axis=1))
            raise StaleReferenceError(
                f"Rows {changed.tolist()[:10]} of {self.source.describe()} "
                "differ from the rows read before. Inputs must not change "
                "during a run.",
            )
        return rows, replace(self, digests=digests)

    def matches(self, rows: np.ndarray) -> bool:
        """Whether ``rows`` are still the rows this reference read."""
        return (
            self.digests is not None
            and len(rows) == len(self)
            and np.array_equal(row_digests(rows), self.digests)
        )

    @staticmethod
    def concat(references: Sequence[Reference]) -> Reference:
        """One reference to the rows of ``references``, in order."""
        source = ConcatSource(tuple(references))
        parts = [r.digests for r in references if r.digests is not None]
        digests = np.concatenate(parts) if len(parts) == len(references) else None
        return Reference(source, range(source.rows), digests)


class ConcatSource(ColumnSource):
    """The rows of several references, one after another."""

    def __init__(self, parts: Tuple[Reference, ...]) -> None:
        super().__init__()
        self.parts = parts

    @property
    def rows(self) -> int:
        return sum(len(part) for part in self.parts)

    @property
    def nbytes_per_row(self) -> int:
        return max((part.source.nbytes_per_row for part in self.parts), default=0)

    @property
    def files(self) -> Tuple[FileId, ...]:
        return tuple(dict.fromkeys(f for part in self.parts for f in part.source.files))

    @property
    def domain(self) -> str:
        domains = {part.source.domain for part in self.parts}
        return domains.pop() if len(domains) == 1 else ""

    def describe(self) -> str:
        return " + ".join(dict.fromkeys(part.source.describe() for part in self.parts))

    def _read(self, index: RowIndex) -> np.ndarray:
        rows = np.concatenate([part.source.read(part.index) for part in self.parts])
        return rows[_as_selector(index)]
