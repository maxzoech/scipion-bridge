"""Generic async sink writer protocol for streaming pipeline output.

A sink writes every item on its own: state across items (e.g. a running
average) belongs to stages of the pipeline (``scan``, ``accumulate``), where
lineage and checkpoints see it, and the sink writes the latest state. With a
run store, a run continuing in the store of an earlier run skips the items
its sinks committed; a writer keeping state across items would miss them.
"""

from __future__ import annotations

import asyncio
import enum
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from scipion_bridge.core.struct import references
from scipion_bridge.core.struct.schema import SchemaConvertible
from scipion_bridge.core.utils.files import write_atomic

# What a writer reports of a write, to undo the writes after it (JSON values).
Receipt = Dict[str, Any]


@runtime_checkable
class SinkWriter(Protocol):
    """Protocol for structured asynchronous streaming output.

    Implementations write SchemaConvertible containers (Struct, Set, Collection)
    atomically to a persistence target (Zarr, PostgreSQL, etc.).
    """

    async def write(self, item: SchemaConvertible) -> None:
        """Asynchronously write a SchemaConvertible item and ensure durable commit."""
        ...

    async def finalize(self) -> None:
        """Asynchronously flush pending writes and release resources."""
        ...


@runtime_checkable
class CommittingSinkWriter(SinkWriter, Protocol):
    """A sink writer that writes items with their ids, and can undo the
    writes no commit of the run store covers.

    A sink records a ``Commit`` after every write, with the receipt of the
    write. Commits are written in batches, so after a crash the target can
    hold writes without a commit; before its first item, a run continuing in
    the same run store calls ``recover`` with the receipt of the last commit,
    and the writer removes everything written after it. Then every item
    without a commit is written again, and none twice.

    Without a run store the sink calls ``write_identified`` and never
    ``recover``.
    """

    async def write_identified(self, item: Any, lid: bytes) -> Optional[Receipt]:
        """Write ``item``, whose id is ``lid``, durably, and return what undoing
        the writes after it needs."""
        ...

    async def recover(self, receipt: Optional[Receipt]) -> None:
        """Remove the writes after the one of ``receipt``; with ``None``, every
        write: no item was committed."""
        ...


class UniqueElements(enum.Enum):
    """What a sink does with an element it writes a second time.

    Elements are told apart by their element ids, per field of the Sets
    holding them (e.g. ``particles`` of every class). Elements without ids
    are not checked.
    """

    #: Write it; nothing is checked.
    OFF = "off"
    #: Write it and log the item ids and the number of elements written again.
    WARN = "warn"
    #: Fail the pipeline: for targets keyed by element, e.g. a Scipion set.
    RAISE = "raise"


class CallbackSinkWriter:
    """Async adapter wrapping a callable as a SinkWriter for testing/debugging.

    The callback runs in a thread, so that a slow callback does not block the
    event loop of the stage writing, which keeps accepting items meanwhile.
    """

    def __init__(self, callback: Callable[[Any], Any]):
        self.callback = callback

    async def write(self, item: Any) -> None:
        await asyncio.to_thread(self.callback, item)

    async def finalize(self) -> None:
        pass


class StoreSinkWriter:
    """Keeps every item in a file of its own, named by its id.

    ``<path>/<item id>.pickle``, written to a temporary name and renamed: a
    file is never partial, and writing an item again replaces it, so nothing
    needs undoing. The files can be read while the run goes (``read``).
    Rows that reference input files this process reads unchanged are kept as
    references, not bytes: reading the item reads them from the files.
    """

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = Path(path).resolve()

    async def write(self, item: Any) -> None:
        raise NotImplementedError(
            "StoreSinkWriter names its files by the ids of the items "
            "(write_identified).",
        )

    async def write_identified(self, item: Any, lid: bytes) -> Optional[Receipt]:
        await asyncio.to_thread(self._write, item, lid)
        return None

    async def recover(self, receipt: Optional[Receipt]) -> None:
        pass

    async def finalize(self) -> None:
        pass

    def read(self, lid: bytes) -> Any:
        """The item of id ``lid``."""
        return pickle.loads((self.path / f"{lid.hex()}.pickle").read_bytes())

    def _write(self, item: Any, lid: bytes) -> None:
        write_atomic(
            self.path / f"{lid.hex()}.pickle",
            references.dumps(item, references.PicklingMode.STORE),
        )
