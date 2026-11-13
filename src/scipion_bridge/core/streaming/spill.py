"""Storage of the items that overflow the buffers of a streaming pipeline.

A stage keeps a bounded number of the items waiting for it in memory; further
items are handed to a ``SpillStore`` until the stage is ready for them. The
store decides what to persist: the naive ``PickleSpillStore`` writes every
item, while a store aware of the provenance of the data may write only the
columns of a ``Set`` that changed and reference the data that already exists on
disk.

Items enter the transport between the stages of the Ray backend in a single
place (``_deliver``), so that such a store can later also keep unchanged data
out of the object store altogether.
"""

from __future__ import annotations

import abc
import contextlib
import functools
import pickle
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Generic, Hashable, TypeVar

from ..struct import references

H = TypeVar("H", bound=Hashable)


class SpillStore(abc.ABC, Generic[H]):
    """Stores the items that overflow the buffer of a pipeline stage.

    Every item is stored once, read back once and then discarded. Handles are
    small and picklable; the items themselves never pass through them.
    """

    @abc.abstractmethod
    def put(self, item: Any) -> H:
        """Store ``item`` and return the handle to read it back."""
        ...

    @abc.abstractmethod
    def get(self, handle: H) -> Any:
        """Reconstruct the item stored under ``handle``."""
        ...

    @abc.abstractmethod
    def discard(self, handle: H) -> None:
        """Free the storage of a consumed item."""
        ...

    def size(self, handle: H) -> int:
        """The bytes stored for ``handle``, for the stats; 0 if unknown."""
        return 0

    def close(self) -> None:
        """Free all storage of the store, once its stage has stopped.

        Called in the process of the stage, on the node of its storage.
        """


# Creates the store of a stage from the label of the stage.
SpillStoreFactory = Callable[[str], SpillStore[Any]]


class PickleSpillStore(SpillStore[str]):
    """Pickles every item into a file of its own inside ``directory``.

    The directory is created on the first item, on the node of the stage. A
    relative ``directory`` lies in the temporary directory of the process
    using the store, so every stage spills on its own node; ``close``
    removes it, and its first component once empty.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def path(self) -> Path:
        """The directory, resolved in the calling process."""
        match self.directory.is_absolute():
            case True:
                return self.directory
            case False:
                return Path(tempfile.gettempdir()) / self.directory

    def put(self, item: Any) -> str:
        """Pickle ``item``; the rows of references are kept only if needed.

        Columns of Sets that reference rows of input files are written as
        the reference when this process can read the files again
        (``core/struct/references.py``), so a spilled particle stack takes
        the bytes of its metadata, not of its images.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        handle = f"{uuid.uuid4().hex}.pkl"
        with (
            (self.path / handle).open("wb") as file,
            references.pickling(references.PicklingMode.SPILL),
        ):
            pickle.dump(item, file, protocol=pickle.HIGHEST_PROTOCOL)
        return handle

    def size(self, handle: str) -> int:
        return (self.path / handle).stat().st_size

    def get(self, handle: str) -> Any:
        with (self.path / handle).open("rb") as file:
            return pickle.load(file)

    def discard(self, handle: str) -> None:
        (self.path / handle).unlink()

    def close(self) -> None:
        # A store that never spilled has no directory.
        shutil.rmtree(self.path, ignore_errors=True)
        match self.directory.is_absolute():
            case True:
                pass
            case False:
                # The directory of the pipeline on this node, shared by its
                # stages: the last one to close removes it.
                root = Path(tempfile.gettempdir()) / self.directory.parts[0]
                with contextlib.suppress(OSError):
                    root.rmdir()


def pickle_spill_store(root: Path) -> SpillStoreFactory:
    """Factory of ``PickleSpillStore``s writing into a directory per stage under ``root``.

    A relative ``root`` lies in the temporary directory of every stage's
    process, on its own node.
    """
    return functools.partial(_pickle_store_in, root)


def _pickle_store_in(root: Path, label: str) -> PickleSpillStore:
    return PickleSpillStore(root / re.sub(r"[^\w.-]", "_", label))
