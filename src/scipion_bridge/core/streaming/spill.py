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
import functools
import pickle
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Generic, Hashable, TypeVar

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


# Creates the store of a stage from the label of the stage.
SpillStoreFactory = Callable[[str], SpillStore[Any]]


class PickleSpillStore(SpillStore[str]):
    """Pickles every item into a file of its own inside ``directory``.

    The directory is created on the first item, on the node of the stage.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def put(self, item: Any) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = f"{uuid.uuid4().hex}.pkl"
        with (self.directory / handle).open("wb") as file:
            pickle.dump(item, file, protocol=pickle.HIGHEST_PROTOCOL)
        return handle

    def get(self, handle: str) -> Any:
        with (self.directory / handle).open("rb") as file:
            return pickle.load(file)

    def discard(self, handle: str) -> None:
        (self.directory / handle).unlink()


def pickle_spill_store(root: Path) -> SpillStoreFactory:
    """Factory of ``PickleSpillStore``s writing into a directory per stage under ``root``."""
    return functools.partial(_pickle_store_in, root)


def _pickle_store_in(root: Path, label: str) -> PickleSpillStore:
    return PickleSpillStore(root / re.sub(r"[^\w.-]", "_", label))
