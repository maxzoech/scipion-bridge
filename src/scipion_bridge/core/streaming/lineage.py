"""Item ids: the identity of every item between the stages of a pipeline.

Every item a stage emits gets a 16-byte item id, carried next to the item
(never inside it, so user code does not see it). Item ids are deterministic:
the same inputs, received in the same order, give the same ids in every
run. Lineage links an item to the items it was made from by these ids, and
sinks will skip items whose id they already committed, which makes a resumed
run write every output once.

Every stage derives the ids of what it emits with a ``LineagePolicy``:

=================  ======================================================
Stage              Item id
=================  ======================================================
Source             ``("source", stage, digest(element ids), k)``, or
                   ``("source", stage, "seq", n)`` for items without
                   element ids
Map                ``("map", stage, parent)``
Accumulator        ``("acc", stage, sorted(parents), previous, index)``
Merge, sink,       the id of the item, unchanged
router
=================  ======================================================

- A source names an item by the element ids it holds, i.e. its exact input
  rows, whichever loader read it and in whatever order. ``k`` counts the
  items with the same element ids before it, so an input sent twice gives
  two items. Items without element ids (plain values) are counted in the
  order the source receives them (``n``).
- An accumulator's parents are the items it received since its last
  emission, and the item it processed last, which may have left rows in its
  state (e.g. the rest of a ``chunk``). ``previous`` is its last emission:
  it chains the history of the stage, so a ``scan`` does not list every item
  it folded. Parents are sorted, so the order of arrival across the inputs
  of a stage does not change the id.
- Inside a ``group_by`` child, the source passes the id of the item the
  router sent on.

Rows a stage creates get element ids from the id of the item they are
emitted in (``fill_created``): ``(item id, row)``.

The stage is named by its label, which is unique in a pipeline, including
the children of a ``group_by`` (whose labels hold their key).

With a run store, every stage that makes items records a ``LineageRecord``
per item: its id, its parents and the element ids it holds (as segments,
``ElementSegment``), and for source items the input files of their rows.
Merges, sinks and routers make no items, so they record nothing.
"""

from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..struct import identity
from ..struct.id_column import IdSegment
from ..struct.source import FileId
from ..struct.storage import RootEngine
from ..utils.hashing import item_id

from .provenance import sets_of

# The 16-byte id of an item.
ItemId = bytes


def map_id(stage: str, parent: ItemId) -> ItemId:
    """The id of the result of the map ``stage`` called on the item ``parent``."""
    return item_id(b"map", stage.encode(), parent)


def element_digest(item: Any) -> Optional[bytes]:
    """A digest of the element ids in ``item``, or ``None`` if it holds none.

    It covers the ids of every Set and Struct in the item and of the Sets
    nested in them, by where they are in the item.
    """
    parts: List[bytes] = []
    assigned = False
    for where, obj in sets_of(item):
        for key, column in sorted(identity.hidden_ids(obj).items()):
            assigned = assigned or column.any_assigned
            parts += [where.encode(), "/".join(key).encode(), *column.digest_parts()]
    return item_id(*parts) if assigned else None


def fill_created(item: Any, lid: ItemId) -> None:
    """Give the rows without element ids in ``item``, which a stage emits
    with the id ``lid``, ids of their own: ``(item_id(lid, where), row)``.

    ``where`` tells the Sets of an item apart, e.g. the slots of a
    Collection or the elements of a tuple.
    """
    for where, obj in sets_of(item):
        identity.fill_unassigned(obj, item_id(lid, where.encode()))


@dataclass(frozen=True)
class ElementSegment:
    """The element ids of consecutive elements of a Set in an item.

    Attributes:
        where: Where the Set is in the item (see ``sets_of``), e.g. ``""``
            for the item itself or ``"['classes'][3]"``.
        path: The field of the Set within that object, ``""`` for the object
            itself, e.g. ``"particles"``.
        count: The number of elements.
        namespace: The namespace of positional ids, else ``None``.
        start: The first position of a run of positions ``start..stop-1``.
        stop: The end of that run.
        positions: Other positions, as little-endian int64 bytes; also the
            positions of mixed ids.
        explicit: The 16-byte ids of explicit (hashed) ids, concatenated.
        namespaces: The namespaces of mixed ids (16 bytes each, concatenated).
        indices: The namespace of every mixed id, as little-endian int32
            indices into ``namespaces``.

    Elements without ids have neither namespaces nor explicit ids.
    """

    where: str
    path: str
    count: int
    namespace: Optional[bytes] = None
    start: Optional[int] = None
    stop: Optional[int] = None
    positions: Optional[bytes] = None
    explicit: Optional[bytes] = None
    namespaces: Optional[bytes] = None
    indices: Optional[bytes] = None

    @staticmethod
    def of(where: str, path: str, segment: IdSegment) -> ElementSegment:
        if segment.namespaces is not None and segment.indices is not None:
            return ElementSegment(
                where,
                path,
                len(segment),
                positions=np.asarray(segment.positions, dtype="<i8").tobytes(),
                namespaces=np.ascontiguousarray(segment.namespaces).tobytes(),
                indices=np.asarray(segment.indices, dtype="<i4").tobytes(),
            )
        elif segment.explicit is not None:
            return ElementSegment(
                where,
                path,
                len(segment),
                explicit=np.ascontiguousarray(segment.explicit).tobytes(),
            )
        elif segment.namespace is None:
            return ElementSegment(where, path, len(segment))
        elif isinstance(segment.positions, range) and segment.positions.step == 1:
            return ElementSegment(
                where,
                path,
                len(segment),
                segment.namespace,
                segment.positions.start,
                segment.positions.stop,
            )
        else:
            positions = np.asarray(segment.positions, dtype="<i8")
            return ElementSegment(
                where,
                path,
                len(segment),
                segment.namespace,
                positions=positions.tobytes(),
            )

    def position_array(self) -> NDArray[np.int64]:
        """The positions of positional ids."""
        if self.start is not None and self.stop is not None:
            return np.arange(self.start, self.stop, dtype=np.int64)
        elif self.positions is not None:
            return np.frombuffer(self.positions, dtype="<i8").astype(np.int64)
        else:
            return np.empty(0, dtype=np.int64)

    def holds(self, namespace: bytes, position: int) -> bool:
        """Whether the element ``(namespace, position)`` is one of these."""
        if self.namespaces is not None and self.indices is not None:
            return self._holds_mixed(self.namespaces, self.indices, namespace, position)
        elif self.namespace != namespace:
            return False
        elif self.start is not None and self.stop is not None:
            return self.start <= position < self.stop
        else:
            return bool(np.any(self.position_array() == position))

    def _holds_mixed(
        self,
        namespaces: bytes,
        indices: bytes,
        namespace: bytes,
        position: int,
    ) -> bool:
        table = np.frombuffer(namespaces, dtype="V16")
        matches = np.flatnonzero(table == np.frombuffer(namespace, dtype="V16")[0])
        if len(matches) == 0:
            return False
        else:
            of_namespace = np.frombuffer(indices, dtype="<i4") == matches[0]
            return bool(np.any(of_namespace & (self.position_array() == position)))


@dataclass(frozen=True)
class LineageRecord:
    """An item a stage made: what lineage stores of it.

    Attributes:
        item_id: The id of the item.
        stage: The label of the stage that made it.
        kind: ``"source"``, ``"map"`` or ``"accumulate"``.
        parents: The ids of the items it was made from, sorted; none for a
            source item.
        elements: The element ids it holds; empty if ``inherited``.
        files: For a source item, the input files its rows reference.
        inherited: Whether it holds the element ids of its only parent, e.g.
            a map changing values, not rows. Most maps do, so their records
            do not repeat the ids.
    """

    item_id: ItemId
    stage: str
    kind: str
    parents: Tuple[ItemId, ...]
    elements: Tuple[ElementSegment, ...]
    files: Tuple[FileId, ...] = ()
    inherited: bool = False


def element_segments(item: Any) -> Tuple[ElementSegment, ...]:
    """The element ids of every Set and Struct in ``item``, as segments."""
    return tuple(
        ElementSegment.of(where, "/".join(key), segment)
        for where, obj in sets_of(item)
        for key, column in sorted(identity.hidden_ids(obj).items())
        for segment in column.segments
    )


def call_elements(item: Any, result: Any) -> Optional[Tuple[ElementSegment, ...]]:
    """The element ids of the ``result`` of a map called on ``item``; ``None``
    if they are the ids of ``item``."""
    if result is item:
        return None
    elements = element_segments(result)
    return None if elements == element_segments(item) else elements


def input_files(item: Any) -> Tuple[FileId, ...]:
    """The files the rows of ``item`` reference, e.g. its input stacks."""
    files = [
        file
        for _, obj in sets_of(item)
        if isinstance(engine := obj.storage.root_storage, RootEngine)
        for reference in engine.references().values()
        for file in reference.source.files
    ]
    return tuple(dict.fromkeys(files))


class LineagePolicy(abc.ABC):
    """How a stage derives the ids of the items it emits.

    ``kind`` names the items it makes in lineage records; stages that pass
    items on make none (``None``).
    """

    kind: Optional[str] = None

    def __init__(self, stage: str) -> None:
        """
        Args:
            stage: The label of the stage.
        """
        self._stage = stage.encode()
        # The parents of the items emitted last.
        self.emitted_parents: Tuple[ItemId, ...] = ()

    def receive(self, item: Any, lid: Optional[ItemId]) -> ItemId:
        """The id of an arriving item; ``lid`` is the id it was sent with.

        Only sources receive items without an id, from the driver and the
        loaders.
        """
        assert lid is not None, "Items between stages carry their id."
        return lid

    @abc.abstractmethod
    def emit(self, received: ItemId, count: int) -> List[ItemId]:
        """The ids of the ``count`` items the stage emits for ``received``."""
        ...

    def flush(self, count: int) -> List[ItemId]:
        """The ids of the ``count`` items the stage emits on FLUSH."""
        assert count == 0, "Only accumulators emit on FLUSH."
        return []


class PassThrough(LineagePolicy):
    """Stages emitting the items they receive: merges, sinks, routers."""

    def emit(self, received: ItemId, count: int) -> List[ItemId]:
        assert count <= 1, "A pass-through stage emits what it receives."
        return [received] * count


class SourceLineage(PassThrough):
    """Sources: items from the driver and the loaders get an id from their
    element ids; items a ``group_by`` router sent keep theirs."""

    kind = "source"

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        # Items received with the same element ids, by their digest.
        self._seen: Counter[bytes] = Counter()
        # Items received without element ids.
        self._unnamed = 0

    def receive(self, item: Any, lid: Optional[ItemId]) -> ItemId:
        if lid is not None:
            return lid
        digest = element_digest(item)
        if digest is None:
            self._unnamed += 1
            return item_id(b"source", self._stage, b"seq", _int(self._unnamed))
        else:
            k = self._seen[digest]
            self._seen[digest] += 1
            return item_id(b"source", self._stage, digest, _int(k))


class AccumulatorLineage(LineagePolicy):
    """Accumulators: emissions name the items received since the last one."""

    kind = "accumulate"

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        # Items received since the last emission.
        self._pending: List[ItemId] = []
        # The item processed when the stage last emitted, which may have
        # left rows in the state.
        self._carried: Optional[ItemId] = None
        self._previous: Optional[ItemId] = None

    def emit(self, received: ItemId, count: int) -> List[ItemId]:
        self._pending.append(received)
        if count == 0:
            return []
        ids = self._emissions(count)
        self._carried = received
        return ids

    def flush(self, count: int) -> List[ItemId]:
        ids = self._emissions(count) if count else []
        self._carried = None
        self._pending = []
        return ids

    def parents(self) -> List[ItemId]:
        """The parents of the next emission, sorted."""
        carried = [] if self._carried is None else [self._carried]
        return sorted({*self._pending, *carried})

    def _emissions(self, count: int) -> List[ItemId]:
        parents = self.parents()
        self.emitted_parents = tuple(parents)
        previous = b"" if self._previous is None else self._previous
        ids = [
            item_id(b"acc", self._stage, *parents, b"previous", previous, _int(i))
            for i in range(count)
        ]
        self._pending = []
        self._previous = ids[-1]
        return ids


def _int(value: int) -> bytes:
    return value.to_bytes(8, "little")
