"""Questions about a run, answered from its lineage records.

``LineageGraph`` reads the records of a run (see ``run_store.py``):

- ``ancestry(item_id)``: the items an item was made from, back to the source
  items, with the stages that made them and the input files they read;
- ``trace_element(namespace, position)``: where an element came from. An
  element of an input file is the answer itself (the file and its row); an
  element a stage created is traced from the item that created it;
- ``holders(namespace, position)``: the items holding an element, e.g. the
  outputs a particle ended up in.

Lineage records that an output was made by a stage, with its code and
parameters, *using these items in some way*: rows that move (views,
``astype``) keep their element ids exactly, values computed from many rows
inside a map depend on the whole item.

The graph is columnar, so that it holds the runs of millions of particles:
records are rows of NumPy arrays and the element ids stay in the Arrow
buffers they were read into. Only the records a query returns become
``LineageRecord`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ..struct.source import FileId
from .lineage import ElementSegment, ItemId, LineageRecord
from .run_store import RunStore, StageInfo, records_table

# An element id as a NumPy scalar, for comparisons with arrays of them.
_ID = np.dtype("V16")


@dataclass(frozen=True)
class Ancestry:
    """The items an item was made from.

    Attributes:
        item: The record of the item.
        item_ids: The item and every item it was made from, nearest first.
        sources: The source items among them.
        stages: The stages that made them, nearest first, with their
            fingerprints where the run recorded them.
        files: The input files the source items read.
        missing: Parents without a record, e.g. of a stage whose records
            were not written because the run failed.
    """

    item: LineageRecord
    item_ids: Tuple[ItemId, ...]
    sources: Tuple[LineageRecord, ...]
    stages: Tuple[StageInfo, ...]
    files: Tuple[FileId, ...]
    missing: Tuple[ItemId, ...]


@dataclass(frozen=True)
class ElementOrigin:
    """Where an element came from.

    Attributes:
        file: The input file the element is a row of, if it is one.
        row: Its row in that file.
        created_by: The record of the item a stage created it in, if a
            stage created it.
        ancestry: The ancestry of that item.
    """

    file: Optional[FileId] = None
    row: Optional[int] = None
    created_by: Optional[LineageRecord] = None
    ancestry: Optional[Ancestry] = None


@dataclass(frozen=True)
class StageSummary:
    """What a stage made in a run.

    Attributes:
        stage: The stage.
        items: The items it made.
        elements: The elements of those items.
        input_elements: The elements among them that are rows of input files.
        inherited: The items holding the elements of their parent (maps
            changing values, not rows).
    """

    stage: StageInfo
    items: int
    elements: int
    input_elements: int
    inherited: int


class LineageGraph:
    """The lineage records of a run, indexed for queries."""

    def __init__(self, table: pa.Table, stages: Mapping[str, StageInfo]) -> None:
        """
        Args:
            table: The records, as ``RunStore.read_lineage_table`` returns them.
            stages: The stages of the run, by label.
        """
        table = table.combine_chunks()
        ids: List[ItemId] = table.column("item_id").to_pylist()
        first: Dict[ItemId, int] = {}
        for row, lid in enumerate(ids):
            first.setdefault(lid, row)
        if len(first) < len(ids):
            # A run continuing an earlier one in its store records the items
            # again, with the same ids: keep the first record of each.
            table = table.take(list(first.values())).combine_chunks()
            ids = list(first)
        self.stages = dict(stages)
        self._table = table
        self._ids = ids
        self._row = {lid: row for row, lid in enumerate(self._ids)}
        stage = table.column("stage").dictionary_encode().combine_chunks()
        self._stage_labels: List[str] = stage.dictionary.to_pylist()
        self._stage = stage.indices.to_numpy()
        self._kind: List[str] = table.column("kind").to_pylist()
        # Null in parts written before records could inherit their elements.
        self._inherited = (
            table.column("inherited").fill_null(False).to_numpy(zero_copy_only=False)
        )
        parents = table.column("parents").chunk(0)
        self._parent_offsets = parents.offsets.to_numpy()
        flat_parents = parents.flatten().to_pylist()
        self._parents = np.array(
            [self._row.get(p, -1) for p in flat_parents],
            dtype=np.int64,
        )
        self._file_table, self._file_rows = _source_files(table)
        self._segments = _Segments(table.column("elements").chunk(0))

    @staticmethod
    def of(store: RunStore) -> LineageGraph:
        """The lineage of the run kept in ``store``, as written so far."""
        return LineageGraph(store.read_lineage_table(), store.read_stages())

    @staticmethod
    def from_records(
        records: Sequence[LineageRecord],
        stages: Mapping[str, StageInfo],
    ) -> LineageGraph:
        return LineageGraph(records_table(records), stages)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._row

    def item_ids(self) -> List[ItemId]:
        """The ids of every recorded item, in the order they were written."""
        return list(self._ids)

    def record(self, item_id: ItemId) -> LineageRecord:
        """The record of an item, with its own element ids.

        Raises:
            KeyError: If no record of the item exists.
        """
        row = self._row[item_id]
        return LineageRecord(
            item_id=item_id,
            stage=self._stage_labels[self._stage[row]],
            kind=self._kind[row],
            parents=self.parents(item_id),
            elements=tuple(self._segments.of_record(row)),
            files=self._files_of(row),
            inherited=bool(self._inherited[row]),
        )

    def elements(self, item_id: ItemId) -> Tuple[ElementSegment, ...]:
        """The element ids an item holds, its own or its parent's."""
        row = self._row[item_id]
        while self._inherited[row] and self._parent_rows(row).size:
            row = int(self._parent_rows(row)[0])
        return tuple(self._segments.of_record(row))

    def element_count(self, item_id: ItemId) -> Tuple[int, int]:
        """The elements an item holds, and how many of them are input rows."""
        row = self._row[item_id]
        own, inputs = self._element_counts
        return int(own[row]), int(inputs[row])

    def stage_of(self, item_id: ItemId) -> str:
        return self._stage_labels[self._stage[self._row[item_id]]]

    def kind_of(self, item_id: ItemId) -> str:
        return self._kind[self._row[item_id]]

    def files(self) -> int:
        """The number of distinct input files of the run."""
        return len(self._file_table)

    def file(self, index: int) -> FileId:
        """The input file ``index`` (``0 <= index < files()``)."""
        return self._file(index)

    def find_files(self, text: str, limit: int) -> List[Tuple[int, FileId]]:
        """Up to ``limit`` input files whose path contains ``text``."""
        found = [
            index for index, (path, _, _) in enumerate(self._file_table) if text in path
        ]
        return [(index, self._file(index)) for index in found[:limit]]

    def file_index(self, namespace: bytes) -> Optional[int]:
        """The input file whose rows have ``namespace``, if one has."""
        return self._file_namespaces.get(namespace)

    def rows_of(self, namespace: bytes) -> NDArray[np.int64]:
        """The positions of the elements of ``namespace`` the source items
        held, sorted: e.g. the rows of an input stack the run read."""
        return self._segments.positions_of(namespace, self._source_rows)

    def warm(self) -> None:
        """Build the indices of the queries now, e.g. in a background thread,
        so that the first query does not wait for them."""
        _ = self._element_counts, self._segments._index, self._children_by_parent

    @cached_property
    def _source_rows(self) -> NDArray[np.int64]:
        return np.flatnonzero(np.array(self._kind) == "source")

    def parents(self, item_id: ItemId) -> Tuple[ItemId, ...]:
        row = self._row[item_id]
        start, stop = self._parent_offsets[row], self._parent_offsets[row + 1]
        flat = self._table.column("parents").chunk(0).values
        return tuple(flat[start:stop].to_pylist())

    def children(self, item_id: ItemId) -> Tuple[ItemId, ...]:
        """The items made from an item."""
        row = self._row[item_id]
        start, stop = np.searchsorted(self._children_by_parent[0], [row, row + 1])
        return tuple(self._ids[r] for r in self._children_by_parent[1][start:stop])

    def leaves(self) -> List[ItemId]:
        """The items no stage made anything from: the outputs of the run, and
        items a stage absorbed without emitting."""
        has_children = np.zeros(len(self), dtype=bool)
        has_children[self._parents[self._parents >= 0]] = True
        return [self._ids[row] for row in np.flatnonzero(~has_children)]

    def items_of(self, stage: str) -> List[ItemId]:
        """The items the stage ``stage`` made, in the order they were written."""
        index = self._stage_labels.index(stage)
        return [self._ids[row] for row in np.flatnonzero(self._stage == index)]

    def stage_summaries(self) -> List[StageSummary]:
        """What every stage made, in the order of the stages of the run."""
        counts = self._element_counts
        summaries = [
            StageSummary(
                stage=self._stage_info(label),
                items=int(np.sum(mask)),
                elements=int(counts[0][mask].sum()),
                input_elements=int(counts[1][mask].sum()),
                inherited=int(self._inherited[mask].sum()),
            )
            for index, label in enumerate(self._stage_labels)
            for mask in [self._stage == index]
        ]
        return sorted(
            summaries,
            key=lambda s: (_stage_position(s.stage.label), s.stage.label),
        )

    def ancestry(self, item_id: ItemId) -> Ancestry:
        """The items the item ``item_id`` was made from.

        Raises:
            KeyError: If no record of the item exists.
        """
        start = self._row[item_id]
        visited = np.zeros(len(self), dtype=bool)
        order: List[int] = []
        missing: Dict[ItemId, None] = {}
        frontier = np.array([start], dtype=np.int64)
        visited[start] = True
        while frontier.size:
            order.extend(frontier.tolist())
            parents = self._parents_of(frontier)
            if np.any(parents < 0):
                missing.update(
                    (lid, None)
                    for row in frontier
                    for lid in self._missing_parents(int(row))
                )
            parents = np.unique(parents[parents >= 0])
            frontier = parents[~visited[parents]]
            visited[frontier] = True
        sources = [row for row in order if self._kind[row] == "source"]
        labels = dict.fromkeys(self._stage_labels[self._stage[row]] for row in order)
        return Ancestry(
            item=self.record(item_id),
            item_ids=tuple(self._ids[row] for row in order),
            sources=tuple(self.record(self._ids[row]) for row in sources),
            stages=tuple(self._stage_info(label) for label in labels),
            files=tuple(
                self._file(index)
                for index in dict.fromkeys(
                    index for row in sources for index in self._file_rows.get(row, ())
                )
            ),
            missing=tuple(missing),
        )

    def descendants(self, item_id: ItemId) -> List[ItemId]:
        """The item and every item made from it, nearest first: e.g. the
        outputs an input chunk went into."""
        start = self._row[item_id]
        visited = np.zeros(len(self), dtype=bool)
        order: List[int] = []
        frontier = np.array([start], dtype=np.int64)
        visited[start] = True
        parents, children = self._children_by_parent
        while frontier.size:
            order.extend(frontier.tolist())
            starts = np.searchsorted(parents, frontier)
            stops = np.searchsorted(parents, frontier + 1)
            found = np.unique(_gather_ranges(children, starts, stops - starts))
            frontier = found[~visited[found]]
            visited[frontier] = True
        return [self._ids[row] for row in order]

    def stage_links(
        self,
        item_ids: Sequence[ItemId],
    ) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """How the items ``item_ids`` spread over the stages: the items of
        every stage, and the parent-child links between their items, by the
        stages of the parent and the child."""
        rows = np.array([self._row[lid] for lid in item_ids], dtype=np.int64)
        stages = np.bincount(self._stage[rows], minlength=len(self._stage_labels))
        selected = np.zeros(len(self), dtype=bool)
        selected[rows] = True
        lengths = self._parent_offsets[rows + 1] - self._parent_offsets[rows]
        children = np.repeat(rows, lengths)
        parents = self._parents_of(rows)
        linked = (parents >= 0) & selected[np.maximum(parents, 0)]
        pairs = (
            self._stage[parents[linked]].astype(np.int64) * len(self._stage_labels)
            + self._stage[children[linked]]
        )
        codes, counts = np.unique(pairs, return_counts=True)
        n = len(self._stage_labels)
        return (
            {
                self._stage_labels[i]: int(count)
                for i, count in enumerate(stages)
                if count
            },
            {
                (self._stage_labels[code // n], self._stage_labels[code % n]): int(c)
                for code, c in zip(codes.tolist(), counts.tolist())
            },
        )

    def trace_element(self, namespace: bytes, position: int) -> ElementOrigin:
        """Where the element ``(namespace, position)`` came from.

        Raises:
            KeyError: If no record holds the element.
        """
        if namespace in self._file_namespaces:
            file = self._file(self._file_namespaces[namespace])
            return ElementOrigin(file=file, row=position)
        holders = self._segments.holder_rows(namespace, position)
        held = set(holders.tolist())
        creators = [
            row
            for row in holders.tolist()
            if not held.intersection(self._parent_rows(row).tolist())
        ]
        if not creators:
            raise KeyError(f"No item of the run holds the element {position}.")
        created_by = self._ids[creators[0]]
        return ElementOrigin(
            created_by=self.record(created_by),
            ancestry=self.ancestry(created_by),
        )

    def holders(self, namespace: bytes, position: int) -> List[ItemId]:
        """Every item holding the element ``(namespace, position)``: the
        items whose element ids hold it, and the items inheriting theirs."""
        rows = list(self._segments.holder_rows(namespace, position).tolist())
        found = set(rows)
        queue = list(rows)
        while queue:
            row = queue.pop()
            for child in self._child_rows(row):
                if self._inherited[child] and child not in found:
                    found.add(child)
                    rows.append(child)
                    queue.append(child)
        return [self._ids[row] for row in rows]

    @cached_property
    def _file_namespaces(self) -> Dict[bytes, int]:
        """The input files by the namespace of their rows."""
        return {
            FileId.namespace_of(path, size, mtime_ns): index
            for index, (path, size, mtime_ns) in enumerate(self._file_table)
        }

    def _file(self, index: int) -> FileId:
        path, size, mtime_ns = self._file_table[index]
        return FileId(Path(path), size, mtime_ns)

    def _files_of(self, row: int) -> Tuple[FileId, ...]:
        return tuple(self._file(index) for index in self._file_rows.get(row, ()))

    @cached_property
    def _children_by_parent(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(parent row, child row) pairs, sorted by parent row."""
        lengths = np.diff(self._parent_offsets)
        child = np.repeat(np.arange(len(self), dtype=np.int64), lengths)
        known = self._parents >= 0
        parent, child = self._parents[known], child[known]
        order = np.argsort(parent, kind="stable")
        return parent[order], child[order]

    @cached_property
    def _element_counts(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The elements of every record, and those that are input rows; an
        inheriting record counts the elements of its parent."""
        own, inputs = self._segments.counts(len(self), list(self._file_namespaces))
        # Every inheriting record takes the counts of the first ancestor
        # that holds element ids of its own.
        rows = np.flatnonzero(self._inherited)
        source = self._first_parent[rows]
        while True:
            follow = (source >= 0) & self._inherited[np.maximum(source, 0)]
            if not follow.any():
                break
            source[follow] = self._first_parent[source[follow]]
        found = source >= 0
        own[rows[found]] = own[source[found]]
        inputs[rows[found]] = inputs[source[found]]
        return own, inputs

    @cached_property
    def _first_parent(self) -> NDArray[np.int64]:
        """The first parent row of every record, -1 without one."""
        lengths = np.diff(self._parent_offsets)
        first = np.full(len(self), -1, dtype=np.int64)
        has = lengths > 0
        first[has] = self._parents[self._parent_offsets[:-1][has]]
        return first

    def _parent_rows(self, row: int) -> NDArray[np.int64]:
        return self._parents[self._parent_offsets[row] : self._parent_offsets[row + 1]]

    def _parents_of(self, rows: NDArray[np.int64]) -> NDArray[np.int64]:
        """The parent rows of ``rows``, concatenated (-1 for missing ones)."""
        starts = self._parent_offsets[rows]
        return _gather_ranges(
            self._parents, starts, self._parent_offsets[rows + 1] - starts
        )

    def _child_rows(self, row: int) -> List[int]:
        parents, children = self._children_by_parent
        start, stop = np.searchsorted(parents, [row, row + 1])
        return children[start:stop].tolist()

    def _missing_parents(self, row: int) -> List[ItemId]:
        """The parents of a record that have no record."""
        unknown = np.flatnonzero(self._parent_rows(row) < 0)
        if unknown.size == 0:
            return []
        else:
            flat = self._table.column("parents").chunk(0).values
            start = int(self._parent_offsets[row])
            return [flat[start + int(i)].as_py() for i in unknown]

    def _stage_info(self, label: str) -> StageInfo:
        """The stage ``label``, as recorded, or only its label."""
        return self.stages.get(label, StageInfo(label, "", None))


class _Segments:
    """The element-id segments of all records, as flat columns.

    Positional and mixed segments are indexed by namespace, so finding the
    items holding an element reads only the segments of its namespace.
    """

    def __init__(self, elements: pa.ListArray) -> None:
        self._offsets = elements.offsets.to_numpy()
        flat = elements.flatten()
        self._flat = flat
        self.count = flat.field("count").to_numpy(zero_copy_only=False)
        self.start = flat.field("start").fill_null(-1).to_numpy()
        self.stop = flat.field("stop").fill_null(-1).to_numpy()
        self.owner = np.repeat(
            np.arange(len(self._offsets) - 1, dtype=np.int64),
            np.diff(self._offsets),
        )
        namespace = flat.field("namespace")
        self._namespace = _ids_of(namespace)
        self._has_namespace = namespace.is_valid().to_numpy(zero_copy_only=False)
        self._positions = flat.field("positions")
        self._namespaces = flat.field("namespaces")
        self._indices = flat.field("indices")

    def of_record(self, row: int) -> Iterator[ElementSegment]:
        for index in range(self._offsets[row], self._offsets[row + 1]):
            yield ElementSegment(**self._flat[int(index)].as_py())

    def positions_of(
        self,
        namespace: bytes,
        records: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """The positions of ``namespace`` in the segments of ``records``."""
        key = np.frombuffer(namespace, dtype=_ID)[0]
        keys, segments = self._index
        start, stop = np.searchsorted(keys, [_prefix(key), _prefix(key) + 1])
        found: List[NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
        for segment in np.unique(segments[start:stop]).tolist():
            if np.isin(self.owner[segment], records):
                found.append(self._positions_in(segment, key))
        return np.unique(np.concatenate(found))

    def _positions_in(self, segment: int, key: np.void) -> NDArray[np.int64]:
        """The positions of the elements of ``key`` in a segment."""
        if self._has_namespace[segment]:
            if self._namespace[segment] != key:
                return np.empty(0, dtype=np.int64)
            elif self.start[segment] >= 0:
                return np.arange(self.start[segment], self.stop[segment])
            else:
                return _view(self._positions, segment, np.dtype("<i8")).astype(np.int64)
        table = _view(self._namespaces, segment, _ID)
        matches = np.flatnonzero(table == key)
        if matches.size == 0:
            return np.empty(0, dtype=np.int64)
        positions = _view(self._positions, segment, np.dtype("<i8"))
        indices = _view(self._indices, segment, np.dtype("<i4"))
        return positions[indices == matches[0]].astype(np.int64)

    def holder_rows(self, namespace: bytes, position: int) -> NDArray[np.int64]:
        """The records with a segment holding ``(namespace, position)``, in
        the order they were written."""
        key = np.frombuffer(namespace, dtype=_ID)[0]
        keys, segments = self._index
        start, stop = np.searchsorted(keys, [_prefix(key), _prefix(key) + 1])
        holding = [
            int(segment)
            for segment in np.unique(segments[start:stop])
            if self._holds(int(segment), key, position)
        ]
        return np.unique(self.owner[holding]) if holding else np.empty(0, np.int64)

    def counts(
        self,
        records: int,
        input_namespaces: Sequence[bytes],
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The elements of every record, and those of ``input_namespaces``."""
        own = np.bincount(self.owner, weights=self.count, minlength=records)
        keys = np.sort(_prefixes(np.frombuffer(b"".join(input_namespaces), dtype=_ID)))
        positional = self._has_namespace & _members(_prefixes(self._namespace), keys)
        inputs = np.bincount(
            self.owner[positional],
            weights=self.count[positional],
            minlength=records,
        ).astype(np.float64)
        tables, table_owner = self._mixed_tables
        offsets, data = _binary_layout(self._indices)
        if data is not None and tables.size:
            is_input = _members(_prefixes(tables), keys)
            mixed = np.flatnonzero(np.diff(offsets) > 0)
            lengths = (np.diff(offsets) // 4)[mixed]
            indices = np.frombuffer(
                data,
                dtype="<i4",
                count=int(lengths.sum()),
                offset=int(offsets[0]),
            )
            # The first entry of the table of every mixed segment.
            table_start = np.searchsorted(table_owner, mixed)
            element_table = np.repeat(table_start, lengths) + indices
            per_segment = np.bincount(
                np.repeat(np.arange(mixed.size), lengths),
                weights=is_input[element_table],
                minlength=mixed.size,
            )
            inputs += np.bincount(
                self.owner[mixed],
                weights=per_segment,
                minlength=records,
            )
        return own.astype(np.int64), inputs.astype(np.int64)

    @cached_property
    def _mixed_tables(self) -> Tuple[NDArray[np.void], NDArray[np.int64]]:
        """The namespaces of every mixed segment, concatenated, and the
        segment of each."""
        offsets, data = _binary_layout(self._namespaces)
        lengths = np.diff(offsets) // 16
        if data is None or lengths.sum() == 0:
            return np.empty(0, dtype=_ID), np.empty(0, dtype=np.int64)
        tables = np.frombuffer(
            data,
            dtype=_ID,
            count=int(lengths.sum()),
            offset=int(offsets[0]),
        )
        return tables, np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)

    @cached_property
    def _index(self) -> Tuple[NDArray[np.uint64], NDArray[np.int64]]:
        """The namespace of every positional segment and of every namespace
        of a mixed segment, as the first 8 bytes of the namespace, with the
        segment; sorted by namespace. Candidates are checked in full."""
        positional = np.flatnonzero(self._has_namespace)
        tables, table_owner = self._mixed_tables
        keys = np.concatenate(
            [_prefixes(self._namespace[positional]), _prefixes(tables)],
        )
        segments = np.concatenate([positional, table_owner])
        order = np.argsort(keys, kind="stable")
        return keys[order], segments[order]

    def _holds(self, segment: int, key: np.void, position: int) -> bool:
        """Whether a segment indexed under the prefix of ``key`` holds
        ``(key, position)``."""
        if self._has_namespace[segment]:
            if self._namespace[segment] != key:
                return False
            elif self.start[segment] >= 0:
                return bool(self.start[segment] <= position < self.stop[segment])
            else:
                positions = _view(self._positions, segment, np.dtype("<i8"))
                return bool(np.any(positions == position))
        table = _view(self._namespaces, segment, _ID)
        matches = np.flatnonzero(table == key)
        if matches.size == 0:
            return False
        else:
            positions = _view(self._positions, segment, np.dtype("<i8"))
            indices = _view(self._indices, segment, np.dtype("<i4"))
            return bool(np.any((indices == matches[0]) & (positions == position)))


def _source_files(
    table: pa.Table,
) -> Tuple[List[Tuple[str, int, int]], Dict[int, Tuple[int, ...]]]:
    """The input files of the source records: every distinct file, and the
    files of every source record as indices of them."""
    files = table.column("files").chunk(0)
    offsets = files.offsets.to_numpy()
    flat = files.flatten()
    entries = zip(
        flat.field("path").to_pylist(),
        flat.field("size").to_pylist(),
        flat.field("mtime_ns").to_pylist(),
    )
    index: Dict[Tuple[str, int, int], int] = {}
    indices = [index.setdefault(entry, len(index)) for entry in entries]
    rows = np.flatnonzero(np.diff(offsets) > 0)
    return list(index), {
        int(row): tuple(indices[offsets[row] : offsets[row + 1]]) for row in rows
    }


def _ids_of(array: pa.FixedSizeBinaryArray) -> NDArray[np.void]:
    """The 16-byte values of ``array`` (zeros where null), without copies."""
    data = array.buffers()[1]
    if data is None:
        return np.zeros(len(array), dtype=_ID)
    return np.frombuffer(data, dtype=_ID, count=len(array), offset=array.offset * 16)


def _binary_layout(
    array: pa.BinaryArray,
) -> Tuple[NDArray[np.int32], Optional[pa.Buffer]]:
    """The offsets of the values of ``array`` into its data, and the data."""
    _, offsets, data = array.buffers()
    assert offsets is not None
    return (
        np.frombuffer(
            offsets, dtype="<i4", count=len(array) + 1, offset=array.offset * 4
        ),
        data,
    )


def _prefix(key: np.void) -> np.uint64:
    """The first 8 bytes of a 16-byte id, as an integer to sort by."""
    return np.frombuffer(key.tobytes()[:8], dtype="<u8")[0]


def _prefixes(keys: NDArray[np.void]) -> NDArray[np.uint64]:
    """The first 8 bytes of every 16-byte id of ``keys``."""
    raw = np.frombuffer(keys.tobytes(), dtype="<u8").reshape(-1, 2)
    return raw[:, 0].copy()


def _gather_ranges(
    values: NDArray[np.int64],
    starts: NDArray[np.int64],
    lengths: NDArray[np.int64],
) -> NDArray[np.int64]:
    """``values[starts[i] : starts[i] + lengths[i]]`` for every ``i``, joined."""
    offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
    return values[offsets + np.arange(lengths.sum())]


def _members(values: NDArray[np.uint64], keys: NDArray[np.uint64]) -> NDArray[np.bool_]:
    """Which of ``values`` are in the sorted ``keys``."""
    if keys.size == 0:
        return np.zeros(values.shape, dtype=bool)
    found = np.searchsorted(keys, values)
    return keys[np.minimum(found, keys.size - 1)] == values


def _view(array: pa.BinaryArray, index: int, dtype: np.dtype) -> NDArray:
    """The bytes of ``array[index]`` as an array of ``dtype``, without copies."""
    return np.frombuffer(array[index].as_buffer(), dtype=dtype)


def _stage_position(label: str) -> Tuple[int, ...]:
    """The position of a stage in its pipeline: ``19:group_by[a]/3:chunk``
    sorts after stage 19 and by its position in the child."""
    return tuple(
        int(part.split(":", 1)[0]) if part.split(":", 1)[0].isdigit() else 0
        for part in label.split("/")
    )


def element_arrays(
    segments: Sequence[ElementSegment],
) -> Tuple[NDArray[np.void], NDArray[np.int64]]:
    """The elements of ``segments`` as two arrays: the namespace (or the
    explicit id) of every element, and its position (-1 for explicit ids and
    elements without ids, whose namespace is zeros)."""
    namespaces: List[NDArray[np.void]] = [np.empty(0, dtype=_ID)]
    positions: List[NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    for segment in segments:
        if segment.namespaces is not None and segment.indices is not None:
            table = np.frombuffer(segment.namespaces, dtype=_ID)
            namespaces.append(table[np.frombuffer(segment.indices, dtype="<i4")])
            positions.append(segment.position_array())
        elif segment.namespace is not None:
            key = np.frombuffer(segment.namespace, dtype=_ID)
            namespaces.append(np.repeat(key, segment.count))
            positions.append(segment.position_array())
        elif segment.explicit is not None:
            namespaces.append(np.frombuffer(segment.explicit, dtype=_ID))
            positions.append(np.full(segment.count, -1, dtype=np.int64))
        else:
            namespaces.append(np.zeros(segment.count, dtype=_ID))
            positions.append(np.full(segment.count, -1, dtype=np.int64))
    return np.concatenate(namespaces), np.concatenate(positions)
