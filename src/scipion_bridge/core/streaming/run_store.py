"""The run store: where a run keeps its manifest and lineage.

A run of a pipeline with a run store records:

- its **manifest**: what ran on what (the library version, the protocols and
  their fingerprints, the parameters, the inputs and the chunking), written
  by the runner;
- its **report**: the graph of its pipeline (``PipelineReport.to_json``),
  written by the runner when the run starts and with the stats of the
  stages when it ends;
- its **stages**: the label, kind and ``stage_fp`` of every stage, written
  by every compiled pipeline when it starts, the children of a ``group_by``
  included;
- its **lineage**: a ``LineageRecord`` per item a stage made (see
  ``lineage.py``), written in batches while the run goes and at every FLUSH;
- its **commit log**: a ``Commit`` per item a sink wrote, appended after the
  write. A run continuing in the same store skips the items its sinks
  committed, which makes the sinks idempotent;
- its **checkpoints**: the items passing every ``checkpoint()`` and
  ``tap()``, by item id, with a commit per item (the order) and a mark once
  FLUSH passed it (complete).

``LineageGraph`` (``lineage_graph.py``) answers questions about a run from
them. ``RunStore`` is abstract so that other stores (e.g. a database of a
ScipionWeb server) can replace the files of ``FileRunStore``.
"""

from __future__ import annotations

import abc
import json
import os
import pickle
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..struct import references
from ..struct.source import FileId
from ..utils.files import write_atomic
from ..utils.hashing import item_id
from .lineage import ElementSegment, ItemId, LineageRecord
from .sink_writer import Receipt


class RunStoreMismatchError(ValueError):
    """A run continues in a run store of a run that computed something else."""


@dataclass(frozen=True)
class StageInfo:
    """A stage of a run.

    Attributes:
        label: Its label, unique in the run.
        kind: The kind of its IR node, e.g. ``"IRMap"``.
        stage_fp: The fingerprint of its protocol's code, parameters and
            state; ``None`` outside of protocols.
    """

    label: str
    kind: str
    stage_fp: Optional[str]


@dataclass(frozen=True)
class CheckpointInfo:
    """A checkpoint or tap of a run.

    Attributes:
        label: The label of its stage.
        replay: Whether a rerun replays it (a checkpoint, not a tap).
        items: The items it kept so far.
        complete: Whether FLUSH passed it: it kept every item.
        repeated: The elements that reached it a second time, by their
            element ids; known once complete.
    """

    label: str
    replay: bool
    items: int
    complete: bool
    repeated: int = 0


@dataclass(frozen=True)
class Commit:
    """An item a sink wrote durably.

    Attributes:
        item_id: The id of the item.
        sink: The label of the sink.
        elements: The number of elements of the item's Sets.
        receipt: What the writer reported of the write, for writers that can
            undo the writes after it (``CommittingSinkWriter``).
    """

    item_id: ItemId
    sink: str
    elements: int
    receipt: Optional[Receipt] = None


class RunStore(abc.ABC):
    """Where a run keeps its manifest, its stages and its lineage.

    A run store is passed to the stages of a pipeline, so it must pickle and
    work in every process of the cluster.
    """

    @abc.abstractmethod
    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Store the manifest of the run (JSON values)."""
        ...

    @abc.abstractmethod
    def read_manifest(self) -> Dict[str, Any]:
        """The manifest of the run; empty if none was written."""
        ...

    @abc.abstractmethod
    def write_report(self, report: str) -> None:
        """Store the report of the run's pipeline (``PipelineReport.to_json``)."""
        ...

    @abc.abstractmethod
    def read_report(self) -> Optional[str]:
        """The report of the run's pipeline, if one was written."""
        ...

    @abc.abstractmethod
    def write_stages(self, pipeline: str, stages: Sequence[StageInfo]) -> None:
        """Store the stages of the pipeline ``pipeline`` (an id of it)."""
        ...

    @abc.abstractmethod
    def read_stages(self) -> Dict[str, StageInfo]:
        """The stages of every pipeline of the run, by label."""
        ...

    @abc.abstractmethod
    def append_lineage(self, records: Sequence[LineageRecord]) -> None:
        """Add lineage records."""
        ...

    @abc.abstractmethod
    def read_lineage_table(self) -> pa.Table:
        """Every lineage record of the run, as a table of ``LINEAGE_SCHEMA``."""
        ...

    def read_lineage(self) -> List[LineageRecord]:
        """Every lineage record of the run."""
        return [_record_of(row) for row in self.read_lineage_table().to_pylist()]

    @abc.abstractmethod
    def append_commits(self, commits: Sequence[Commit]) -> None:
        """Add commits, after the writes they record are durable."""
        ...

    @abc.abstractmethod
    def read_commits(self, sink: Optional[str] = None) -> List[Commit]:
        """The commits of the sink labelled ``sink``, or of every sink, in
        the order they were appended."""
        ...

    @abc.abstractmethod
    def begin_checkpoint(self, label: str, replay: bool) -> None:
        """Register the checkpoint (or tap) labelled ``label``."""
        ...

    @abc.abstractmethod
    def store_item(self, label: str, lid: ItemId, item: Any) -> None:
        """Keep ``item``, whose id is ``lid``, at the checkpoint ``label``.

        The checkpoint commits it (``append_commits``) once this returned.
        """
        ...

    @abc.abstractmethod
    def read_item(self, label: str, lid: ItemId) -> Any:
        """The item of id ``lid`` kept at the checkpoint ``label``."""
        ...

    @abc.abstractmethod
    def complete_checkpoint(self, label: str, repeated: int) -> None:
        """Mark the checkpoint ``label`` complete: FLUSH passed it."""
        ...

    @abc.abstractmethod
    def read_checkpoints(self) -> Dict[str, CheckpointInfo]:
        """The checkpoints and taps of the run, by label."""
        ...

    def check_stages(self, stages: Sequence[StageInfo]) -> None:
        """Refuse ``stages`` if a stage of the same label was stored with
        another kind or fingerprint: the items of the run in the store are
        not the items of these stages.

        Raises:
            RunStoreMismatchError: For the stages that changed.
        """
        stored = self.read_stages()
        changed = [
            stage.label
            for stage in stages
            if stage.label in stored and stored[stage.label] != stage
        ]
        if changed:
            raise RunStoreMismatchError(
                f"The run store {self} holds a run whose stages {changed} "
                "computed something else (other code, parameters or kind). "
                "Use a new run store for this run.",
            )


class FileRunStore(RunStore):
    """A run store in a directory, readable while the run goes.

    ::

        <path>/manifest.json
        <path>/report.json
        <path>/stages/<pipeline>.json
        <path>/lineage/<time>-<part>.parquet
        <path>/commits/<time>-<part>.parquet
        <path>/checkpoints/<digest of label>/info.json, complete.json
        <path>/checkpoints/<digest of label>/<item id>.pickle

    Every file is written to a temporary name and renamed, so readers never
    see a partial file. On a cluster, the directory must be shared by every
    node.
    """

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = Path(path).resolve()

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        write_atomic(
            self.path / "manifest.json",
            json.dumps(dict(manifest), indent=2, default=str).encode(),
        )

    def read_manifest(self) -> Dict[str, Any]:
        path = self.path / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def write_report(self, report: str) -> None:
        write_atomic(self.path / "report.json", report.encode())

    def read_report(self) -> Optional[str]:
        path = self.path / "report.json"
        return path.read_text() if path.exists() else None

    def write_stages(self, pipeline: str, stages: Sequence[StageInfo]) -> None:
        write_atomic(
            self.path / "stages" / f"{pipeline}.json",
            json.dumps([asdict(stage) for stage in stages], indent=2).encode(),
        )

    def read_stages(self) -> Dict[str, StageInfo]:
        return {
            stage["label"]: StageInfo(**stage)
            for path in sorted((self.path / "stages").glob("*.json"))
            for stage in json.loads(path.read_text())
        }

    def append_lineage(self, records: Sequence[LineageRecord]) -> None:
        if not records:
            return
        _write_part(self.path / "lineage", records_table(records))

    def read_lineage_table(self) -> pa.Table:
        return _read_parts(self.path / "lineage", LINEAGE_SCHEMA)

    def append_commits(self, commits: Sequence[Commit]) -> None:
        if not commits:
            return
        table = pa.Table.from_arrays(
            [
                pa.array([c.item_id for c in commits], _ID),
                pa.array([c.sink for c in commits], pa.string()),
                pa.array([c.elements for c in commits], pa.int64()),
                pa.array(
                    [
                        None if c.receipt is None else json.dumps(c.receipt)
                        for c in commits
                    ],
                    pa.string(),
                ),
            ],
            schema=COMMIT_SCHEMA,
        )
        _write_part(self.path / "commits", table)

    def read_commits(self, sink: Optional[str] = None) -> List[Commit]:
        table = _read_parts(self.path / "commits", COMMIT_SCHEMA)
        return [
            Commit(
                row["item_id"],
                row["sink"],
                row["elements"],
                None if row["receipt"] is None else json.loads(row["receipt"]),
            )
            for row in table.to_pylist()
            if sink is None or row["sink"] == sink
        ]

    def begin_checkpoint(self, label: str, replay: bool) -> None:
        write_atomic(
            self._checkpoint(label) / "info.json",
            json.dumps({"label": label, "replay": replay}).encode(),
        )

    def store_item(self, label: str, lid: ItemId, item: Any) -> None:
        write_atomic(
            self._checkpoint(label) / f"{lid.hex()}.pickle",
            references.dumps(item, references.PicklingMode.STORE),
        )

    def read_item(self, label: str, lid: ItemId) -> Any:
        return pickle.loads(
            (self._checkpoint(label) / f"{lid.hex()}.pickle").read_bytes()
        )

    def complete_checkpoint(self, label: str, repeated: int) -> None:
        write_atomic(
            self._checkpoint(label) / "complete.json",
            json.dumps({"repeated": repeated}).encode(),
        )

    def read_checkpoints(self) -> Dict[str, CheckpointInfo]:
        counts = Counter(commit.sink for commit in self.read_commits())
        checkpoints = {}
        for info_path in sorted(self.path.glob("checkpoints/*/info.json")):
            info = json.loads(info_path.read_text())
            complete = info_path.with_name("complete.json")
            checkpoints[info["label"]] = CheckpointInfo(
                info["label"],
                info["replay"],
                counts[info["label"]],
                complete.exists(),
                (
                    json.loads(complete.read_text())["repeated"]
                    if complete.exists()
                    else 0
                ),
            )
        return checkpoints

    def _checkpoint(self, label: str) -> Path:
        """The directory of the checkpoint ``label`` (labels hold ``/``)."""
        return self.path / "checkpoints" / item_id(b"checkpoint", label.encode()).hex()

    def __repr__(self) -> str:
        return f"FileRunStore({str(self.path)!r})"


_ID = pa.binary(16)
_SEGMENT = pa.struct(
    [
        ("where", pa.string()),
        ("path", pa.string()),
        ("count", pa.int64()),
        ("namespace", _ID),
        ("start", pa.int64()),
        ("stop", pa.int64()),
        ("positions", pa.binary()),
        ("explicit", pa.binary()),
        ("namespaces", pa.binary()),
        ("indices", pa.binary()),
    ],
)
_FILE = pa.struct(
    [("path", pa.string()), ("size", pa.int64()), ("mtime_ns", pa.int64())],
)
# One row per item a sink wrote; the receipt as JSON.
COMMIT_SCHEMA = pa.schema(
    [
        ("item_id", _ID),
        ("sink", pa.string()),
        ("elements", pa.int64()),
        ("receipt", pa.string()),
    ],
)
# One row per item a stage made.
LINEAGE_SCHEMA = pa.schema(
    [
        ("item_id", _ID),
        ("stage", pa.string()),
        ("kind", pa.string()),
        ("parents", pa.list_(_ID)),
        ("elements", pa.list_(_SEGMENT)),
        ("files", pa.list_(_FILE)),
        ("inherited", pa.bool_()),
    ],
)


def records_table(records: Sequence[LineageRecord]) -> pa.Table:
    """``records`` as a table of ``LINEAGE_SCHEMA``.

    Built column by column: a record of the classes of a classification
    holds a segment per class, and converting every segment to a dict costs
    more than the rest of recording it.
    """
    segments = [segment for record in records for segment in record.elements]
    files = [file for record in records for file in record.files]
    return pa.Table.from_arrays(
        [
            pa.array([r.item_id for r in records], _ID),
            pa.array([r.stage for r in records], pa.string()),
            pa.array([r.kind for r in records], pa.string()),
            _lists(
                [r.parents for r in records],
                pa.array(
                    [p for r in records for p in r.parents],
                    _ID,
                ),
            ),
            _lists(
                [r.elements for r in records],
                pa.StructArray.from_arrays(
                    [
                        pa.array([getattr(s, f.name) for s in segments], f.type)
                        for f in _SEGMENT
                    ],
                    fields=list(_SEGMENT),
                ),
            ),
            _lists(
                [r.files for r in records],
                pa.StructArray.from_arrays(
                    [
                        pa.array([str(f.path) for f in files], pa.string()),
                        pa.array([f.size for f in files], pa.int64()),
                        pa.array([f.mtime_ns for f in files], pa.int64()),
                    ],
                    fields=list(_FILE),
                ),
            ),
            pa.array([r.inherited for r in records], pa.bool_()),
        ],
        schema=LINEAGE_SCHEMA,
    )


def _lists(groups: Sequence[Sequence[Any]], values: pa.Array) -> pa.ListArray:
    """``values`` split into lists of the lengths of ``groups``."""
    offsets = np.cumsum([0, *(len(group) for group in groups)], dtype=np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), values)


def _record_of(row: Mapping[str, Any]) -> LineageRecord:
    return LineageRecord(
        item_id=row["item_id"],
        stage=row["stage"],
        kind=row["kind"],
        parents=tuple(row["parents"]),
        elements=tuple(ElementSegment(**segment) for segment in row["elements"]),
        files=tuple(
            FileId(Path(f["path"]), f["size"], f["mtime_ns"]) for f in row["files"]
        ),
        inherited=row["inherited"],
    )


def _write_part(directory: Path, table: pa.Table) -> None:
    """Write ``table`` as a new part of the parts in ``directory``.

    Parts sort in the order they were written (per process).
    """
    path = directory / f"{time.time_ns():020d}-{uuid.uuid4().hex}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def _read_parts(directory: Path, schema: pa.Schema) -> pa.Table:
    """The parts in ``directory`` as one table, in the order of their names."""
    parts = sorted(directory.glob("*.parquet"))
    return pa.concat_tables(
        [schema.empty_table()] + [pq.read_table(path, schema=schema) for path in parts],
    )
