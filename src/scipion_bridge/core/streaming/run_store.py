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
  ``lineage.py``), buffered per stage and appended at every FLUSH.

``LineageGraph`` (``lineage_graph.py``) answers questions about a run from
them. ``RunStore`` is abstract so that other stores (e.g. a database of a
ScipionWeb server) can replace the files of ``FileRunStore``. Checkpointing
will add the commit log of sinks and snapshots of stages.
"""

from __future__ import annotations

import abc
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..struct.source import FileId
from .lineage import ElementSegment, LineageRecord


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


class FileRunStore(RunStore):
    """A run store in a directory, readable while the run goes.

    ::

        <path>/manifest.json
        <path>/report.json
        <path>/stages/<pipeline>.json
        <path>/lineage/<time>-<part>.parquet

    Every file is written to a temporary name and renamed, so readers never
    see a partial file. On a cluster, the directory must be shared by every
    node.
    """

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = Path(path).resolve()

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        _write_atomic(
            self.path / "manifest.json",
            json.dumps(dict(manifest), indent=2, default=str).encode(),
        )

    def read_manifest(self) -> Dict[str, Any]:
        path = self.path / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def write_report(self, report: str) -> None:
        _write_atomic(self.path / "report.json", report.encode())

    def read_report(self) -> Optional[str]:
        path = self.path / "report.json"
        return path.read_text() if path.exists() else None

    def write_stages(self, pipeline: str, stages: Sequence[StageInfo]) -> None:
        _write_atomic(
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
        table = records_table(records)
        # Parts sort in the order they were written (per process).
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex}.parquet"
        path = self.path / "lineage" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        pq.write_table(table, temporary)
        os.replace(temporary, path)

    def read_lineage_table(self) -> pa.Table:
        parts = sorted((self.path / "lineage").glob("*.parquet"))
        return pa.concat_tables(
            [LINEAGE_SCHEMA.empty_table()]
            + [pq.read_table(path, schema=LINEAGE_SCHEMA) for path in parts],
        )


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


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
