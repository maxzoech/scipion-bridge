"""Lineage of a run of a million particles: writing, loading and querying.

Generates the lineage records of a run shaped like the 2D classification
pipeline (small stacks, chunks of 256, maps that keep the elements of their
item, classes, a flatten and per-class chunks), writes them to a
``FileRunStore``, and times ``LineageGraph`` on them::

    python benchmarks/bench_lineage.py --particles 1000000 --run-dir /tmp/run

The numbers depend on the disk and the CPU; the shape of the run is what
matters: records scale with chunks, element ids with particles.
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
from typing import List

import numpy as np

from scipion_bridge.core.streaming import (
    FileRunStore,
    LineageGraph,
    LineageRecord,
    StageInfo,
)
from scipion_bridge.core.streaming.lineage import ElementSegment
from scipion_bridge.core.struct.id_column import IdColumn
from scipion_bridge.core.struct.source import FileId
from scipion_bridge.core.utils.hashing import item_id

CHUNK = 256
CLASSES = 50
PER_STACK = 5
CLASS_CHUNK = 500


def _segments(column: IdColumn, where: str = "", path: str = "") -> tuple:
    return tuple(ElementSegment.of(where, path, s) for s in column.segments)


def generate(store: FileRunStore, particles: int, seed: int = 0) -> List[bytes]:
    """Write the records of a run of ``particles``; returns its outputs."""
    rng = np.random.default_rng(seed)
    stacks = [
        FileId(Path(f"/data/stacks/stack_{k:06d}.mrcs"), 1000, k)
        for k in range(particles // PER_STACK)
    ]
    namespaces = [f.namespace for f in stacks]
    records: List[LineageRecord] = []
    outputs: List[bytes] = []
    # Members waiting per class for the per-class chunk: (column, parent ids).
    waiting = {k: ([], []) for k in range(CLASSES)}
    labels = [
        "0:source(particles)",
        "1:map(SetSamplingRate)",
        "2:map(verify)",
        "3:map(Select)",
        "4:chunk(256) → astype",
        "5:map(raw_input)",
        "6:map(forward)",
        "7:map(build_embeddings)",
        "8:map(verify)",
        "14:map(classify)",
        "18:flatten",
        "19:group_by[w]/1:map(extract)",
        "19:group_by[w]/3:chunk(500)",
        "19:group_by[w]/7:map(subdivide)",
        "21:map(verify_outputs)",
    ]
    store.write_stages("bench", [StageInfo(label, "IR", label) for label in labels])

    def emit(record: LineageRecord) -> bytes:
        records.append(record)
        if len(records) >= 5000:
            store.append_lineage(records)
            records.clear()
        return record.item_id

    def chain(parent: bytes, stage: str) -> bytes:
        return emit(
            LineageRecord(
                item_id(parent, stage.encode()),
                stage,
                "map",
                (parent,),
                (),
                inherited=True,
            ),
        )

    for chunk in range(particles // CHUNK):
        rows = np.arange(chunk * CHUNK, (chunk + 1) * CHUNK)
        column = IdColumn.concat(
            [
                IdColumn.of(namespaces[stack], range(start, stop))
                for stack, start, stop in _runs(rows)
            ],
        )
        files = tuple(stacks[s] for s, _, _ in _runs(rows))
        lid = emit(
            LineageRecord(
                item_id(b"source", chunk.to_bytes(8, "little")),
                labels[0],
                "source",
                (),
                _segments(column),
                files,
            ),
        )
        for stage in labels[1:4]:
            lid = chain(lid, stage)
        lid = emit(
            LineageRecord(
                item_id(lid, b"chunk"),
                labels[4],
                "accumulate",
                (lid,),
                _segments(column),
            ),
        )
        for stage in labels[5:9]:
            lid = chain(lid, stage)
        assignment = rng.integers(0, CLASSES, size=CHUNK)
        classes = {
            k: column.narrow(np.flatnonzero(assignment == k)) for k in range(CLASSES)
        }
        classified = emit(
            LineageRecord(
                item_id(lid, b"classify"),
                labels[9],
                "map",
                (lid,),
                tuple(
                    s
                    for k, members in classes.items()
                    for s in _segments(members, f"['classes'][{k}]", "particles")
                ),
            ),
        )
        for k, members in classes.items():
            flat = emit(
                LineageRecord(
                    item_id(classified, k.to_bytes(4, "little")),
                    labels[10],
                    "accumulate",
                    (classified,),
                    _segments(members, "", "particles"),
                ),
            )
            extracted = emit(
                LineageRecord(
                    item_id(flat, b"extract"),
                    labels[11],
                    "map",
                    (flat,),
                    _segments(members),
                ),
            )
            columns, parents = waiting[k]
            columns.append(members)
            parents.append(extracted)
            if sum(len(c) for c in columns) >= CLASS_CHUNK:
                merged = IdColumn.concat(columns)
                class_chunk = emit(
                    LineageRecord(
                        item_id(extracted, b"chunk500"),
                        labels[12],
                        "accumulate",
                        tuple(sorted(parents)),
                        _segments(merged),
                    ),
                )
                waiting[k] = ([], [])
                subdivided = emit(
                    LineageRecord(
                        item_id(class_chunk, b"subdivide"),
                        labels[13],
                        "map",
                        (class_chunk,),
                        _segments(merged, f"['classes'][{k}]", "particles"),
                    ),
                )
                outputs.append(chain(subdivided, labels[14]))
    store.append_lineage(records)
    return outputs


def bench_recording(store: FileRunStore) -> None:
    """The time to record an item, per item and stage, for items shaped like
    those of the 2D pipeline: a chunk of 1000 particles of 200 stacks of 5,
    and its 15 classes."""
    import scipion_bridge as B
    from scipion_bridge.core.streaming.lineage import (
        call_elements,
        element_segments,
        fill_created,
    )
    from scipion_bridge.core.streaming.run_store import records_table
    from scipion_bridge.core.struct import concat, identity
    from scipion_bridge.single_particle import Class2D, Particle

    parts = []
    for stack in range(200):
        part = B.Set[Particle](capacity=PER_STACK)
        part["sampling_rate"] = np.ones((PER_STACK, 1))
        namespace = item_id(b"stack", stack.to_bytes(4, "little"))
        identity.set_ids(part, IdColumn.of(namespace, range(PER_STACK)))
        parts.append(part)
    chunk = {"particles": concat(parts)}
    labels = np.random.default_rng(0).integers(0, 15, 1000)
    classes = {
        "classes": B.Collection[Class2D](
            size=15,
            items={
                k: Class2D(particles=chunk["particles"][labels == k], class_id=k)
                for k in range(15)
            },
        ),
    }

    def per_item(label: str, fn, n: int = 200) -> None:
        fn()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        print(f"{label:<48} {(time.perf_counter() - start) / n * 1e6:9.1f} µs")

    per_item("element_segments(chunk)", lambda: element_segments(chunk))
    per_item("element_segments(15 classes)", lambda: element_segments(classes))
    per_item("call_elements(chunk -> chunk)", lambda: call_elements(chunk, chunk))
    per_item(
        "call_elements(chunk -> 15 classes)", lambda: call_elements(chunk, classes)
    )
    per_item(
        "fill_created(15 classes), always on", lambda: fill_created(classes, b"i" * 16)
    )
    elements = element_segments(classes)
    records = [
        LineageRecord(
            item_id(k.to_bytes(4, "little")), "s", "map", (b"p" * 16,), elements
        )
        for k in range(2000)
    ]
    start = time.perf_counter()
    for _ in range(3):
        records_table(records)
    per_record = (time.perf_counter() - start) / 3 / len(records) * 1e6
    print(f"{'records_table of 2000 records, per record':<48} {per_record:9.1f} µs")
    start = time.perf_counter()
    for _ in range(3):
        store.append_lineage(records)
    per_record = (time.perf_counter() - start) / 3 / len(records) * 1e6
    print(f"{'append_lineage of 2000 records, per record':<48} {per_record:9.1f} µs")


def _runs(rows: np.ndarray):
    """The (stack, first row, end row) runs of consecutive ``rows``."""
    stacks = rows // PER_STACK
    for stack in np.unique(stacks):
        in_stack = rows[stacks == stack] % PER_STACK
        yield int(stack), int(in_stack.min()), int(in_stack.max()) + 1


def _timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    print(f"{label:<40} {time.perf_counter() - start:8.3f} s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Query the run in --run-dir instead of generating it. Generate "
        "and query in two processes: loading in the process that generated "
        "the run measures its garbage collector, not the loading.",
    )
    parser.add_argument(
        "--recording",
        action="store_true",
        help="Time the recording of single items instead (parts go to --run-dir).",
    )
    args = parser.parse_args()

    store = FileRunStore(args.run_dir)
    if args.recording:
        bench_recording(store)
        return
    if not args.reuse:
        _timed("generate and write", lambda: generate(store, args.particles))
        return
    graph0 = LineageGraph.of(store)
    outputs = [
        lid for lid in graph0.leaves() if "verify_outputs" in graph0.stage_of(lid)
    ]
    del graph0
    size = sum(p.stat().st_size for p in args.run_dir.rglob("*") if p.is_file())
    print(f"{'run directory':<40} {size / 2**20:8.1f} MiB")

    graph = _timed("load", lambda: LineageGraph.of(store))
    print(f"{'records':<40} {len(graph):8d}")
    _timed("stage summaries", graph.stage_summaries)
    _timed("leaves", graph.leaves)
    ancestry = _timed("ancestry of an output", lambda: graph.ancestry(outputs[-1]))
    print(
        f"{'  items, sources, files':<40} {len(ancestry.item_ids)}, {len(ancestry.sources)}, {len(ancestry.files)}"
    )
    member = graph.elements(outputs[-1])[0]
    namespaces = np.frombuffer(
        member.namespaces or member.namespace or b"", dtype="V16"
    )
    namespace = namespaces[0].tobytes()
    position = int(member.position_array()[0])
    _timed(
        "trace an element (cold index)",
        lambda: graph.trace_element(namespace, position),
    )
    holders = _timed(
        "holders of an input particle", lambda: graph.holders(namespace, position)
    )
    print(f"{'  holders':<40} {len(holders):8d}")
    _timed("holders again (warm)", lambda: graph.holders(namespace, position))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10
    print(f"{'peak memory':<40} {rss:8.0f} MiB")


if __name__ == "__main__":
    main()
