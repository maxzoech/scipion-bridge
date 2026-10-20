"""Parallel loading of many input files into a Ray pipeline.

Datasets such as the particles extracted per micrograph come as thousands of
small files. ``RayInputLoader`` actors resolve shards of the files in
parallel, regroup the small per-file sets into chunks of a target size, and
push them directly into a source actor of the pipeline, so that neither the
driver nor the per-item overhead of tiny chunks limits the throughput.
"""

import glob
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, TypeVar

import ray # type: ignore

from ...core.struct.set import Set, concat
from ...core.struct.struct import Struct
from ...core.typed.resolve import ComposedResolver, estimate_optimal_chunk_size
from .container import configure_ray_env
from .profiling import ProfileConfig, make_recorder

T = TypeVar("T")
S = TypeVar("S", bound=Struct)


def expand_pattern(pattern: str) -> List[Path]:
    """The files matching the glob ``pattern`` (``**`` matches directories), sorted."""
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No files match the pattern '{pattern}'.")

    return [Path(match) for match in matches]


def shard(values: Sequence[T], n: int) -> List[Sequence[T]]:
    """Split ``values`` round-robin into at most ``n`` non-empty shards."""
    return [values[i::n] for i in range(min(n, len(values)))]


def _item_nbytes(chunk: Set[S]) -> Optional[int]:
    """Bytes per item of ``chunk``, or ``None`` if it is empty."""
    n = len(chunk)
    if n > 0:
        return chunk.to_arrow().nbytes // n
    else:
        return None
            


def _regroup(chunks: Iterable[Set[S]], size: int) -> Iterator[Set[S]]:
    """Regroup ``chunks`` into sets of ``size`` items; the last may be smaller."""
    buffer: List[Set[S]] = []
    buffered = 0

    for chunk in chunks:
        buffer.append(chunk)
        buffered += len(chunk)
        if buffered < size:
            continue

        merged = concat(buffer)
        full = buffered - buffered % size

        yield from (merged[k : k + size] for k in range(0, full, size))

        rest = merged[full:]
        buffer = [rest] if len(rest) > 0 else []
        buffered = len(rest)

    if buffered > 0:
        yield concat(buffer)


def coalesce(
    chunks: Iterable[Set[S]],
    *,
    chunk_size: Optional[int] = None,
    target_bytes: Optional[int] = None,
) -> Iterator[Set[S]]:
    """Regroup ``chunks`` into sets of ``chunk_size`` items.

    Without ``chunk_size``, the size is the number of items of the first chunk
    that fit into ``target_bytes`` (see ``estimate_optimal_chunk_size``). The
    last set holds the remaining items.
    """
    it = iter(chunks)
    first = next(it, None)
    if first is None:
        return

    size = chunk_size or estimate_optimal_chunk_size(
        _item_nbytes(first),
        target_bytes=target_bytes,
    )
    
    yield from _regroup(itertools.chain([first], it), size)


@dataclass
class LoaderStats:
    """Metrics of a loader.

    Attributes:
        files: Input values the loader resolved.
        chunks: Chunks pushed into the pipeline.
        items: Items in those chunks.
        read_s: Time spent resolving and regrouping the inputs.
        push_s: Time spent pushing chunks. High values mean the pipeline
            is slower than the loaders.
    """

    files: int = 0
    chunks: int = 0
    items: int = 0
    read_s: float = 0.0
    push_s: float = 0.0


@ray.remote
class RayInputLoader:
    """Resolves input values and pushes the chunks into a pipeline source."""

    def __init__(
        self,
        resolver: ComposedResolver,
        parameters: Mapping[str, Any],
        profile: Optional[ProfileConfig],
        label: str,
    ) -> None:
        configure_ray_env(parameters=parameters)
        self._resolver = resolver
        self._recorder = make_recorder(profile, label)

    def run(
        self,
        values: Sequence[Any],
        source: Any,
        chunk_size: Optional[int],
        target_bytes: Optional[int],
    ) -> LoaderStats:
        """Push the chunks of all ``values`` into the ``source`` actor.

        Every push waits until the source buffered the chunk, so the loader
        slows down to the pace of the pipeline.
        """
        stats = LoaderStats(files=len(values))
        resolved = (
            chunk
            for value in values
            for chunk in self._resolver.iter(
                value,
                chunk_size=chunk_size,
                target_bytes=target_bytes,
            )
        )
        chunks = coalesce(resolved, chunk_size=chunk_size, target_bytes=target_bytes)

        while True:
            start = time.perf_counter()
            with self._recorder.span("read", "loader") as details:
                chunk = next(chunks, None)
                details["items"] = 0 if chunk is None else len(chunk)
            stats.read_s += time.perf_counter() - start

            if chunk is None:
                break

            start = time.perf_counter()
            with self._recorder.span("push", "loader", items=len(chunk)):
                ray.get(source.push.remote(chunk))
            stats.push_s += time.perf_counter() - start
            stats.chunks += 1
            stats.items += len(chunk)

        self._recorder.instant("done", "loader", level=logging.DEBUG)
        self._recorder.flush_sync()
        return stats
