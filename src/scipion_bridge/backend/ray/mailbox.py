"""Buffer of the items waiting for a stage of a Ray pipeline.

The mailbox decouples the stages: ``put`` returns as soon as an item is
buffered, so an upstream stage does not wait while the stage is busy. Once
``buffer_size`` items wait, ``put`` waits for room: this is the backpressure
that bounds the backlog of a pipeline. Waiting items stay in memory, either
as they arrived or as references into the object store; beyond
``SpillPolicy.threshold`` of them, further items are written to a
``SpillStore`` and no reference to them is kept, so their memory is freed.
The mailbox makes no Ray calls itself: references are awaited, which in a Ray
actor fetches them.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Hashable, Optional, Tuple

from scipion_bridge.core.streaming.backend import StageStats
from scipion_bridge.core.streaming.node import FlushSignal
from scipion_bridge.core.streaming.spill import SpillStore

from .profiling import NullRecorder, Recorder

_Barrier = asyncio.Future[None]
# A waiting item, the port it arrived on and, for a FLUSH, its barrier.
MailboxEntry = Tuple[Any, int, Optional[_Barrier]]


@dataclass(frozen=True)
class Ref:
    """Reference to an item in the object store, sent between stages.

    Ray resolves only object references passed directly as arguments, so a
    wrapped reference reaches the receiving stage without its data.
    """

    ref: Any


@dataclass(frozen=True)
class Spilled:
    """An item written to a spill store, or being written."""

    task: asyncio.Task[Hashable]
    store: SpillStore[Any]


@dataclass(frozen=True)
class SpillPolicy:
    """Spill items to ``store`` once ``threshold`` items wait in memory."""

    threshold: int
    store: SpillStore[Any]


class Mailbox:
    """FIFO buffer of the items waiting for a stage.

    Every item is held either in memory (as a reference into the object
    store) or in the spill store, decided when it arrives, so the order of the
    items is kept across both. FLUSH is never spilled.

    ``buffer_size`` counts items, whatever their size: in a Ray actor, every
    item in memory pins its object in the object store until it is consumed.
    """

    def __init__(
        self,
        stats: StageStats,
        buffer_size: Optional[int],
        spill: Optional[SpillPolicy],
        recorder: Recorder = NullRecorder(),
    ) -> None:
        """
        Args:
            stats: Metrics of the stage, updated with the buffering metrics.
            buffer_size: Most items waiting at once; ``put`` waits for room
                beyond it. ``None`` buffers without limit.
            spill: When to spill items, and where. ``None`` keeps every item
                in memory.
            recorder: Records the waits for room, fetches and spills.
        """
        self._stats = stats
        self._recorder = recorder
        self._buffer_size = buffer_size
        self._spill = spill
        self._entries: Deque[MailboxEntry] = deque()
        self._in_memory = 0
        self._changed = asyncio.Condition()

    def __len__(self) -> int:
        return len(self._entries)

    async def put(self, item: Any, port: int, barrier: Optional[_Barrier]) -> None:
        """Buffer an item, waiting only while the mailbox is full."""
        async with self._changed:
            if not self._has_room():
                with self._recorder.span(
                    "backpressure",
                    "mailbox",
                    overlapping=True,
                    waiting=len(self._entries),
                ):
                    await self._changed.wait_for(self._has_room)
            self._entries.append((self._admit(item), port, barrier))
            self._stats.buffered_peak = max(
                self._stats.buffered_peak,
                len(self._entries),
            )
            self._changed.notify_all()

    async def get(self) -> MailboxEntry:
        """Take the oldest waiting item; it is resolved with ``resolve``."""
        async with self._changed:
            await self._changed.wait_for(lambda: bool(self._entries))
            entry = self._entries.popleft()
            match entry:
                case (Spilled() | FlushSignal(), _, _):
                    pass
                case _:
                    self._in_memory -= 1
            self._changed.notify_all()
            return entry

    async def resolve(self, item: Any) -> Any:
        """The item of an entry taken by ``get``, read from where it waited."""
        t_start = time.perf_counter()
        with self._recorder.span("fetch", "fetch") as details:
            match item:
                case Spilled(task=task, store=store):
                    details["source"] = "spill"
                    handle = await task
                    value = await asyncio.to_thread(store.get, handle)
                    await asyncio.to_thread(store.discard, handle)
                case Ref(ref=ref):
                    details["source"] = "object_store"
                    value = await ref
                case _:
                    details["source"] = "inline"
                    value = item

        self._stats.fetch_s += time.perf_counter() - t_start
        return value

    def _has_room(self) -> bool:
        return self._buffer_size is None or len(self._entries) < self._buffer_size

    def _admit(self, item: Any) -> Any:
        """The entry of an arriving item: the item itself, or its spilled copy."""
        match (item, self._spill):
            case (FlushSignal(), _):
                return item
            case (_, SpillPolicy(threshold=threshold, store=store)) if (
                self._in_memory >= threshold
            ):
                return Spilled(asyncio.create_task(self._write(item, store)), store)
            case _:
                self._in_memory += 1
                return item

    async def _write(self, item: Any, store: SpillStore[Any]) -> Hashable:
        """Write an item to the spill store; a reference is fetched first."""
        t_start = time.perf_counter()
        with self._recorder.span("spill", "spill", overlapping=True):
            match item:
                case Ref(ref=ref):
                    value = await ref
                case _:
                    value = item

            handle = await asyncio.to_thread(store.put, value)
        self._stats.spilled += 1
        self._stats.spill_s += time.perf_counter() - t_start
        return handle
