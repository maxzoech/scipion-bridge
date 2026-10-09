"""Tests for the mailbox buffering the items of a Ray stage (in-process).

Completed asyncio futures stand in for references into the object store:
the mailbox only awaits them.
"""

import asyncio
import gc
import weakref

import pytest

from scipion_bridge.backend.ray.mailbox import Mailbox, Ref, SpillPolicy, Spilled
from scipion_bridge.core.streaming.backend import StageStats
from scipion_bridge.core.streaming.node import FLUSH, FlushSignal
from scipion_bridge.core.streaming.spill import PickleSpillStore, SpillStore


class RecordingStore(SpillStore[int]):
    """Spill store keeping the items in memory and recording every call."""

    def __init__(self):
        self.items = {}
        self.calls = []

    def put(self, item):
        handle = len(self.calls)
        self.items[handle] = item
        self.calls.append(("put", handle))
        return handle

    def get(self, handle):
        self.calls.append(("get", handle))
        return self.items[handle]

    def discard(self, handle):
        self.calls.append(("discard", handle))
        del self.items[handle]


def _ref(value):
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return Ref(future)


async def _drain(mailbox, count):
    entries = [await mailbox.get() for _ in range(count)]
    return [await mailbox.resolve(item) for item, _, _ in entries]


def test_items_are_fetched_in_arrival_order():
    async def scenario():
        mailbox = Mailbox(StageStats(), buffer_size=None, spill=None)
        for i in range(5):
            await mailbox.put(_ref(i), 0, None)
        return await _drain(mailbox, 5)

    assert asyncio.run(scenario()) == list(range(5))


def test_put_does_not_wait_with_an_unbounded_buffer():
    async def scenario():
        stats = StageStats()
        mailbox = Mailbox(stats, buffer_size=None, spill=None)
        for i in range(100):
            await asyncio.wait_for(mailbox.put(_ref(i), 0, None), timeout=1.0)
        return stats

    assert asyncio.run(scenario()).buffered_peak == 100


def test_put_waits_for_room_at_the_buffer_size():
    async def scenario():
        mailbox = Mailbox(StageStats(), buffer_size=2, spill=None)
        await mailbox.put(_ref(0), 0, None)
        await mailbox.put(_ref(1), 0, None)

        third = asyncio.create_task(mailbox.put(_ref(2), 0, None))
        await asyncio.sleep(0.01)
        blocked = not third.done()

        await mailbox.get()
        await asyncio.wait_for(third, timeout=1.0)
        return blocked, len(mailbox)

    assert asyncio.run(scenario()) == (True, 2)


def test_items_beyond_the_threshold_are_spilled_in_order():
    async def scenario():
        store = RecordingStore()
        stats = StageStats()
        mailbox = Mailbox(
            stats,
            buffer_size=None,
            spill=SpillPolicy(threshold=2, store=store),
        )
        for i in range(5):
            await mailbox.put(_ref(i), 0, None)
        kinds = [type(item) for item, _, _ in list(mailbox._entries)]
        return await _drain(mailbox, 5), kinds, store, stats

    values, kinds, store, stats = asyncio.run(scenario())

    assert values == list(range(5))
    assert kinds == [Ref, Ref, Spilled, Spilled, Spilled]
    assert stats.spilled == 3
    assert [call for call, _ in store.calls].count("put") == 3
    assert [call for call, _ in store.calls].count("get") == 3
    assert [call for call, _ in store.calls].count("discard") == 3
    assert store.items == {}


class Payload:
    """An item that can be referenced weakly, to observe when it is freed."""

    def __init__(self, value):
        self.value = value


def test_spilled_items_are_no_longer_referenced(tmp_path):
    # In a Ray actor, a referenced item pins its object in the object store.
    async def scenario():
        mailbox = Mailbox(
            StageStats(),
            buffer_size=None,
            spill=SpillPolicy(threshold=1, store=PickleSpillStore(tmp_path)),
        )
        items = [Payload(i) for i in range(3)]
        refs = [weakref.ref(item) for item in items]
        for item in items:
            await mailbox.put(item, 0, None)
        del items, item
        for entry, _, _ in list(mailbox._entries):
            match entry:
                case Spilled(task=task):
                    await task
                case _:
                    pass
        gc.collect()
        alive = [ref() is not None for ref in refs]
        values = [payload.value for payload in await _drain(mailbox, 3)]
        return alive, values

    alive, values = asyncio.run(scenario())

    assert alive == [True, False, False]
    assert values == [0, 1, 2]


def test_items_are_kept_in_memory_again_once_the_backlog_drains():
    async def scenario():
        mailbox = Mailbox(
            StageStats(),
            buffer_size=None,
            spill=SpillPolicy(threshold=1, store=RecordingStore()),
        )
        await mailbox.put(_ref(0), 0, None)
        await mailbox.put(_ref(1), 0, None)
        await _drain(mailbox, 2)
        await mailbox.put(_ref(2), 0, None)
        return [type(item) for item, _, _ in list(mailbox._entries)]

    assert asyncio.run(scenario()) == [Ref]


def test_flush_is_never_spilled_and_keeps_its_barrier():
    async def scenario():
        mailbox = Mailbox(
            StageStats(),
            buffer_size=None,
            spill=SpillPolicy(threshold=1, store=RecordingStore()),
        )
        barrier = asyncio.get_running_loop().create_future()
        await mailbox.put(_ref(0), 0, None)
        await mailbox.put(_ref(1), 0, None)
        await mailbox.put(FLUSH, 1, barrier)
        entries = [await mailbox.get() for _ in range(3)]
        return entries, barrier

    entries, barrier = asyncio.run(scenario())

    assert isinstance(entries[1][0], Spilled)
    assert isinstance(entries[2][0], FlushSignal)
    assert entries[2][1:] == (1, barrier)


def test_failing_spill_store_fails_the_fetch():
    class FailingStore(RecordingStore):
        def put(self, item):
            raise OSError("disk full")

    async def scenario():
        mailbox = Mailbox(
            StageStats(),
            buffer_size=None,
            spill=SpillPolicy(threshold=0, store=FailingStore()),
        )
        await mailbox.put(_ref(0), 0, None)
        item, _, _ = await mailbox.get()
        await mailbox.resolve(item)

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(scenario())
