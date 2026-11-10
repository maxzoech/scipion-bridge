"""Spreading the calls of a map over the executors it leased from its pool."""

import asyncio
from typing import Any, List, Optional, Sequence, Tuple

import pytest

from scipion_bridge.backend.ray.backend import _LeasedExecutors


class FakePool:
    """Hands out new executors, named after their id, recording the leases."""

    def __init__(self) -> None:
        self.leases: List[Tuple[int, ...]] = []

    async def lease(self, held: Sequence[int]) -> Optional[Tuple[int, Any]]:
        self.leases.append(tuple(held))
        executor_id = len(self.leases) - 1
        return executor_id, f"executor {executor_id}"


def _ignore() -> None:
    pass


async def _hold(executors: _LeasedExecutors, calls: int) -> List[Any]:
    """Enter ``calls`` slots at once and return their executors, then leave them."""
    stack = []
    contexts = [executors.slot() for _ in range(calls)]
    for context in contexts:
        stack.append(await context.__aenter__())
    for context in contexts:
        await context.__aexit__(None, None, None)
    return stack


def test_a_call_uses_a_busy_executor_with_room_while_another_is_leased():
    # The lease may wait for an executor to start; the call does not.
    pool = FakePool()
    executors = _LeasedExecutors(pool.lease, replicas=2, capacity=2, on_idle=_ignore)

    used = asyncio.run(_hold(executors, 4))

    assert used == ["executor 0", "executor 0", "executor 1", "executor 1"]
    assert pool.leases == [(), (0,)]


def test_an_idle_executor_is_used_before_leasing_another():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(
            pool.lease, replicas=2, capacity=2, on_idle=_ignore
        )
        async with executors.slot() as first:
            pass
        async with executors.slot() as second:
            pass
        return first, second, pool.leases

    assert asyncio.run(scenario()) == ("executor 0", "executor 0", [()])


def test_a_call_goes_to_the_least_busy_executor():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(
            pool.lease, replicas=2, capacity=2, on_idle=_ignore
        )
        await _hold(executors, 4)  # leases both executors
        async with executors.slot() as first:
            async with executors.slot() as second:
                return first, second

    assert asyncio.run(scenario()) == ("executor 0", "executor 1")


def test_a_full_pool_waits_for_a_call_to_finish():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(
            pool.lease, replicas=1, capacity=1, on_idle=_ignore
        )
        order = []

        async def call(name, seconds):
            async with executors.slot():
                order.append(f"start {name}")
                await asyncio.sleep(seconds)
                order.append(f"end {name}")

        await asyncio.gather(call("a", 0.05), call("b", 0))
        return order, pool.leases

    order, leases = asyncio.run(scenario())

    assert order == ["start a", "end a", "start b", "end b"]
    assert leases == [()]


def test_clearing_forgets_the_leases():
    pool = FakePool()
    executors = _LeasedExecutors(pool.lease, replicas=1, capacity=1, on_idle=_ignore)
    asyncio.run(_hold(executors, 1))

    assert executors.ids == [0]
    executors.clear()
    assert executors.ids == []


def test_a_map_is_idle_when_it_holds_leases_without_calls():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(
            pool.lease, replicas=1, capacity=1, on_idle=_ignore
        )
        before = executors.idle
        async with executors.slot():
            during = executors.idle
        return before, during, executors.idle

    assert asyncio.run(scenario()) == (False, False, True)


def test_a_failed_lease_reaches_a_call_with_no_executor():
    async def fail(held):
        raise RuntimeError("model file missing")

    executors = _LeasedExecutors(fail, replicas=1, capacity=1, on_idle=_ignore)

    with pytest.raises(RuntimeError, match="model file missing"):
        asyncio.run(_hold(executors, 1))


def test_calls_keep_running_while_a_lease_waits():
    async def scenario():
        requested = []

        async def lease(held):
            requested.append(tuple(held))
            match held:
                case []:
                    return 0, "executor 0"
                case _:
                    await asyncio.sleep(3600)  # waits for a GPU of another pool

        executors = _LeasedExecutors(lease, replicas=2, capacity=1, on_idle=_ignore)

        async def call(seconds):
            async with executors.slot() as executor:
                await asyncio.sleep(seconds)
                return executor

        # The second call needs another executor; its lease never completes,
        # but it runs on the first one once that is free.
        used = await asyncio.gather(call(0.05), call(0.0))
        executors.stop()
        return used, requested

    used, requested = asyncio.run(asyncio.wait_for(scenario(), timeout=5))

    assert used == ["executor 0", "executor 0"]
    assert requested == [(), (0,)]


def test_a_lease_after_the_release_is_ignored():
    async def scenario():
        async def released(held):
            return None

        executors = _LeasedExecutors(released, replicas=1, capacity=1, on_idle=_ignore)
        waiting = asyncio.ensure_future(_hold(executors, 1))
        await asyncio.sleep(0.05)
        done = waiting.done()
        waiting.cancel()
        return done, executors.ids

    assert asyncio.run(scenario()) == (False, [])


def test_a_late_lease_reports_the_map_idle():
    # The lease arrives after the call it was for ran on another executor;
    # the map must still return it, or the executor is never released.
    async def scenario():
        reports = []
        arrive = asyncio.Event()

        async def lease(held):
            match held:
                case []:
                    return 0, "executor 0"
                case _:
                    await arrive.wait()
                    return 1, "executor 1"

        executors = _LeasedExecutors(
            lease,
            replicas=2,
            capacity=2,
            on_idle=lambda: reports.append(executors.ids),
        )

        async def call():
            async with executors.slot():
                await asyncio.sleep(0.01)

        await asyncio.gather(call(), call())
        arrive.set()
        await asyncio.sleep(0.01)
        return reports

    reports = asyncio.run(scenario())

    assert reports[-1] == [0, 1]
