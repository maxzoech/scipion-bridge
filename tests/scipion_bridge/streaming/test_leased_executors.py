"""Spreading the calls of a map over the executors it leased from its pool."""

import asyncio
from typing import Any, List, Sequence, Tuple

from scipion_bridge.backend.ray.backend import _LeasedExecutors


class FakePool:
    """Hands out new executors, named after their id, recording the leases."""

    def __init__(self) -> None:
        self.leases: List[Tuple[int, ...]] = []

    async def lease(self, held: Sequence[int]) -> Tuple[int, Any]:
        self.leases.append(tuple(held))
        executor_id = len(self.leases) - 1
        return executor_id, f"executor {executor_id}"


async def _hold(executors: _LeasedExecutors, calls: int) -> List[Any]:
    """Enter ``calls`` slots at once and return their executors, then leave them."""
    stack = []
    contexts = [executors.slot() for _ in range(calls)]
    for context in contexts:
        stack.append(await context.__aenter__())
    for context in contexts:
        await context.__aexit__(None, None, None)
    return stack


def test_calls_spread_over_the_executors_before_queueing():
    pool = FakePool()
    executors = _LeasedExecutors(pool.lease, replicas=2, capacity=2)

    used = asyncio.run(_hold(executors, 4))

    assert used == ["executor 0", "executor 1", "executor 0", "executor 1"]
    assert pool.leases == [(), (0,)]


def test_an_idle_executor_is_used_before_leasing_another():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(pool.lease, replicas=2, capacity=2)
        async with executors.slot() as first:
            pass
        async with executors.slot() as second:
            pass
        return first, second, pool.leases

    assert asyncio.run(scenario()) == ("executor 0", "executor 0", [()])


def test_a_call_goes_to_the_least_busy_executor():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(pool.lease, replicas=2, capacity=2)
        await _hold(executors, 4)  # leases both executors
        async with executors.slot() as first:
            async with executors.slot() as second:
                return first, second

    assert asyncio.run(scenario()) == ("executor 0", "executor 1")


def test_a_full_pool_waits_for_a_call_to_finish():
    async def scenario():
        pool = FakePool()
        executors = _LeasedExecutors(pool.lease, replicas=1, capacity=1)
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
    executors = _LeasedExecutors(pool.lease, replicas=1, capacity=1)
    asyncio.run(_hold(executors, 1))

    assert executors.ids == [0]
    executors.clear()
    assert executors.ids == []
