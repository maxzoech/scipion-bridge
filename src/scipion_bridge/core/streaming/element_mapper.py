"""Parallel per-element processing of collections (``Op.map_element``).

For every incoming collection ``col`` the stage runs::

    for i in range(len(col)):
        col[i] = func(col[i])

with the loop distributed over a thread pool, and forwards the same, modified
collection. Any collection with a fixed length and random access works
(``list``, ``np.ndarray``, ``Set``, ...).

The pools are the state of an ``IRAccumulate`` stage: they are created on the
worker executing the stage and reused for every collection. With
``executor="process"`` the loop still runs in the thread pool and ``func`` is
called in a process pool, started with ``spawn`` by default since forking a
process that has initialized CUDA or JAX corrupts the child.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import multiprocessing
from multiprocessing.pool import Pool, ThreadPool
import os
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import cloudpickle

Executor = Literal["thread", "process"]
StartMethod = Literal["spawn", "forkserver", "fork"]
Workers = Union[int, Literal["auto"]]


@dataclass(frozen=True)
class ElementMapConfig:
    """Execution options of ``Op.map_element``.

    Attributes:
        workers: Number of parallel workers. ``"auto"`` uses one worker per
            element of the collection, capped at the CPUs available to the
            process; after ``.chunk(n)`` this is ``min(n, CPUs)``.
        executor: ``"thread"`` runs ``func`` in the thread pool;
            ``"process"`` runs it in a process pool (for pure-Python work that
            holds the GIL).
        start_method: Start method of the process pool; only valid with
            ``executor="process"``. Defaults to ``"spawn"``.
        chunksize: Number of consecutive indices per pool task. Defaults to 1
            for ``workers="auto"``, else ``ceil(len(col) / (4 * workers))``.
    """

    workers: Workers = "auto"
    executor: Executor = "thread"
    start_method: Optional[StartMethod] = None
    chunksize: Optional[int] = None

    def __post_init__(self) -> None:
        match self.workers:
            case "auto":
                pass
            case int(n) if not isinstance(n, bool) and n > 0:
                pass
            case _:
                raise ValueError(
                    f"workers must be 'auto' or a positive integer, got {self.workers!r}.",
                )

        match (self.executor, self.start_method):
            case ("thread", None):
                pass
            case ("thread", _):
                raise ValueError(
                    "start_method is only supported with executor='process'.",
                )
            case ("process", None | "spawn" | "forkserver" | "fork"):
                pass
            case ("process", _):
                raise ValueError(
                    f"Unknown start_method {self.start_method!r}; "
                    "expected 'spawn', 'forkserver' or 'fork'.",
                )
            case _:
                raise ValueError(
                    f"Unknown executor {self.executor!r}; expected 'thread' or 'process'.",
                )

        match self.chunksize:
            case None:
                pass
            case int(n) if n > 0:
                pass
            case _:
                raise ValueError(
                    f"chunksize must be a positive integer, got {self.chunksize!r}.",
                )


@dataclass
class _ElementPools:
    """Worker pools of a map_element stage; ``size`` is the number of workers."""

    size: int = 0
    threads: Optional[ThreadPool] = None
    processes: Optional[Pool] = None

    def close(self) -> None:
        for pool in (self.threads, self.processes):
            if pool is not None:
                pool.terminate()


def make_element_mapper(
    func: Callable[[Any], Any],
    config: ElementMapConfig,
) -> Tuple[
    Callable[[_ElementPools, Any], Tuple[_ElementPools, List[Any]]],
    Callable[[], _ElementPools],
]:
    """Build the ``accumulate_fn`` and ``initial_state_fn`` of a map_element stage."""

    def initial_state() -> _ElementPools:
        match config.workers:
            case "auto":
                # Sized on the first collection.
                return _ElementPools()
            case int(n):
                return _create_pools(n, func, config)

    def accumulate(state: _ElementPools, col: Any) -> Tuple[_ElementPools, List[Any]]:
        n = len(col)
        match n:
            case 0:
                return (state, [col])
            case _:
                size = _required_size(config, n)
                pools = (
                    state if size <= state.size else _resize(state, size, func, config)
                )
                assert pools.threads is not None
                pools.threads.map(
                    partial(_apply_at, col, func, pools.processes),
                    range(n),
                    _chunksize(config, n, pools.size),
                )
                return (pools, [col])

    return accumulate, initial_state


def _available_cpus() -> int:
    return len(os.sched_getaffinity(0))


def _required_size(config: ElementMapConfig, n: int) -> int:
    match config.workers:
        case "auto":
            return min(n, _available_cpus())
        case int(workers):
            return workers


def _chunksize(config: ElementMapConfig, n: int, workers: int) -> int:
    match (config.chunksize, config.workers):
        case (int(chunksize), _):
            return chunksize
        case (None, "auto"):
            return 1
        case _:
            return max(1, math.ceil(n / (4 * workers)))


def _create_pools(
    size: int,
    func: Callable[[Any], Any],
    config: ElementMapConfig,
) -> _ElementPools:
    match config.executor:
        case "thread":
            processes = None
        case "process":
            context = multiprocessing.get_context(config.start_method or "spawn")
            processes = context.Pool(
                size,
                initializer=_load_func,
                initargs=(cloudpickle.dumps(func),),
            )
    return _ElementPools(size=size, threads=ThreadPool(size), processes=processes)


def _resize(
    state: _ElementPools,
    size: int,
    func: Callable[[Any], Any],
    config: ElementMapConfig,
) -> _ElementPools:
    state.close()
    return _create_pools(size, func, config)


def _apply_at(
    col: Any,
    func: Callable[[Any], Any],
    processes: Optional[Pool],
    index: int,
) -> None:
    match processes:
        case None:
            col[index] = func(col[index])
        case _:
            col[index] = processes.apply(_call_func, (col[index],))


# Function executed by the process pool; loaded once per worker process.
_func: Optional[Callable[[Any], Any]] = None


def _load_func(payload: bytes) -> None:
    global _func
    _func = cloudpickle.loads(payload)


def _call_func(element: Any) -> Any:
    assert _func is not None, "Process pool worker was not initialized."
    return _func(element)
