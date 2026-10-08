"""Parallel per-element processing of collections (``Op.map_element``).

For every incoming collection ``col`` the stage runs::

    for i in range(len(col)):
        col[i] = func(col[i])

with the loop distributed over a thread pool, and forwards the same, modified
collection. Any collection with a fixed length and random access works
(``list``, ``np.ndarray``, ``Set``, ...).

The stage is a stateless map: :class:`ElementMapper` only carries the id of
its pools, which cannot be pickled. Every process running the stage (e.g. a
Ray task worker or a long-running executor) creates the pools on its first
collection and keeps them in a cache of the process, reused for every later
collection. With
``executor="process"`` the loop still runs in the thread pool and ``func`` is
called in a process pool, started with ``spawn`` by default since forking a
process that has initialized CUDA or JAX corrupts the child.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import math
import multiprocessing
from multiprocessing.pool import Pool, ThreadPool
import os
import threading
from typing import Any, Callable, Dict, Literal, Optional, Union
import uuid

import cloudpickle

from ..environment.compute import CPUS_ENV_VAR

Executor = Literal["thread", "process"]
StartMethod = Literal["spawn", "forkserver", "fork"]
Workers = Union[int, Literal["auto"]]


@dataclass(frozen=True)
class ElementMapConfig:
    """Execution options of ``Op.map_element``.

    Attributes:
        workers: Number of parallel workers. ``"auto"`` uses one worker per
            element of the collection, capped at the CPUs available to the
            process (the reserved cores if the backend reserved some); after
            ``.chunk(n)`` this is ``min(n, CPUs)``.
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


@dataclass(frozen=True)
class ElementMapper:
    """The function of a map_element stage: ``col -> col``, modified in place.

    Attributes:
        func: Function applied to each element.
        config: Execution options.
        stage: Id of the stage's pools in the cache of each process.
    """

    func: Callable[[Any], Any]
    config: ElementMapConfig
    stage: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __call__(self, col: Any) -> Any:
        n = len(col)
        match n:
            case 0:
                return col
            case _:
                pools = _get_pools(
                    self.stage,
                    _required_size(self.config, n),
                    self.func,
                    self.config,
                )
                assert pools.threads is not None
                pools.threads.map(
                    partial(_apply_at, col, self.func, pools.processes),
                    range(n),
                    _chunksize(self.config, n, pools.size),
                )
                return col


# Pools of the map_element stages that ran in this process, by stage id.
_pools: Dict[str, _ElementPools] = {}
_pools_lock = threading.Lock()


def _get_pools(
    stage: str,
    size: int,
    func: Callable[[Any], Any],
    config: ElementMapConfig,
) -> _ElementPools:
    """Return the pools of ``stage`` with at least ``size`` workers."""
    with _pools_lock:
        pools = _pools.get(stage, _ElementPools())
        match size <= pools.size:
            case True:
                return pools
            case False:
                pools.close()
                _pools[stage] = _create_pools(size, func, config)
                return _pools[stage]


def _available_cpus() -> int:
    """CPU cores reserved for this process by the backend, else all available."""
    match os.environ.get(CPUS_ENV_VAR):
        case None:
            return len(os.sched_getaffinity(0))
        case cpus:
            return max(1, int(float(cpus)))


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
