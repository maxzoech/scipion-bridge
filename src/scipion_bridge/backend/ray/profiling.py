"""Profiling of Ray pipelines: a live log and a trace of every worker.

Every process of a profiled pipeline (stage actors, map tasks, executors of
compute groups, the driver) records what it does as events through a
``Recorder``. Each event is logged right away on the logger
``scipion_bridge.profile``; Ray forwards the output of the workers to the
driver. The events are also sent in batches to a ``ProfileCollector``, which
appends them to a trace file in the Trace Event Format of Chrome, opened with
https://ui.perfetto.dev or ``chrome://tracing``.

The trace is written while the pipeline runs: the format allows the closing
``]`` to be missing, so the file can be opened at any time and reloaded to see
newer events. It lags at most ``flush_interval_s`` behind the workers.
"""

from __future__ import annotations

import abc
import json
import logging
import itertools
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger("scipion_bridge.profile")

# Longest time, by default, an event waits in a worker before it is sent to
# the collector.
DEFAULT_FLUSH_INTERVAL_S = 1.0


@dataclass(frozen=True)
class ProfileEvent:
    """Something a process of the pipeline did.

    Attributes:
        name: What happened, e.g. ``process`` or ``executor_start``.
        lane: The part of the process it happened in, e.g. the ``compute``
            loop of a stage; a thread of the process in the trace.
        ts_us: Wall clock time of the start in microseconds, comparable
            across the processes of a node.
        dur_us: Duration in microseconds, or ``None`` for an instant event.
        args: Details, e.g. the sequence number of the item.
        overlapping: Whether spans of the lane may overlap, e.g. the calls of
            a map in flight; they are drawn on rows of their own.
    """

    name: str
    lane: str
    ts_us: int
    dur_us: Optional[int]
    args: Mapping[str, Any] = field(default_factory=dict)
    overlapping: bool = False


@dataclass(frozen=True)
class ProcessInfo:
    """The process recording events: a stage actor, a task worker or the driver."""

    label: str
    pid: int
    node: str


def _format(process: ProcessInfo, event: ProfileEvent) -> str:
    """The log line of an event."""
    duration = "" if event.dur_us is None else f" {event.dur_us / 1000:.1f}ms"
    details = "".join(f" {key}={value}" for key, value in event.args.items())
    return f"[{process.label}] {event.lane}/{event.name}{duration}{details}"


class Recorder(abc.ABC):
    """Records the events of a process."""

    @contextmanager
    def span(
        self,
        name: str,
        lane: str,
        *,
        level: int = logging.DEBUG,
        overlapping: bool = False,
        **args: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Record the duration of the enclosed block, which may await.

        Yields the details of the event, to which the block may add. Spans
        that may overlap others of their lane, e.g. concurrent calls, set
        ``overlapping``.
        """
        details = dict(args)
        start_ns = time.time_ns()
        try:
            yield details
        finally:
            end_ns = time.time_ns()
            self.record(
                ProfileEvent(
                    name=name,
                    lane=lane,
                    ts_us=start_ns // 1000,
                    dur_us=(end_ns - start_ns) // 1000,
                    args=details,
                    overlapping=overlapping,
                ),
                level,
            )

    def instant(
        self,
        name: str,
        lane: str,
        *,
        level: int = logging.INFO,
        **args: Any,
    ) -> None:
        """Record an event without duration."""
        self.record(
            ProfileEvent(
                name=name,
                lane=lane,
                ts_us=time.time_ns() // 1000,
                dur_us=None,
                args=args,
            ),
            level,
        )

    @abc.abstractmethod
    def record(self, event: ProfileEvent, level: int) -> None:
        """Log an event and queue it for the collector."""
        ...

    @abc.abstractmethod
    async def flush(self) -> None:
        """Send the queued events and wait until the collector wrote them."""
        ...

    @abc.abstractmethod
    def flush_sync(self) -> None:
        """Like ``flush``, blocking; for processes without an event loop."""
        ...


class NullRecorder(Recorder):
    """Recorder of a pipeline that is not profiled: it records nothing."""

    def record(self, event: ProfileEvent, level: int) -> None:
        pass

    async def flush(self) -> None:
        pass

    def flush_sync(self) -> None:
        pass


class TraceRecorder(Recorder):
    """Logs the events of a process and sends them to the collector in batches.

    A batch is sent once its oldest event waited ``flush_interval_s``, checked
    when an event is recorded. Ray delivers the batches of a process in order.
    Threads may record concurrently (e.g. in a threaded actor).
    """

    def __init__(
        self,
        process: ProcessInfo,
        collector: Any,
        flush_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            process: The process recording the events.
            collector: Handle of the ``ProfileCollector``.
            flush_interval_s: Longest time an event waits before it is sent.
            clock: Time source of the batching, in seconds.
        """
        self._process = process
        self._collector = collector
        self._flush_interval_s = flush_interval_s
        self._clock = clock
        self._events: List[ProfileEvent] = []
        self._first_queued = 0.0
        self._lock = threading.Lock()

    def record(self, event: ProfileEvent, level: int) -> None:
        logger.log(level, "%s", _format(self._process, event))
        with self._lock:
            if not self._events:
                self._first_queued = self._clock()
            self._events.append(event)
            due = self._clock() - self._first_queued >= self._flush_interval_s
        if due:
            self._send()

    async def flush(self) -> None:
        await self._send()

    def flush_sync(self) -> None:
        ray.get(self._send())

    def _send(self) -> Any:
        """Send the queued events; returns the reference of the call."""
        with self._lock:
            events, self._events = self._events, []
        return self._collector.record.remote(self._process, events)


@dataclass(frozen=True)
class ProfileConfig:
    """Profiling of a pipeline, shared by all of its processes.

    Nested pipelines (the children of a ``group_by``) receive the
    configuration of their parent and write into its trace.
    """

    path: Path
    collector: Any
    log_level: int = logging.INFO
    flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S

    def recorder(self, label: str) -> TraceRecorder:
        """The recorder of the process ``label`` this is called in."""
        _configure_logging(self.log_level)
        return TraceRecorder(
            ProcessInfo(
                label=label,
                pid=os.getpid(),
                node=ray.get_runtime_context().get_node_id(),
            ),
            self.collector,
            self.flush_interval_s,
        )


def make_recorder(profile: Optional[ProfileConfig], label: str) -> Recorder:
    """The recorder of the process ``label``; it records nothing without profiling."""
    match profile:
        case None:
            return NullRecorder()
        case ProfileConfig():
            return profile.recorder(label)


@cache
def _configure_logging(level: int) -> None:
    """Print the profiling log of this process at ``level``, once per process.

    Ray workers have no logging configuration, so the log would otherwise be
    dropped below WARNING. The log does not propagate, so that a driver with
    a configured root logger does not print it twice.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


class TraceWriter:
    """Appends events to a trace file in the Trace Event Format of Chrome.

    The file is a JSON array whose closing ``]`` is written by ``close``.
    Every event is preceded by its separator, so the file followed by ``]``
    is valid JSON at any time; trace viewers accept the file without it.

    The processes are numbered by the writer, since the process ids of
    different nodes may collide; the process id is part of their name. Spans
    that may overlap are written as async events, which viewers draw on rows
    of their own.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w")
        self._file.write("[")
        self._separator = "\n"
        self._processes: Dict[Tuple[str, int], int] = {}
        self._lanes: Dict[Tuple[int, str], int] = {}
        self._async_ids = itertools.count(1)

    def write(self, process: ProcessInfo, events: List[ProfileEvent]) -> None:
        """Append the events of a process, and flush the file."""
        records = [
            record
            for event in events
            for record in [
                *self._metadata(process, event),
                *self._events(process, event),
            ]
        ]
        for record in records:
            self._file.write(self._separator + json.dumps(record, default=repr))
            self._separator = ",\n"
        self._file.flush()

    def close(self) -> None:
        """Terminate the JSON array and close the file."""
        self._file.write("\n]\n")
        self._file.close()

    def _metadata(self, process: ProcessInfo, event: ProfileEvent) -> List[Any]:
        """Names of the process and lane of ``event``, if they are new."""
        key = (process.node, process.pid)
        names: List[Any] = []
        if key not in self._processes:
            self._processes[key] = len(self._processes) + 1
            names.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": self._processes[key],
                    "args": {"name": f"{process.label} (pid {process.pid})"},
                },
            )

        pid = self._processes[key]
        if (pid, event.lane) not in self._lanes:
            self._lanes[(pid, event.lane)] = len(self._lanes) + 1
            names.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": self._lanes[(pid, event.lane)],
                    "args": {"name": event.lane},
                },
            )
        return names

    def _events(self, process: ProcessInfo, event: ProfileEvent) -> List[Any]:
        """The trace records of ``event``."""
        pid = self._processes[(process.node, process.pid)]
        common = {
            "name": event.name,
            "pid": pid,
            "tid": self._lanes[(pid, event.lane)],
            "ts": event.ts_us,
            "args": dict(event.args),
        }
        match (event.dur_us, event.overlapping):
            case (None, _):
                return [{**common, "ph": "i", "s": "t"}]
            case (duration, False):
                return [{**common, "ph": "X", "dur": duration}]
            case (duration, True):
                span = {"cat": event.lane, "id": next(self._async_ids)}
                return [
                    {**common, **span, "ph": "b"},
                    {
                        **common,
                        **span,
                        "ph": "e",
                        "ts": event.ts_us + duration,
                        "args": {},
                    },
                ]


@ray.remote(num_cpus=0)
class ProfileCollector:
    """Writes the events of every process of a pipeline to its trace file."""

    def __init__(self, path: Path) -> None:
        self._writer = TraceWriter(path)

    def record(self, process: ProcessInfo, events: List[ProfileEvent]) -> None:
        """Append a batch of events of ``process`` to the trace."""
        self._writer.write(process, events)

    def close(self) -> None:
        """Complete the trace file."""
        self._writer.close()


def start_profile(
    path: Path,
    log_level: int = logging.INFO,
    flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
) -> ProfileConfig:
    """Start the collector of a profiled pipeline, writing the trace to ``path``.

    The collector runs on the node of the caller, so that ``path`` is local
    to it.
    """
    path = path.resolve()
    collector = ProfileCollector.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            ray.get_runtime_context().get_node_id(),
            soft=False,
        ),
    ).remote(path)
    _configure_logging(log_level)
    logger.info("Writing the profiling trace to %s", path)
    return ProfileConfig(
        path=path,
        collector=collector,
        log_level=log_level,
        flush_interval_s=flush_interval_s,
    )
