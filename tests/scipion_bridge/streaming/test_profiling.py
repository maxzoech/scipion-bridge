"""Profiling of Ray pipelines: recorders, the trace file and its integration."""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.backend.ray.mailbox import Mailbox, SpillPolicy
from scipion_bridge.backend.ray.profiling import (
    NullRecorder,
    ProcessInfo,
    ProfileEvent,
    Recorder,
    TraceRecorder,
    TraceWriter,
    logger,
)
from scipion_bridge.core.streaming.backend import StageStats
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.spill import PickleSpillStore

_PROCESS = ProcessInfo(label="stage", pid=42, node="node")


class _Done:
    """A completed call of an actor: awaiting it returns at once."""

    def __await__(self):
        return iter(())


class FakeCollector:
    """Stands in for the handle of a ProfileCollector, recording the batches."""

    def __init__(self):
        self.batches = []
        self.record = SimpleNamespace(remote=self._record)

    def _record(self, process, events):
        self.batches.append((process, events))
        return _Done()


class ListRecorder(Recorder):
    """Recorder keeping the events in a list."""

    def __init__(self):
        self.events = []

    def record(self, event, level):
        self.events.append(event)

    async def flush(self):
        pass

    def flush_sync(self):
        pass


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ListHandler(logging.Handler):
    """Log handler keeping the records in a list."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def profile_log():
    """The records of the profiling log, which may not propagate to the root."""
    handler = ListHandler()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(level)


def _recorder(collector, clock=None, flush_interval_s=60.0):
    return TraceRecorder(_PROCESS, collector, flush_interval_s, clock or FakeClock())


# -- Recorders ---------------------------------------------------------------


def test_span_records_duration_and_details_added_inside():
    recorder = ListRecorder()
    with recorder.span("process", "compute", seq=1) as details:
        details["port"] = 0

    (event,) = recorder.events
    assert (event.name, event.lane, dict(event.args)) == (
        "process",
        "compute",
        {"seq": 1, "port": 0},
    )
    assert event.dur_us is not None and event.dur_us >= 0
    assert not event.overlapping


def test_span_is_recorded_when_the_block_raises():
    recorder = ListRecorder()
    with pytest.raises(ValueError):
        with recorder.span("process", "compute"):
            raise ValueError("boom")

    assert [event.name for event in recorder.events] == ["process"]


def test_instant_has_no_duration():
    recorder = ListRecorder()
    recorder.instant("connect", "stage", inputs=2)

    (event,) = recorder.events
    assert event.dur_us is None
    assert dict(event.args) == {"inputs": 2}


def test_null_recorder_records_nothing():
    recorder = NullRecorder()
    with recorder.span("process", "compute") as details:
        details["seq"] = 1
    recorder.instant("connect", "stage")
    asyncio.run(recorder.flush())
    recorder.flush_sync()


def test_events_are_logged_at_their_level(profile_log):
    recorder = _recorder(FakeCollector())
    with recorder.span("process", "compute", seq=3):
        pass
    recorder.instant("connect", "stage")

    assert [
        (r.levelno, r.getMessage().split(" ")[:2]) for r in profile_log.records
    ] == [
        (logging.DEBUG, ["[stage]", "compute/process"]),
        (logging.INFO, ["[stage]", "stage/connect"]),
    ]
    assert "seq=3" in profile_log.records[0].getMessage()


def test_events_wait_for_flush():
    collector = FakeCollector()
    recorder = _recorder(collector)
    recorder.instant("a", "stage")
    recorder.instant("b", "stage")
    assert collector.batches == []

    asyncio.run(recorder.flush())

    ((process, events),) = collector.batches
    assert process == _PROCESS
    assert [event.name for event in events] == ["a", "b"]


def test_events_are_sent_once_the_oldest_waited_the_interval():
    collector = FakeCollector()
    clock = FakeClock()
    recorder = _recorder(collector, clock, flush_interval_s=1.0)

    recorder.instant("a", "stage")
    clock.now = 0.5
    recorder.instant("b", "stage")
    assert collector.batches == []

    clock.now = 1.0
    recorder.instant("c", "stage")
    clock.now = 1.5
    recorder.instant("d", "stage")

    assert [[e.name for e in events] for _, events in collector.batches] == [
        ["a", "b", "c"],
    ]


# -- Trace file --------------------------------------------------------------


def _event(name, lane, ts_us, dur_us=None, overlapping=False, **args):
    return ProfileEvent(
        name=name,
        lane=lane,
        ts_us=ts_us,
        dur_us=dur_us,
        args=args,
        overlapping=overlapping,
    )


def test_trace_is_readable_while_it_is_written(tmp_path):
    path = tmp_path / "trace.json"
    writer = TraceWriter(path)
    writer.write(_PROCESS, [_event("process", "compute", 10, 5, seq=1)])
    writer.write(_PROCESS, [_event("process", "compute", 20, 5, seq=2)])

    # Trace viewers complete the missing ``]`` of a running trace.
    records = json.loads(path.read_text() + "]")
    assert [r["args"].get("seq") for r in records if r["ph"] == "X"] == [1, 2]

    writer.close()
    assert json.loads(path.read_text()) == records


def test_trace_names_processes_and_lanes_once(tmp_path):
    path = tmp_path / "trace.json"
    writer = TraceWriter(path)
    other = ProcessInfo(label="sink", pid=42, node="other-node")
    writer.write(
        _PROCESS,
        [_event("process", "compute", 0, 1), _event("forward", "forward", 1, 1)],
    )
    writer.write(_PROCESS, [_event("process", "compute", 2, 1)])
    writer.write(other, [_event("process", "compute", 3, 1)])
    writer.close()

    records = json.loads(path.read_text())
    processes = {
        r["pid"]: r["args"]["name"] for r in records if r["name"] == "process_name"
    }
    lanes = {
        (r["pid"], r["tid"]): r["args"]["name"]
        for r in records
        if r["name"] == "thread_name"
    }
    # The same pid on another node is another process.
    assert sorted(processes.values()) == ["sink (pid 42)", "stage (pid 42)"]
    assert sorted(lanes.values()) == ["compute", "compute", "forward"]
    assert len(lanes) == 3
    spans = [r for r in records if r["ph"] == "X"]
    assert {(r["pid"], r["tid"]) for r in spans} == set(lanes)


def test_trace_writes_instants_and_overlapping_spans(tmp_path):
    path = tmp_path / "trace.json"
    writer = TraceWriter(path)
    writer.write(
        _PROCESS,
        [
            _event("connect", "stage", 0),
            _event("call", "map:f", 10, 30, overlapping=True, seq=1),
            _event("call", "map:f", 20, 30, overlapping=True, seq=2),
        ],
    )
    writer.close()

    records = [r for r in json.loads(path.read_text()) if r["ph"] != "M"]
    assert records[0]["ph"] == "i"
    calls = [(r["ph"], r["id"], r["ts"]) for r in records[1:]]
    assert calls == [("b", 1, 10), ("e", 1, 40), ("b", 2, 20), ("e", 2, 50)]


# -- Mailbox -----------------------------------------------------------------


def test_mailbox_records_backpressure_fetch_and_spill(tmp_path):
    async def scenario(recorder):
        mailbox = Mailbox(
            StageStats(),
            buffer_size=2,
            spill=SpillPolicy(threshold=1, store=PickleSpillStore(tmp_path)),
            recorder=recorder,
        )
        await mailbox.put(1, 0, None)
        await mailbox.put(2, 0, None)
        waiting = asyncio.create_task(mailbox.put(3, 0, None))
        await asyncio.sleep(0)
        for _ in range(3):
            item, _, _ = await mailbox.get()
            await mailbox.resolve(item)
        await waiting

    recorder = ListRecorder()
    asyncio.run(scenario(recorder))

    names = [(event.name, dict(event.args)) for event in recorder.events]
    assert ("backpressure", {"waiting": 2}) in names
    # Item 3 arrives once item 1 left memory, so only item 2 spills.
    assert [args["source"] for name, args in names if name == "fetch"] == [
        "inline",
        "spill",
        "inline",
    ]
    assert sum(name == "spill" for name, _ in names) == 1
    assert all(
        event.overlapping
        for event in recorder.events
        if event.name in ("backpressure", "spill")
    )


# -- Ray ---------------------------------------------------------------------


def _second(item):
    return item[1]


def _double(value):
    return 2 * value


@pytest.mark.usefixtures("ray_cluster")
def test_profiled_pipeline_writes_the_events_of_every_process(tmp_path):
    path = tmp_path / "trace.json"
    node = (
        Source("x")
        .group_by(0, lambda stream: stream.map(_second).map(_double))
        .unkey()
        .sink(print)
    )
    (sink,) = lower([node])
    pipeline = RayBackend(init_ray=False, profile=path).compile([sink])
    try:
        for value in range(3):
            pipeline.send("x", ("a", value))
        pipeline.flush()

        # Every event up to the FLUSH is in the trace before it is closed.
        running = json.loads(path.read_text() + "]")
    finally:
        pipeline.close()

    records = json.loads(path.read_text())
    assert len(records) == len(running)
    processes = {
        r["pid"]: r["args"]["name"] for r in records if r["name"] == "process_name"
    }
    events = [r for r in records if r["ph"] != "M"]

    def labels_of(name):
        return {processes[r["pid"]] for r in events if r["name"] == name}

    assert any(label.startswith("driver") for label in labels_of("send"))
    assert any(label.startswith("task worker") for label in labels_of("execute"))
    assert len(labels_of("start_child")) == 1
    # The stages of the child of key "a" record their items as well.
    assert any(label.startswith("group_by[a]") for label in labels_of("process"))
    sends = [r for r in events if r["name"] == "send"]
    assert len(sends) == 3
    executes = [r for r in events if r["name"] == "execute"]
    assert len(executes) == 6
