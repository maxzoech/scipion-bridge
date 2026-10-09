"""Scheduling protocol stages by their compute resources on Ray.

The test cluster has 2 CPUs and 2 logical GPUs (see conftest).
"""

import json
import logging
import os
import random
import subprocess
import sys
import time
from typing import Any

from numpy._core.multiarray import (
    _set_madvise_hugepage,  # pyright: ignore[reportAttributeAccessIssue]
)
import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray import backend
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.environment.compute import (
    CPUS_ENV_VAR,
    NUMPY_HUGEPAGE_ENV_VAR,
)
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter

# Timing assertions: run on one xdist worker, after one another.
pytestmark = [
    pytest.mark.usefixtures("ray_cluster"),
    pytest.mark.xdist_group("timing"),
]

CLUSTER_GPUS = 2
CLUSTER_CPUS = 2


@ray.remote
class Recorder:
    """Records the calls of protocol functions running in other processes."""

    def __init__(self):
        self.calls = []
        self.builds = 0

    def record(self, call):
        self.calls.append(call)

    def build(self):
        self.builds += 1

    def get_calls(self):
        return self.calls

    def get_builds(self):
        return self.builds


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def get(self):
        return self.items


def _record(recorder, label, seconds=0.0):
    start = time.time()
    time.sleep(seconds)
    call = {
        "label": label,
        "start": start,
        "end": time.time(),
        "gpus": ray.get_gpu_ids(),
        "pid": os.getpid(),
    }
    ray.get(recorder.record.remote(call))


def _max_overlap(calls):
    events = sorted(
        [(call["start"], 1) for call in calls] + [(call["end"], -1) for call in calls],
    )
    running = peak = 0
    for _, step in events:
        running += step
        peak = max(peak, running)
    return peak


def _compile(protocol, collector, parameters=None):
    writer = CallbackSinkWriter(lambda item: ray.get(collector.append.remote(item)))
    (sink,) = lower([protocol.get_pipeline().write_to(writer)])
    return RayBackend(init_ray=False, parameters=parameters or {}).compile([sink])


def _wait_for_free_gpus():
    deadline = time.monotonic() + 30
    while (
        ray.available_resources().get("GPU", 0) < CLUSTER_GPUS
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)
    return ray.available_resources().get("GPU", 0)


def _key(item):
    return item % 6


# -- Ephemeral ---------------------------------------------------------------


# No reserved CPUs (the default), so only the GPUs limit how many calls run.
@B.resources(gpus=1)
class PerKeyWork(B.Protocol):
    items: B.Input[int] = B.Input()
    offset: B.Field[int] = B.Field(default=0)
    recorder: Any

    def outputs(self):
        return {"out": int}

    def steps(self):
        return (
            self.items.group_by(_key, lambda stream: stream.map(self._work))
            .unkey()
            .map(self._output)
        )

    def _work(self, item):
        # The first item of every key only creates the child pipeline.
        match item < 6:
            case True:
                _record(self.recorder, "start")
            case False:
                _record(self.recorder, "work", seconds=0.5)
        return item + self.offset.value

    def _output(self, keyed):
        return {"out": keyed.value}


def test_ephemeral_calls_are_queued_on_the_gpus():
    protocol = PerKeyWork()
    protocol.recorder = Recorder.remote()
    collector = Collector.remote()
    pipeline = _compile(protocol, collector, parameters={"offset": 100})

    try:
        # Child pipelines start one after another; start all before measuring.
        for item in range(6):
            pipeline.send("items", item)
        pipeline.flush()
        for item in range(6, 12):
            pipeline.send("items", item)
        pipeline.flush()

        calls = [
            call
            for call in ray.get(protocol.recorder.get_calls.remote())
            if call["label"] == "work"
        ]
        assert len(calls) == 6
        assert _max_overlap(calls) == CLUSTER_GPUS
        assert all(len(call["gpus"]) == 1 for call in calls)
        # Field.value is served inside the tasks.
        results = ray.get(collector.get.remote())
        assert sorted(result["out"] for result in results) == list(range(100, 112))
    finally:
        pipeline.close()


# -- Long-running ------------------------------------------------------------


def _build_model(protocol):
    ray.get(protocol.recorder.build.remote())
    return 10


@B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
class Inference(B.Protocol):
    items: B.Input[int] = B.Input()
    model: B.Resource[int] = B.Resource(builder=_build_model)
    recorder: Any

    def outputs(self):
        return {"out": int}

    def steps(self):
        return self.items.map(self._forward).map(self._output)

    def _forward(self, item):
        _record(self.recorder, "forward")
        return item + self.model

    def _output(self, item):
        _record(self.recorder, "output")
        return {"out": item}


def test_long_running_stages_share_one_process():
    protocol = Inference()
    protocol.recorder = Recorder.remote()
    collector = Collector.remote()
    pipeline = _compile(protocol, collector)

    try:
        for item in range(3):
            pipeline.send("items", item)
        pipeline.flush()

        calls = ray.get(protocol.recorder.get_calls.remote())
        assert len(calls) == 6
        assert len({call["pid"] for call in calls}) == 1
        assert len({tuple(call["gpus"]) for call in calls}) == 1
        assert len(calls[0]["gpus"]) == 1
        assert ray.get(protocol.recorder.get_builds.remote()) == 1
        results = ray.get(collector.get.remote())
        assert [result["out"] for result in results] == [10, 11, 12]
    finally:
        pipeline.close()


def test_long_running_resources_are_released_on_flush():
    protocol = Inference()
    protocol.recorder = Recorder.remote()
    pipeline = _compile(protocol, Collector.remote())

    try:
        pipeline.send("items", 0)
        pipeline.flush()
        assert _wait_for_free_gpus() == CLUSTER_GPUS

        # Items after a flush acquire the resources again, in a new process.
        pipeline.send("items", 1)
        pipeline.flush()

        calls = ray.get(protocol.recorder.get_calls.remote())
        first, second = calls[0]["pid"], calls[-1]["pid"]
        assert first != second
        assert ray.get(protocol.recorder.get_builds.remote()) == 2
        assert _wait_for_free_gpus() == CLUSTER_GPUS
    finally:
        pipeline.close()


def _restore_slowly(value):
    time.sleep(0.2)  # Stands in for deserializing a large chunk.
    return SlowItem(value)


class SlowItem:
    """Item that takes a while to deserialize in the receiving process."""

    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return _restore_slowly, (self.value,)


@B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
class SlowInputInference(B.Protocol):
    items: B.Input[SlowItem] = B.Input()
    recorder: Any

    def outputs(self):
        return {"out": int}

    def steps(self):
        return self.items.map(self._forward)

    def _forward(self, item):
        _record(self.recorder, "forward", seconds=0.4)
        return {"out": item.value}


def test_long_running_calls_overlap_with_the_transfer_of_the_next():
    protocol = SlowInputInference()
    protocol.recorder = Recorder.remote()
    collector = Collector.remote()
    pipeline = _compile(protocol, collector)

    try:
        for item in range(6):
            pipeline.send("items", SlowItem(item))
        pipeline.flush()

        calls = sorted(
            ray.get(protocol.recorder.get_calls.remote()),
            key=lambda call: call["start"],
        )
        # The next item is deserialized while the current one computes, so
        # the function runs back to back, one call at a time. The first gap
        # also waits for the second item to reach the executor.
        gaps = [
            after["start"] - before["end"] for before, after in zip(calls, calls[1:])
        ]
        assert _max_overlap(calls) == 1
        assert sum(gaps[1:]) / len(gaps[1:]) < 0.1
        results = ray.get(collector.get.remote())
        assert [result["out"] for result in results] == list(range(6))
    finally:
        pipeline.close()


@B.resources(gpus=CLUSTER_GPUS + 1, task=B.TaskType.LONG_RUNNING)
class TooLarge(B.Protocol):
    items: B.Input[int] = B.Input()

    def outputs(self):
        return {"out": int}

    def steps(self):
        return self.items.map(lambda item: {"out": item})


def test_requesting_more_than_the_cluster_has_raises():
    with pytest.raises(ValueError, match="require"):
        _compile(TooLarge(), Collector.remote())


# -- map_element -------------------------------------------------------------


def _double(x):
    return 2 * x


def _preprocess_then_sum(task):
    @B.resources(gpus=1, task=task)
    class Preprocess(B.Protocol):
        items: B.Input[list] = B.Input()

        def outputs(self):
            return {"out": int}

        def steps(self):
            return self.items.map_element(_double).map(self._sum)

        def _sum(self, col):
            return {"out": sum(col)}

    return Preprocess()


@pytest.mark.parametrize("task", list(B.TaskType))
def test_map_element_runs_with_compute_resources(task):
    # The thread pools of map_element cannot be pickled, so they must not be
    # sent along with the calls to the task or executor.
    collector = Collector.remote()
    pipeline = _compile(_preprocess_then_sum(task), collector)

    try:
        pipeline.send("items", [1, 2, 3])
        pipeline.send("items", [4, 5])
        pipeline.flush()

        results = ray.get(collector.get.remote())
        assert [result["out"] for result in results] == [12, 18]
    finally:
        pipeline.close()


# -- SHARED resources --------------------------------------------------------


class Shared(B.Protocol):
    items: B.Input[int] = B.Input()
    constants: B.Resource[dict] = B.Resource(
        builder=lambda protocol: ray.get_runtime_context().get_assigned_resources(),
        scope=B.ResourceScope.SHARED,
    )

    def outputs(self):
        return {"out": dict}

    def steps(self):
        return self.items.map(lambda item: {"out": self.constants})


def test_shared_resources_are_built_with_a_share_of_the_cpus():
    collector = Collector.remote()
    pipeline = _compile(Shared(), collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        (result,) = ray.get(collector.get.remote())
        # 35% of the 2 CPUs of the test cluster, but at least 1.
        assert result["out"] == {"CPU": 1.0}
    finally:
        pipeline.close()


# -- CPUs --------------------------------------------------------------------


def _per_key_sleep(**resources):
    @B.resources(**resources)
    class PerKeySleep(B.Protocol):
        items: B.Input[int] = B.Input()
        recorder: Any

        def outputs(self):
            return {"out": int}

        def steps(self):
            return (
                self.items.group_by(_key, lambda stream: stream.map(self._work))
                .unkey()
                .map(self._output)
            )

        def _work(self, item):
            # The first item of every key only creates the child pipeline.
            match item < 6:
                case True:
                    _record(self.recorder, "start")
                case False:
                    # Long enough for Ray to start a worker process per call.
                    _record(self.recorder, "work", seconds=1.5)
            return item

        def _output(self, keyed):
            return {"out": keyed.value}

    return PerKeySleep()


def _overlapping_work(**resources):
    protocol = _per_key_sleep(**resources)
    protocol.recorder = Recorder.remote()
    pipeline = _compile(protocol, Collector.remote())

    try:
        for item in range(12):
            pipeline.send("items", item)
            # Child pipelines start one after another; start all first.
            if item == 5:
                pipeline.flush()
        pipeline.flush()

        calls = [
            call
            for call in ray.get(protocol.recorder.get_calls.remote())
            if call["label"] == "work"
        ]
        assert len(calls) == 6
        return calls
    finally:
        pipeline.close()


def test_calls_without_reserved_cpus_are_not_limited_by_the_cpus():
    # Ray starts worker processes a few at a time, so not all 6 calls need
    # to overlap; more than the 2 CPUs do.
    assert _max_overlap(_overlapping_work(cpus=None)) > CLUSTER_CPUS


def test_reserved_cpus_limit_the_calls_running_at_once():
    assert _max_overlap(_overlapping_work(cpus=1)) == CLUSTER_CPUS


def _numpy_hugepages():
    """NumPy's hugepage setting of this process: the variable and the hint."""
    # _set_madvise_hugepage returns the previous state of the hint.
    hint = _set_madvise_hugepage(False)
    _set_madvise_hugepage(hint)
    return {"hugepage": os.environ.get(NUMPY_HUGEPAGE_ENV_VAR), "hint": hint}


def _environment(item):
    return {
        "out": {
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "cpus": os.environ.get(CPUS_ENV_VAR),
            "xla": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "assigned": ray.get_runtime_context().get_assigned_resources(),
            **_numpy_hugepages(),
        },
    }


def _reports_environment(**resources):
    @B.resources(**resources)
    class ReportsEnvironment(B.Protocol):
        items: B.Input[int] = B.Input()

        def outputs(self):
            return {"out": dict}

        def steps(self):
            return self.items.map(_environment)

    return ReportsEnvironment()


def test_calls_without_reserved_cpus_use_all_cores():
    collector = Collector.remote()
    pipeline = _compile(_reports_environment(cpus=None), collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        ((_, result),) = [item.popitem() for item in ray.get(collector.get.remote())]
        assert result["omp"] == str(CLUSTER_CPUS)
        assert result["cpus"] is None
        assert "CPU" not in result["assigned"]
        assert (result["hugepage"], result["hint"]) == ("0", False)
    finally:
        pipeline.close()


def test_calls_with_reserved_cpus_learn_their_number():
    collector = Collector.remote()
    pipeline = _compile(_reports_environment(cpus=1), collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        ((_, result),) = [item.popitem() for item in ray.get(collector.get.remote())]
        assert result["omp"] == "1"
        assert result["cpus"] == "1"
        assert result["assigned"]["CPU"] == 1
        assert (result["hugepage"], result["hint"]) == ("0", False)
    finally:
        pipeline.close()


def _process(item):
    return {
        "out": {
            "pid": os.getpid(),
            "gpus": ray.get_gpu_ids(),
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _reports_process(**resources):
    @B.resources(**resources)
    class ReportsProcess(B.Protocol):
        items: B.Input[int] = B.Input()

        def outputs(self):
            return {"out": dict}

        def steps(self):
            return self.items.map(_process)

    return ReportsProcess()


def _processes_of_consecutive_calls(protocol, calls=4):
    collector = Collector.remote()
    pipeline = _compile(protocol, collector)

    try:
        for item in range(calls):
            pipeline.send("items", item)
            pipeline.flush()

        return [result["out"] for result in ray.get(collector.get.remote())]
    finally:
        pipeline.close()


@pytest.mark.parametrize("gpus", [1, 0.5])
def test_every_gpu_call_runs_in_a_process_of_its_own(gpus):
    # CUDA binds a process to its first GPU; a reused worker would keep it.
    results = _processes_of_consecutive_calls(_reports_process(gpus=gpus))

    assert len({result["pid"] for result in results}) == len(results)
    assert all(
        result["cuda"] == ",".join(map(str, result["gpus"])) for result in results
    )


def test_cpu_calls_reuse_worker_processes():
    results = _processes_of_consecutive_calls(_reports_process())

    assert len({result["pid"] for result in results}) < len(results)


# -- cpu_only ----------------------------------------------------------------


@ray.remote(num_gpus=CLUSTER_GPUS)
class GpuBlocker:
    """Holds all GPUs of the cluster."""

    def ready(self):
        return True


@B.resources(gpus=1)
class Bookkeeping(B.Protocol):
    items: B.Input[int] = B.Input()

    def outputs(self):
        return {"out": list}

    def steps(self):
        return self.items.map(self._gpus, cpu_only=True)

    def _gpus(self, item):
        return {"out": ray.get_gpu_ids()}


def test_cpu_only_maps_do_not_wait_for_a_gpu():
    blocker = GpuBlocker.remote()
    ray.get(blocker.ready.remote())
    collector = Collector.remote()
    pipeline = _compile(Bookkeeping(), collector)

    try:
        pipeline.send("items", 0)

        deadline = time.monotonic() + 30
        while not ray.get(collector.get.remote()) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert ray.get(collector.get.remote()) == [{"out": []}]
    finally:
        ray.kill(blocker)
        pipeline.close()


@B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
class LongRunningWithBookkeeping(B.Protocol):
    items: B.Input[int] = B.Input()
    recorder: Any

    def outputs(self):
        return {"out": int}

    def steps(self):
        return self.items.map(self._forward).map(self._output, cpu_only=True)

    def _forward(self, item):
        _record(self.recorder, "forward")
        return item

    def _output(self, item):
        _record(self.recorder, "output")
        return {"out": item}


def test_cpu_only_maps_run_outside_the_long_running_executor():
    protocol = LongRunningWithBookkeeping()
    protocol.recorder = Recorder.remote()
    pipeline = _compile(protocol, Collector.remote())

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        forward, output = sorted(
            ray.get(protocol.recorder.get_calls.remote()),
            key=lambda call: call["label"],
        )
        assert len(forward["gpus"]) == 1
        assert output["gpus"] == []
        assert output["pid"] != forward["pid"]
        assert _wait_for_free_gpus() == CLUSTER_GPUS
    finally:
        pipeline.close()


# -- Maps as tasks -----------------------------------------------------------


def _sleep_randomly(item):
    time.sleep(random.uniform(0, 0.05))
    return item


def _chain(source):
    return source.map(_sleep_randomly).map(_sleep_randomly).map(_sleep_randomly)


def test_maps_keep_the_order_of_the_items():
    collector = Collector.remote()
    writer = CallbackSinkWriter(lambda item: ray.get(collector.append.remote(item)))
    (sink,) = lower([_chain(Source("x")).write_to(writer)])
    pipeline = RayBackend(init_ray=False, queue_size=4).compile([sink])

    try:
        for item in range(20):
            pipeline.send("x", item)
        pipeline.flush()

        assert ray.get(collector.get.remote()) == list(range(20))
    finally:
        pipeline.close()


def test_maps_have_no_actors_and_report_stats():
    (sink,) = lower([_chain(Source("x")).sink(lambda item: None)])
    existing = set(ray.util.list_named_actors())
    pipeline = RayBackend(init_ray=False).compile([sink])

    try:
        created = set(ray.util.list_named_actors()) - existing
        assert sorted(name.split(":", 1)[1] for name in created) == [
            "0:source(x)",
            "4:sink(CallbackSinkWriter)",
        ]

        for item in range(3):
            pipeline.send("x", item)
        pipeline.flush()

        stats = pipeline.stats()
        assert list(stats) == [
            "0:source(x)",
            "1:map(_sleep_randomly)",
            "2:map(_sleep_randomly)",
            "3:map(_sleep_randomly)",
            "4:sink(CallbackSinkWriter)",
        ]
        assert all(stats[f"{i}:map(_sleep_randomly)"].items_in == 3 for i in (1, 2, 3))
    finally:
        pipeline.close()


# -- min_vram ----------------------------------------------------------------


@pytest.fixture
def gpus_of_16_gib(monkeypatch):
    """The 2 logical GPUs of the test cluster, with 16 GiB each."""
    monkeypatch.setattr(backend, "_probe_cluster_gpu_memory", lambda: (16.0, 16.0))


def test_memory_claims_share_the_gpus(gpus_of_16_gib):
    # 8 of 16 GiB: half a GPU per call, so 2 calls share each of the 2 GPUs.
    calls = _overlapping_work(min_vram=8)

    assert 2 < _max_overlap(calls) <= 2 * CLUSTER_GPUS
    assert {len(call["gpus"]) for call in calls} == {1}


def test_memory_claims_of_long_running_executors(gpus_of_16_gib):
    collector = Collector.remote()
    protocol = _reports_environment(min_vram=8, task=B.TaskType.LONG_RUNNING)
    pipeline = _compile(protocol, collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        (result,) = ray.get(collector.get.remote())
        assert result["out"]["assigned"]["GPU"] == 0.5
    finally:
        pipeline.close()


@pytest.mark.parametrize(
    ("resources", "xla"),
    [
        ({"min_vram": 8}, "0.450"),
        ({"gpus": 0.25, "task": B.TaskType.LONG_RUNNING}, "0.225"),
        ({"gpus": 1}, None),
    ],
)
def test_shares_of_a_gpu_limit_xla_preallocation(gpus_of_16_gib, resources, xla):
    collector = Collector.remote()
    pipeline = _compile(_reports_environment(**resources), collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        (result,) = ray.get(collector.get.remote())
        assert result["out"]["xla"] == xla
        assert (result["out"]["hugepage"], result["out"]["hint"]) == ("0", False)
    finally:
        pipeline.close()


def test_memory_claim_is_resolved_for_group_by_children(gpus_of_16_gib):
    # The children compile in the router's process, where the patched probe
    # is not visible: the router passes on the memory probed by the parent.
    calls = _overlapping_work(min_vram=4)

    assert {len(call["gpus"]) for call in calls} == {1}
    assert _max_overlap(calls) > CLUSTER_GPUS


def _fake_nvidia_smi(output):
    def run(args, **kwargs):
        assert args[0] == "nvidia-smi"
        return subprocess.CompletedProcess(args, 0, stdout=output)

    return run


def test_gpu_memory_is_read_from_nvidia_smi(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi("15360\n24576\n"))

    assert backend._read_gpu_memory() == [15.0, 24.0]


def test_gpu_memory_is_empty_without_nvidia_smi(monkeypatch):
    def missing(args, **kwargs):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(subprocess, "run", missing)

    assert backend._read_gpu_memory() == []


def test_cluster_probe_reports_the_gpus_of_every_node():
    # The test cluster is a single node: the probe sees its physical GPUs.
    probed = backend._probe_cluster_gpu_memory.__wrapped__()

    assert probed == tuple(backend._read_gpu_memory())


def test_claims_are_sized_for_the_smallest_gpu(caplog):
    caplog.set_level(logging.INFO, logger=backend.__name__)

    num_gpus = backend._resolve_gpus(B.ComputeResources(min_vram=8), (24.0, 16.0))

    assert num_gpus == 0.5
    assert "sized for the smallest GPU (16 GiB)" in caplog.text
    assert "33% of the 24 GiB GPUs stays unused" in caplog.text


# -- NumPy hugepages ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, {"hugepage": "0", "hint": False}), ("1", {"hugepage": "1", "hint": True})],
)
def test_every_process_of_a_pipeline_gets_the_numpy_hugepage_setting(
    monkeypatch,
    numpy_hugepage_hint,
    value,
    expected,
):
    match value:
        case None:
            monkeypatch.delenv(NUMPY_HUGEPAGE_ENV_VAR, raising=False)
        case _:
            monkeypatch.setenv(NUMPY_HUGEPAGE_ENV_VAR, value)
    collector = Collector.remote()
    # The writer runs in the sink actor, whose options do not come from the
    # compute resources of a map.
    writer = CallbackSinkWriter(
        lambda item: ray.get(
            collector.append.remote({"map": item, "sink": _numpy_hugepages()}),
        ),
    )
    (sink,) = lower(
        [Source("items").map(lambda _: _numpy_hugepages()).write_to(writer)],
    )
    pipeline = RayBackend(init_ray=False).compile([sink])

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        (result,) = ray.get(collector.get.remote())
        assert result == {"map": expected, "sink": expected}
        # The driver has imported NumPy already: the backend switches it.
        assert _numpy_hugepages()["hint"] is expected["hint"]
    finally:
        pipeline.close()


# Runs a pipeline on a Ray instance started by the backend, so that the
# workers get the runtime_env of the job and of their task or actor.
_JOB_ENVIRONMENT = """
import json
import os

from numpy._core.multiarray import _set_madvise_hugepage
import ray

from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter


def environment():
    hint = _set_madvise_hugepage(False)
    _set_madvise_hugepage(hint)
    return {
        "hugepage": os.environ.get("NUMPY_MADVISE_HUGEPAGE"),
        "hint": hint,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "omp": os.environ.get("OMP_NUM_THREADS"),
    }


@ray.remote
class Collector:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def get(self):
        return self.items


backend = RayBackend()
driver_hint = _set_madvise_hugepage(False)
collector = Collector.remote()
writer = CallbackSinkWriter(
    lambda item: ray.get(collector.append.remote({"map": item, "sink": environment()})),
)
(sink,) = lower([Source("items").map(lambda _: environment()).write_to(writer)])
pipeline = backend.compile([sink])
pipeline.send("items", 0)
pipeline.flush()
(result,) = ray.get(collector.get.remote())
pipeline.close()
print(json.dumps({**result, "driver_hint": driver_hint}))
"""


def test_job_and_stage_environments_are_merged_in_the_workers():
    environment = {
        **{
            name: value
            for name, value in os.environ.items()
            # PYTHONPATH must reach the workers through the job's runtime_env.
            if name not in {NUMPY_HUGEPAGE_ENV_VAR, "PYTHONPATH"}
        },
        # A Ray instance of its own with 2 CPUs, not the cluster of the session.
        "RAY_ADDRESS": "local",
        "RAY_OVERRIDE_RESOURCES": json.dumps({"CPU": CLUSTER_CPUS}),
    }
    output = subprocess.run(
        [sys.executable, "-c", _JOB_ENVIRONMENT],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    ).stdout
    result = json.loads(output.splitlines()[-1])

    assert result["driver_hint"] is False
    for process in ("map", "sink"):
        assert result[process]["hugepage"] == "0"
        assert result[process]["hint"] is False
        assert os.path.abspath("src") in result[process]["pythonpath"].split(":")
    # Set by the options of the map only, merged with the job's variables.
    assert result["map"]["omp"] == str(CLUSTER_CPUS)
