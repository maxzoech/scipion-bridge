"""Scheduling protocol stages by their compute resources on Ray.

The test cluster has 2 CPUs and 2 logical GPUs (see conftest).
"""

import os
import random
import time
from typing import Any

import pytest
import ray

import scipion_bridge as B
from scipion_bridge.backend.ray.backend import RayBackend
from scipion_bridge.core.environment.compute import CPUS_ENV_VAR
from scipion_bridge.core.streaming.node import lower
from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter

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
                _record(self.recorder, "work", seconds=1.0)
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


def _per_key_sleep(cpus):
    @B.resources(cpus=cpus)
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
                    _record(self.recorder, "work", seconds=3.0)
            return item

        def _output(self, keyed):
            return {"out": keyed.value}

    return PerKeySleep()


def _overlapping_work(cpus):
    protocol = _per_key_sleep(cpus)
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
        return _max_overlap(calls)
    finally:
        pipeline.close()


def test_calls_without_reserved_cpus_are_not_limited_by_the_cpus():
    # Ray starts worker processes a few at a time, so not all 6 calls need
    # to overlap; more than the 2 CPUs do.
    assert _overlapping_work(None) > CLUSTER_CPUS


def test_reserved_cpus_limit_the_calls_running_at_once():
    assert _overlapping_work(1) == CLUSTER_CPUS


def _environment(item):
    return {
        "out": {
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "cpus": os.environ.get(CPUS_ENV_VAR),
            "assigned": ray.get_runtime_context().get_assigned_resources(),
        },
    }


def _reports_environment(cpus):
    @B.resources(cpus=cpus)
    class ReportsEnvironment(B.Protocol):
        items: B.Input[int] = B.Input()

        def outputs(self):
            return {"out": dict}

        def steps(self):
            return self.items.map(_environment)

    return ReportsEnvironment()


def test_calls_without_reserved_cpus_use_all_cores():
    collector = Collector.remote()
    pipeline = _compile(_reports_environment(None), collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        ((_, result),) = [item.popitem() for item in ray.get(collector.get.remote())]
        assert result["omp"] == str(CLUSTER_CPUS)
        assert result["cpus"] is None
        assert "CPU" not in result["assigned"]
    finally:
        pipeline.close()


def test_calls_with_reserved_cpus_learn_their_number():
    collector = Collector.remote()
    pipeline = _compile(_reports_environment(1), collector)

    try:
        pipeline.send("items", 0)
        pipeline.flush()

        ((_, result),) = [item.popitem() for item in ray.get(collector.get.remote())]
        assert result["omp"] == "1"
        assert result["cpus"] == "1"
        assert result["assigned"]["CPU"] == 1
    finally:
        pipeline.close()


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
    time.sleep(random.uniform(0, 0.2))
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
