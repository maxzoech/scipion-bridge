"""How stages and maps account their time, and the metrics derived from it
(in-process)."""

import pytest

import numpy as np

import scipion_bridge as B
from scipion_bridge.backend.ray.actors import CallTiming, _with_lineage, account_call
from scipion_bridge.core.struct import identity
from scipion_bridge.core.struct.id_column import IdColumn
from scipion_bridge.backend.ray.backend import visible_gpus
from scipion_bridge.core.streaming import (
    CallbackSinkWriter,
    IRMap,
    IRSink,
    IRSource,
    StageStats,
    merge_stats,
    report_of,
    throughput_of,
)
from scipion_bridge.core.streaming.backend import stats_layer


# -- Calls of maps -------------------------------------------------------------
def test_a_call_counts_its_execution_as_processing_and_the_rest_as_waiting():
    stats = StageStats()

    # Waited 0.2 s for an executor, then 0.5 s for the other call on it.
    account_call(stats, 0.2, CallTiming(waited_s=0.5, execute_s=1.0))
    account_call(stats, 0.0, CallTiming(waited_s=0.0, execute_s=1.0))

    assert (stats.items_in, stats.items_out) == (2, 2)
    assert stats.process_s == 2.0
    assert stats.wait_s == pytest.approx(0.7)
    assert stats.wait_max_s == pytest.approx(0.7)
    assert stats.last_done_t is not None


def test_a_failed_call_is_counted_without_times():
    stats = StageStats()

    account_call(stats, 0.1, None)

    assert (stats.items_in, stats.process_s, stats.wait_s) == (1, 0.0, 0.1)


def test_the_lineage_of_a_call_is_its_own_time():
    stats = StageStats()

    account_call(stats, 0.0, CallTiming(waited_s=0.0, execute_s=1.0, lineage_s=0.25))

    assert (stats.process_s, stats.lineage_s) == (1.0, 0.25)


class Value(B.Struct):
    value: float


def _values(n):
    s = B.Set[Value](capacity=n)
    s["value"] = np.zeros(n)
    identity.set_ids(s, IdColumn.of(b"F" * 16, range(n)))
    return s


def test_calls_find_element_ids_only_when_the_pipeline_records_lineage():
    item = _values(4)
    timing = CallTiming(waited_s=0.0, execute_s=1.0)

    assert _with_lineage(timing, False, item, item[1:3]) is timing
    recorded = _with_lineage(timing, True, item, item[1:3])
    assert recorded.elements and recorded.lineage_s > 0
    assert recorded.execute_s == 1.0
    # A map returning its item keeps its ids without comparing them.
    assert _with_lineage(timing, True, item, item).elements is None


# -- Derived metrics -----------------------------------------------------------
def test_throughput_counts_the_items_processed_also_of_sinks():
    sink = StageStats(
        items_in=10,
        process_s=1.0,
        first_in_t=0.0,
        last_done_t=5.0,
        capacity=1,
    )

    metrics = throughput_of(sink)

    assert metrics["throughput"] == 2.0
    assert metrics["utilization"] == 0.2


def test_utilization_is_busy_over_the_capacity():
    pool = StageStats(
        items_in=12,
        process_s=24.0,
        first_in_t=0.0,
        last_done_t=10.0,
        capacity=3,
    )
    tasks = StageStats(items_in=12, process_s=24.0, first_in_t=0.0, last_done_t=10.0)

    assert throughput_of(pool)["busy"] == 2.4
    assert throughput_of(pool)["utilization"] == pytest.approx(0.8)
    assert "utilization" not in throughput_of(tasks)


def test_merged_capacities_add_up_unless_one_is_unknown():
    one = StageStats(capacity=1)

    assert merge_stats([one, one]).capacity == 2
    assert merge_stats([one, StageStats()]).capacity is None


def _forward(item):
    return item


def _graph():
    source = IRSource(name="particles")
    pool = IRMap(func=_forward, name="pool")
    stage = IRMap(func=_forward, name="stage")
    tasks = IRMap(func=_forward, name="tasks")
    sink = IRSink(writer=CallbackSinkWriter(lambda item: None))
    source.add_downstream(pool)
    pool.add_downstream(stage)
    stage.add_downstream(tasks)
    tasks.add_downstream(sink)
    return report_of([sink])


def test_the_bottleneck_is_the_most_utilized_stage_not_the_busiest():
    report = _graph()
    ids = {node.title: node.id for node in report.nodes}

    def active(process_s, capacity):
        return StageStats(
            items_in=10,
            process_s=process_s,
            wait_s=5.0,
            first_in_t=0.0,
            last_done_t=10.0,
            capacity=capacity,
        )

    layer = stats_layer(
        report,
        {
            ids["pool"]: active(24.0, 3),
            ids["stage"]: active(9.7, 1),
            ids["tasks"]: active(150.0, None),
        },
    )

    (warning,) = layer.warnings
    assert warning.startswith(f"{ids['stage']} is the bottleneck")
    (panel,) = layer.panels
    assert [row[0] for row in panel.rows] == [ids["stage"], ids["pool"], ids["tasks"]]


# -- Visible GPUs --------------------------------------------------------------
_GPUS = [
    ("0", "GPU-aaaa", "24576"),
    ("1", "GPU-bbbb", "24576"),
    ("2", "GPU-cccc", "81920"),
]


@pytest.mark.parametrize(
    "visible, shown",
    [
        (None, ["0", "1", "2"]),
        ("2,1", ["2", "1"]),
        ("GPU-cccc", ["2"]),
        ("GPU-bb,0", ["1", "0"]),
        ("1,7,2", ["1"]),
        ("", []),
    ],
)
def test_visible_gpus_follow_cuda_visible_devices(visible, shown):
    assert [gpu[0] for gpu in visible_gpus(_GPUS, visible)] == shown
