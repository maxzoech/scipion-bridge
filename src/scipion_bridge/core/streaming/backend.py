"""Streaming backend provider ABCs."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ir import IROp
from .report.model import Annotations, Layer, PipelineReport, TablePanel


@dataclass
class StageStats:
    """Execution metrics of a single pipeline stage.

    Attributes:
        items_in: Data items processed by the stage (excluding FLUSH).
        items_out: Data items forwarded downstream.
        idle_s: Time spent waiting for input. High values mean the stage is
            starved by its upstream.
        process_s: Time spent in the stage logic.
        blocked_s: Time spent waiting for room in the outbox. High values mean
            that the maps on the routes of the stage, which run at most
            ``max_in_flight`` items at a time, are throttling it.
        emit_s: Time spent forwarding items downstream, including
            serialization.
        buffered_peak: Most items waiting at once for the stage. High values
            mean the stage is slower than its upstream.
        spilled: Items that overflowed the buffer into the spill store.
        fetch_s: Time spent reading waiting items, from the object store or
            the spill store.
        spill_s: Time spent writing items to the spill store.
        wait_s: Time items waited before the stage took them: in its
            mailbox, or for an executor of its pool (maps). High values mean
            the stage is slower than its upstream.
        wait_max_s: The longest wait of an item.
        first_in_t: When the stage received its first item, in seconds since
            the epoch; ``None`` before.
        last_out_t: When the stage forwarded its last item so far, in
            seconds since the epoch; ``None`` before.
    """

    items_in: int = 0
    items_out: int = 0
    idle_s: float = 0.0
    process_s: float = 0.0
    blocked_s: float = 0.0
    emit_s: float = 0.0
    buffered_peak: int = 0
    spilled: int = 0
    fetch_s: float = 0.0
    spill_s: float = 0.0
    wait_s: float = 0.0
    wait_max_s: float = 0.0
    first_in_t: Optional[float] = None
    last_out_t: Optional[float] = None

    @property
    def active_s(self) -> Optional[float]:
        """The time from the first item in to the last item out, if known."""
        if self.first_in_t is None or self.last_out_t is None:
            return None
        return max(self.last_out_t - self.first_in_t, 0.0)


def merge_stats(stages: Sequence[StageStats]) -> StageStats:
    """The stats of several stages as one, e.g. the children of a group_by.

    Counts and times add up; peaks and the longest wait are the highest;
    the activity spans from the earliest first item to the latest last one.
    """
    firsts = [s.first_in_t for s in stages if s.first_in_t is not None]
    lasts = [s.last_out_t for s in stages if s.last_out_t is not None]
    return StageStats(
        **{
            field.name: sum(getattr(stage, field.name) for stage in stages)
            for field in fields(StageStats)
            if field.name not in _NOT_SUMMED
        },
        buffered_peak=max((s.buffered_peak for s in stages), default=0),
        wait_max_s=max((s.wait_max_s for s in stages), default=0.0),
        first_in_t=min(firsts, default=None),
        last_out_t=max(lasts, default=None),
    )


_NOT_SUMMED = frozenset({"buffered_peak", "wait_max_s", "first_in_t", "last_out_t"})


def throughput_of(stats: StageStats) -> Dict[str, float]:
    """Metrics derived from the stats of a stage, if it was active.

    ``throughput`` is the items it forwarded per second while it was
    active, ``s_per_item`` the mean time to process an item, ``busy`` the
    items in processing at once on average (at most 1 for a stage on an
    actor of its own; a map on a pool can reach its executors), and
    ``wait_mean_s`` the mean wait of an item before the stage took it.
    """
    active = stats.active_s
    if active is None or stats.items_in == 0:
        return {}
    return {
        "throughput": stats.items_out / active if active > 0 else 0.0,
        "s_per_item": stats.process_s / stats.items_in,
        "busy": stats.process_s / active if active > 0 else 0.0,
        "wait_mean_s": stats.wait_s / stats.items_in,
    }


class CompiledPipeline(abc.ABC):
    """Handle to a compiled, runnable streaming pipeline."""

    def start(self) -> None:
        """Allocate the resources of the pipeline (e.g. start its actors).

        Backends that start them lazily do so on the first ``send`` or
        ``flush`` otherwise.
        """
        pass

    @abc.abstractmethod
    def send(self, source_name: str, value: Any) -> None:
        """Push an item into the named source."""
        ...

    @abc.abstractmethod
    def flush(self) -> None:
        """Drain in-flight tasks and finalize sinks."""
        ...

    @abc.abstractmethod
    def stats(self) -> Dict[str, StageStats]:
        """Return execution metrics per stage, keyed by a readable stage label."""
        ...

    def layers(self, report: PipelineReport) -> Tuple[Layer, ...]:
        """What the backend adds to the report of the pipeline, e.g. the
        placement of its stages; ``report`` is its graph."""
        return ()

    def close(self) -> None:
        """Terminate any resources or actors allocated for this pipeline."""
        pass


class StreamingBackendProvider(abc.ABC):
    """Abstract base for streaming execution backends."""

    @abc.abstractmethod
    def compile(
        self,
        ir_sinks: List[IROp],
    ) -> CompiledPipeline:
        """Compile an IR DAG into a runnable CompiledPipeline."""
        ...


# The child of a group_by in the label of one of its stages: the stages of
# child ``a`` of ``2:group_by`` are labelled ``2:group_by[a]/<stage>``.
_CHILD = re.compile(r"\[[^\]]*\](?=/)")


def stats_layer(report: PipelineReport, stats: Mapping[str, StageStats]) -> Layer:
    """The stats of a pipeline as a layer of its report.

    The stats of the children of a ``group_by`` are merged into the nodes of
    its template (``merge_stats``), with ``children`` their number. Stages of
    the backend without a node in the report, such as the sinks of the
    children, are left out. Stages that were active also get the metrics of
    ``throughput_of``; the panel *Throughput* lists them, busiest first.
    """
    by_node: Dict[str, List[StageStats]] = {}
    for label, stage in stats.items():
        by_node.setdefault(_CHILD.sub("", label), []).append(stage)
    merged = {
        node.id: merge_stats(by_node[node.id])
        for node in report.nodes
        if node.id in by_node
    }
    children = {
        node.id: len(by_node[node.id])
        for node in report.nodes
        if node.parent is not None and node.id in by_node
    }
    derived = {node_id: throughput_of(stage) for node_id, stage in merged.items()}
    return Layer(
        name="stats",
        title="Stats",
        nodes={
            node_id: _annotations(stage, derived[node_id], children.get(node_id))
            for node_id, stage in merged.items()
        },
        metric="busy",
        panels=_throughput_panels(merged, derived),
        warnings=_bottleneck(report, merged, derived),
    )


def _annotations(
    stage: StageStats,
    derived: Mapping[str, float],
    children: Optional[int],
) -> Annotations:
    """The stats of a stage and the metrics derived from them."""
    values: Dict[str, Any] = {
        field.name: getattr(stage, field.name)
        for field in fields(StageStats)
        if getattr(stage, field.name) is not None
    }
    extra = {} if children is None else {"children": children}
    return {**values, **derived, **extra}


def _throughput_panels(
    merged: Mapping[str, StageStats],
    derived: Mapping[str, Mapping[str, float]],
) -> Tuple[TablePanel, ...]:
    active = sorted(
        (node_id for node_id in merged if derived[node_id]),
        key=lambda node_id: -derived[node_id]["busy"],
    )
    if not active:
        return ()
    return (
        TablePanel(
            "Throughput",
            (
                "stage",
                "items",
                "items/s",
                "s/item",
                "busy",
                "wait mean s",
                "wait max s",
                "peak buffered",
                "spilled",
            ),
            tuple(
                (
                    node_id,
                    merged[node_id].items_in,
                    round(derived[node_id]["throughput"], 3),
                    round(derived[node_id]["s_per_item"], 4),
                    round(derived[node_id]["busy"], 2),
                    round(derived[node_id]["wait_mean_s"], 4),
                    round(merged[node_id].wait_max_s, 4),
                    merged[node_id].buffered_peak,
                    merged[node_id].spilled,
                )
                for node_id in active
            ),
        ),
    )


def _bottleneck(
    report: PipelineReport,
    merged: Mapping[str, StageStats],
    derived: Mapping[str, Mapping[str, float]],
) -> Tuple[str, ...]:
    """A warning naming the busiest top-level stage, if items waited for it."""
    top = {node.id for node in report.nodes if node.parent is None}
    candidates = [n for n in merged if n in top and derived[n]]
    if not candidates:
        return ()
    busiest = max(candidates, key=lambda n: derived[n]["busy"])
    waited = merged[busiest].wait_s > merged[busiest].process_s / 10
    if not waited:
        return ()
    return (
        f"{busiest} is the bottleneck: busy {derived[busiest]['busy']:.2f}, "
        f"its items waited {derived[busiest]['wait_mean_s']:.3g} s on average "
        f"({merged[busiest].wait_max_s:.3g} s at most).",
    )
