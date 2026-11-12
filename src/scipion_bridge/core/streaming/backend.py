"""Streaming backend provider ABCs."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Tuple

from .ir import IROp
from .report.model import Annotations, Layer, PipelineReport


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


class CompiledPipeline(abc.ABC):
    """Handle to a compiled, runnable streaming pipeline."""

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

    The stats of the children of a ``group_by`` are added up into the nodes
    of its template; ``buffered_peak`` is the highest of the children, and
    ``children`` their number. Stages of the backend without a node in the
    report, such as the sinks of the children, are left out.
    """
    by_node: Dict[str, List[StageStats]] = {}
    for label, stage in stats.items():
        by_node.setdefault(_CHILD.sub("", label), []).append(stage)
    return Layer(
        name="stats",
        title="Stats",
        nodes={
            node.id: _annotations(by_node[node.id], children=node.parent is not None)
            for node in report.nodes
            if node.id in by_node
        },
        metric="process_s",
    )


def _annotations(stages: List[StageStats], children: bool) -> Annotations:
    """The stats of a stage, or the sum over the children of a group_by."""
    values: Dict[str, Any] = {
        field.name: (
            max(getattr(stage, field.name) for stage in stages)
            if field.name == "buffered_peak"
            else sum(getattr(stage, field.name) for stage in stages)
        )
        for field in fields(StageStats)
    }
    match children:
        case True:
            return {**values, "children": len(stages)}
        case False:
            return values
