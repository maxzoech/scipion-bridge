"""Streaming backend provider ABCs."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List
from .ir import IROp


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
