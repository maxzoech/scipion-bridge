"""Streaming backend provider ABCs."""

from __future__ import annotations

import abc
from typing import Any, List
from .ir import IROp


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


class StreamingBackendProvider(abc.ABC):
    """Abstract base for streaming execution backends."""

    @abc.abstractmethod
    def compile(
        self,
        ir_sinks: List[IROp],
    ) -> CompiledPipeline:
        """Compile an IR DAG into a runnable CompiledPipeline."""
        ...
