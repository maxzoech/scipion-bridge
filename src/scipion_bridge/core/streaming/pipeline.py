from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)
from dependency_injector.wiring import Provide, inject

from .backend import CompiledPipeline, StageStats, StreamingBackendProvider, stats_layer
from .ir import IROp
from .node import Node, lower
from .report.model import PipelineReport, report_of


class Pipeline:
    """
    Compiled streaming engine backed by a StreamingBackendProvider (e.g. Ray).
    """

    def __init__(
        self,
        compiled: CompiledPipeline,
        ir_sinks: Sequence[IROp] = (),
    ):
        """
        Args:
            compiled: The pipeline compiled by a backend.
            ir_sinks: The IR the pipeline was compiled from, for its report.
        """
        self._compiled = compiled
        self._ir_sinks: List[IROp] = list(ir_sinks)
        self._closed = False
        # The stats of the stages when the pipeline was closed; None if it was
        # not closed yet, or released after a failure.
        self._final_stats: Optional[Dict[str, StageStats]] = None

    @classmethod
    @inject
    def from_sink(
        cls,
        *nodes: Node,
        backend: Optional[StreamingBackendProvider] = Provide["streaming_backend"],
    ) -> Pipeline:
        """
        Factory method: Lowers the DAG starting from target nodes and compiles
        it into an executable pipeline using the configured streaming backend.
        """
        if not nodes:
            raise ValueError("Pipeline.from_sink() requires at least one target Node.")

        # Lower the declarative DAG into IR ops
        return cls.from_ir(lower(list(nodes)), backend=backend)

    @classmethod
    @inject
    def from_ir(
        cls,
        ir_sinks: Sequence[IROp],
        *,
        backend: Optional[StreamingBackendProvider] = Provide["streaming_backend"],
    ) -> Pipeline:
        """Compile the IR ending in ``ir_sinks`` with the configured backend."""
        assert backend is not None, "Backend must not be None"
        compiled = backend.compile(list(ir_sinks))
        return cls(compiled=compiled, ir_sinks=ir_sinks)

    @property
    def compiled(self) -> CompiledPipeline:
        """The backend's compiled pipeline, for backend-specific access."""
        return self._compiled

    def send(self, **kwargs: Any) -> None:
        """
        Submit data into the compiled stream using named keyword arguments.

        Usage:
            pipeline.send(particles=particle_set)
        """
        if not kwargs:
            raise ValueError(
                "send() requires named keyword arguments (e.g., pipeline.send(particles=...))"
            )

        self._check_open()
        for name, value in kwargs.items():
            if isinstance(value, list):
                raise TypeError(
                    f"Python lists are not supported for input '{name}'. Use core.struct.Set instead."
                )
            self._compiled.send(name, value)

    def flush(self) -> None:
        """
        Flush all stateful operations in the pipeline by sending a FLUSH sentinel to all input sources.
        """
        self._check_open()
        self._compiled.flush()

    def stats(self) -> Dict[str, StageStats]:
        """Return execution metrics per stage of the compiled pipeline.

        After ``close()``, the stats the stages had when it was closed.

        Raises:
            RuntimeError: If the pipeline was released after a failure,
                without its stats.
        """
        match (self._closed, self._final_stats):
            case (False, _):
                return self._compiled.stats()
            case (True, None):
                raise RuntimeError(
                    "The pipeline was released after a failure; its stats are "
                    "not available.",
                )
            case (True, stats):
                return stats

    def report(self) -> PipelineReport:
        """The graph of the pipeline, annotated with the stats of its stages
        and the layers of its backend (e.g. the placement of its pools).

        Render it with ``scipion_bridge.core.streaming.report``. Also available
        after ``close()``, with the final stats.
        """
        report = report_of(self._ir_sinks)
        layers = [stats_layer(report, self.stats()), *self._compiled.layers(report)]
        for layer in layers:
            report = report.with_layer(layer)
        return report

    def close(self) -> None:
        """Terminate the actors and other resources of the backend.

        Backends such as Ray keep them alive until they are closed: the stages
        hold references to each other, so dropping the pipeline does not free
        them. The stats of the stages remain available. Closing a closed
        pipeline does nothing.
        """
        match self._closed:
            case True:
                pass
            case False:
                self._closed = True
                try:
                    self._final_stats = self._compiled.stats()
                finally:
                    self._compiled.close()

    def _release(self) -> None:
        """Terminate the backend's resources of a failed pipeline, without stats."""
        self._closed = True
        self._compiled.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "The pipeline is closed. Compile a new one to process more items.",
            )

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Flush and close the pipeline, or only release it after an error."""
        match exc_type:
            case None:
                try:
                    self.flush()
                finally:
                    self.close()
            case _:
                # The pipeline failed: flushing it would only raise again and
                # hide the original error.
                self._release()
