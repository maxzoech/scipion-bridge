from __future__ import annotations
from typing import (
    Any,
    Dict,
    Optional,
)
from dependency_injector.wiring import Provide, inject

from .backend import CompiledPipeline, StageStats, StreamingBackendProvider
from .node import Node, lower


class Pipeline:
    """
    Compiled streaming engine backed by a StreamingBackendProvider (e.g. Ray).
    """

    def __init__(self, compiled: CompiledPipeline):
        self._compiled = compiled

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
        ir_sinks = lower(list(nodes))

        # Compile with the backend
        assert backend is not None, "Backend must not be None"
        compiled = backend.compile(ir_sinks)
        return cls(compiled=compiled)

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
        self._compiled.flush()

    def stats(self) -> Dict[str, StageStats]:
        """Return execution metrics per stage of the compiled pipeline."""
        return self._compiled.stats()

    def close(self) -> None:
        """Terminate backend resources and actors allocated for this pipeline."""
        self._compiled.close()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.flush()
