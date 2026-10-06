"""Generic async sink writer protocol for streaming pipeline output."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from scipion_bridge.core.struct.schema import SchemaConvertible


@runtime_checkable
class SinkWriter(Protocol):
    """Protocol for structured asynchronous streaming output.

    Implementations write SchemaConvertible containers (Struct, Set, Collection)
    atomically to a persistence target (Zarr, PostgreSQL, etc.).
    """

    async def write(self, item: SchemaConvertible) -> None:
        """Asynchronously write a SchemaConvertible item and ensure durable commit."""
        ...

    async def finalize(self) -> None:
        """Asynchronously flush pending writes and release resources."""
        ...


class CallbackSinkWriter:
    """Async adapter wrapping a callable as a SinkWriter for testing/debugging."""

    def __init__(self, callback: Callable[[Any], Any]):
        self.callback = callback

    async def write(self, item: Any) -> None:
        self.callback(item)

    async def finalize(self) -> None:
        pass
