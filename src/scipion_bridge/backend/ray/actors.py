from __future__ import annotations
from typing import Any, Callable, List
import asyncio
import ray

from scipion_bridge.core.streaming.node import FlushSignal, FLUSH
from scipion_bridge.core.streaming.sink_writer import SinkWriter


@ray.remote
class RaySourceActor:
    """
    Ray actor representing an input source in the streaming pipeline.
    Dispatches incoming items directly to connected downstream actors.
    """

    def __init__(self, name: str):
        self.name = name
        self._downstream: List[Any] = []

    def set_downstream(self, handles: List[Any]) -> None:
        """Configure downstream actor handles for direct P2P push."""
        self._downstream = handles

    async def push(self, item: Any) -> None:
        """Push an item to all downstream actors."""
        if not self._downstream:
            return
        futures = [handle.push.remote(item) for handle in self._downstream]
        await asyncio.gather(*futures)

    async def flush(self) -> None:
        """Send FLUSH signal to all downstream actors and await completion."""
        if not self._downstream:
            return
        futures = [handle.push.remote(FLUSH) for handle in self._downstream]
        await asyncio.gather(*futures)


@ray.remote
class RayWorkerActor:
    """
    Ray actor representing a transformation stage (IRMap) in the streaming pipeline.
    Applies func(item) and pushes the result directly to downstream actors.
    """

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func
        self._downstream: List[Any] = []

    def set_downstream(self, handles: List[Any]) -> None:
        """Configure downstream actor handles for direct P2P push."""
        self._downstream = handles

    async def push(self, item: Any) -> None:
        """Process an item and forward result or sentinel to downstream actors."""
        if isinstance(item, FlushSignal):
            out = item
        else:
            out = self.func(item)

        if self._downstream:
            futures = [handle.push.remote(out) for handle in self._downstream]
            await asyncio.gather(*futures)


@ray.remote
class RaySinkActor:
    """
    Ray actor wrapping a SinkWriter.
    Dispatches writes and finalizes upon receiving FLUSH signals.
    """

    def __init__(self, writer: SinkWriter):
        self.writer = writer

    async def push(self, item: Any) -> None:
        """Process incoming item or finalize on FlushSignal."""
        if isinstance(item, FlushSignal):
            await self.writer.finalize()
        else:
            await self.writer.write(item)

    async def flush(self) -> None:
        """Explicitly finalize the sink writer."""
        await self.writer.finalize()
