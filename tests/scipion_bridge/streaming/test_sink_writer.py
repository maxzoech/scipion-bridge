import asyncio
from typing import Any
from scipion_bridge.core.streaming.sink_writer import SinkWriter, CallbackSinkWriter


def test_callback_sink_writer_async():
    async def _run():
        received = []
        writer = CallbackSinkWriter(lambda x: received.append(x))

        assert isinstance(writer, SinkWriter)

        await writer.write(10)
        await writer.write(20)
        await writer.finalize()

        assert received == [10, 20]

    asyncio.run(_run())


def test_custom_sink_writer():
    async def _run():
        class CustomWriter(SinkWriter):
            def __init__(self):
                self.items = []
                self.finalized = False

            async def write(self, item: Any) -> None:
                self.items.append(item)

            async def finalize(self) -> None:
                self.finalized = True

        writer = CustomWriter()
        assert isinstance(writer, SinkWriter)

        await writer.write("hello")
        await writer.write("world")
        assert not writer.finalized

        await writer.finalize()
        assert writer.finalized
        assert writer.items == ["hello", "world"]

    asyncio.run(_run())
