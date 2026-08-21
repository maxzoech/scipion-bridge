from .node import Node, FlushSignal, Stream
from typing import Callable, Any, Optional, List


class Sink(Node):
    """
    Terminal output node. Cannot chain further downstream operations.
    """

    def __init__(self, callback: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.callback = callback

    def _sink_callback(self, item: Any) -> Any:
        if isinstance(item, FlushSignal):
            return None
        return self.callback(item)

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].sink(self._sink_callback)