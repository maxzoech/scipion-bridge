from .node import Node
from typing import Callable, Any, Optional, List

from streamz import Stream


class Sink(Node):
    """
    Terminal output node. Cannot chain further downstream operations.
    """

    def __init__(self, callback: Callable[[Any], Any], upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.callback = callback

    def transform(self, stream: Stream) -> Stream:
        return stream.sink(self.callback)