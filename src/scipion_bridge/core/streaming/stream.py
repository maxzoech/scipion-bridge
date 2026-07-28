from typing import TypeVar, Generic, Optional
from scipion_bridge.core.struct import Struct, Set

from streamz import Stream as BackendStream

T = TypeVar("T", Struct, Set)

class Stream(Generic[T]):

    _backend_stream: Optional[BackendStream] = None

    def __init__(self):

        self._backend_stream = None

    def _build(self):
        """
        Build the Scipion Bridge stream into a streaming object backed by
        streamz
        """

        def _map_fn(x):
            print(x)
            return x

        source = BackendStream()
        self._backend_stream = source.map(_map_fn).sink(print)
        

    def send(self, element: T):
        print(element)