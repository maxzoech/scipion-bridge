from typing import Optional, List, Dict, Callable, Any, Type, Union, Tuple, TypeVar
from functools import partial, reduce
from pyrsistent import pdeque, PDeque

from scipion_bridge.core.struct import Struct
from scipion_bridge.core import struct

from .node import Node, FlushSignal, FLUSH, Stream
from .sink import Sink

_NodeT = TypeVar("_NodeT", bound=Node)


class Op(Node):
    """
    Intermediate operation node that allows chaining downstream operations.
    """

    def op(self, node: _NodeT) -> _NodeT:
        """Connect a downstream node to this op."""
        node.upstream.append(self)
        return node

    def map(self, func: Callable[[Any], Any]) -> "MapOp":
        return self.op(MapOp(func))

    def chunk(self, size: int) -> "ChunkOp":
        return self.op(ChunkOp(size))

    def min_chunk(self, size: int) -> "MinChunkOp":
        return self.op(MinChunkOp(size))

    def collect(self, count: Optional[int] = None) -> "CollectOp":
        return self.op(CollectOp(count))

    def combine_latest(self, *others: "Node") -> "CombineLatestOp":
        return CombineLatestOp(self, *others)

    def sink(self, callback: Callable[[Any], Any]) -> Sink:
        """Attach a terminal Sink node and return it."""
        sink_node = Sink(callback)
        self.op(sink_node)
        return sink_node


class Source(Op):
    """
    Entry point input stream node.
    """

    def __init__(self, name: str):
        super().__init__(upstream=[])
        self.name = name

    def compile(
        self,
        sources_map: Dict[str, Stream],
        compile_cache: Optional[Dict[Node, Stream]] = None,
    ) -> Stream:
        del compile_cache
        assert (
            self.name is not None
        ), f"Source node {self} has no name assigned before building."
        return sources_map[self.name]

    def transform(self, *streams: Stream) -> Stream:
        if streams:
            return streams[0]
        raise NotImplementedError(
            "Source node must be compiled via sources_map lookup."
        )

class ReduceOutputOp(Op):
    """Reduce operation node that accumulates input values into a single output."""

    def __init__(self):
        super().__init__(upstream=None)

    def _reduce_func(self, acc: Any, x: Any) -> Any:
        if isinstance(x, FlushSignal):
            return x

        for k, value in x.items():
            if not isinstance(value, struct.Set):
                raise ValueError(
                    f"ReduceOutputOp expects input values to be of type struct.Set, got {type(value)} for key '{k}'."
                )

            if k in acc:
                acc[k] = struct.Set.concat(acc[k], value)
            else:
                acc[k] = value

        return acc

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].accumulate(
            self._reduce_func,
            start={},
            returns_state=False,
        )


class MapOp(Op):
    """Mapping operation node."""

    def __init__(self, func: Callable[[Any], Any]):
        super().__init__(upstream=None)
        self.func = func

    def _map_func(self, x: Any) -> Any:
        if isinstance(x, FlushSignal):
            return x
        return self.func(x)

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].map(self._map_func)


class ChunkOp(Op):
    """
    Merges or splits Set[...] containers to a specific chunk size.
    """

    def __init__(self, size: int, upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.chunk_size = size

    def _accumulate_chunks(
        self,
        state: Tuple[List[struct.Set], int],
        new_set: Union[struct.Set, FlushSignal],
    ) -> Tuple[Tuple[List[struct.Set], int], List[Union[struct.Set, FlushSignal]]]:
        queue, capacity = state
        emitted: List[Union[struct.Set, FlushSignal]]

        if isinstance(new_set, FlushSignal):
            emitted = []
            if queue:
                leftover = (
                    queue[0]
                    if len(queue) == 1
                    else struct.Set.concat(*queue)
                )
                emitted.append(leftover)
            emitted.append(FLUSH)
            
            return ([], 0), emitted

        if not isinstance(new_set, struct.Set):
            raise ValueError("Input for chunk needs to be a set")

        if len(new_set) > 0:
            queue.append(new_set)
            capacity += len(new_set)

        emitted = []

        while capacity >= self.chunk_size:
            accumulated: List[struct.Set] = []
            needed = self.chunk_size

            while queue and needed > 0:
                head = queue[0]
                head_len = len(head)

                if head_len <= needed:
                    accumulated.append(head)
                    needed -= head_len
                    queue.pop(0)
                else:
                    accumulated.append(head[:needed])
                    queue[0] = head[needed:]
                    needed = 0

            chunk = (
                accumulated[0]
                if len(accumulated) == 1
                else struct.Set.concat(*accumulated)
            )
            emitted.append(chunk)
            capacity -= self.chunk_size

        return (queue, capacity), emitted

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].accumulate(
            self._accumulate_chunks,
            start=([], 0),
            returns_state=True,
        ).flatten()


class MinChunkOp(Op):
    """
    Buffers struct.Set containers until the total element count is >= min_size,
    then emits the complete combined set without splitting.
    """

    def __init__(self, min_size: int, upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)
        self.min_size = min_size

    def _accumulate_min_chunks(
        self,
        state: Tuple[List[struct.Set], int],
        new_set: Union[struct.Set, FlushSignal],
    ) -> Tuple[Tuple[List[struct.Set], int], List[Union[struct.Set, FlushSignal]]]:
        queue, capacity = state
        emitted: List[Union[struct.Set, FlushSignal]]

        if isinstance(new_set, FlushSignal):
            emitted = []
            if queue:
                emitted.append(
                    queue[0] if len(queue) == 1 else struct.Set.concat(*queue)
                )
            emitted.append(FLUSH)
            return ([], 0), emitted

        if not isinstance(new_set, struct.Set):
            raise ValueError("Input for MinChunkOp must be a struct.Set")

        if len(new_set) > 0:
            queue.append(new_set)
            capacity += len(new_set)

        emitted = []

        if capacity >= self.min_size:
            emitted.append(
                queue[0] if len(queue) == 1 else struct.Set.concat(*queue)
            )
            queue, capacity = [], 0

        return (queue, capacity), emitted

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].accumulate(
            self._accumulate_min_chunks,
            start=([], 0),
            returns_state=True,
        ).flatten()


class CollectOp(Op):
    """
    Buffers struct.Set containers until the total element count (sum of set lengths)
    reaches or exceeds count (if provided), then emits all buffered sets concatenated
    together at once.
    If count is None, buffers all incoming sets until a FlushSignal is received, then emits
    all buffered elements in a single concatenated batch.
    Once threshold is reached (when count is set), no new elements are emitted.
    If a FlushSignal is received, any currently buffered elements are flushed.
    """

    def __init__(
        self, count: Optional[int] = None, upstream: Optional[List[Node]] = None
    ):
        super().__init__(upstream=upstream)
        if count is not None and count <= 0:
            raise ValueError(
                f"Collect count must be a positive integer or None, got {count}"
            )
        self.count = count

    def _accumulate_collect(
        self,
        state: Tuple[List[struct.Set], int, bool],
        new_set: Union[struct.Set, FlushSignal],
    ) -> Tuple[Tuple[List[struct.Set], int, bool], List[Union[struct.Set, FlushSignal]]]:
        queue, capacity, done = state
        emitted: List[Union[struct.Set, FlushSignal]]

        if isinstance(new_set, FlushSignal):
            emitted = []
            if queue:
                res = (
                    queue[0] if len(queue) == 1 else struct.Set.concat(*queue)
                )
                if self.count is not None and len(res) > self.count:
                    res = res[: self.count]
                emitted.append(res)
            emitted.append(FLUSH)
            return ([], 0, done), emitted

        if not isinstance(new_set, struct.Set):
            raise ValueError("Input for Collect must be a struct.Set")

        if done:
            return (queue, capacity, done), []

        if len(new_set) > 0:
            queue.append(new_set)
            capacity += len(new_set)

        emitted = []

        if self.count is not None and capacity >= self.count:
            res = (
                queue[0] if len(queue) == 1 else struct.Set.concat(*queue)
            )
            if len(res) > self.count:
                res = res[: self.count]
            emitted.append(res)
            return ([], 0, True), emitted

        return (queue, capacity, False), emitted

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].accumulate(
            self._accumulate_collect,
            start=([], 0, False),
            returns_state=True,
        ).flatten()


class CombineLatestOp(Op):
    """
    Combines multiple upstream nodes into a stream of tuples containing the latest
    element emitted by each upstream stream.
    Buffers elements emitted by faster streams before all upstream streams have emitted
    at least once so that no initial elements are dropped.
    Converts FlushSignal into FLUSH sentinels.
    """

    def __init__(self, *upstreams: Node):
        if len(upstreams) < 2:
            raise ValueError("CombineOp requires at least 2 upstream nodes.")
        super().__init__(upstream=list(upstreams))

    def transform(self, *streams: Stream) -> Stream:
        if len(streams) < 2:
            raise ValueError("CombineOp requires at least 2 streams.")

        out_stream = Stream()

        num_streams = len(streams)
        buffers: List[List[Any]] = [[] for _ in range(num_streams)]
        latest: List[Any] = [None] * num_streams
        has_emitted: List[bool] = [False] * num_streams
        all_ready = [False]

        def _drain_initial_buffers():
            max_len = max(len(b) for b in buffers)
            for idx in range(max_len):
                tup = tuple(
                    buffers[i][idx] if idx < len(buffers[i]) else latest[i]
                    for i in range(num_streams)
                )
                out_stream.emit(tup)
            for b in buffers:
                b.clear()

        def _on_emit(stream_idx: int, val: Any):
            if isinstance(val, FlushSignal):
                out_stream.emit(FLUSH)
                return

            if not all_ready[0]:
                buffers[stream_idx].append(val)
                latest[stream_idx] = val
                has_emitted[stream_idx] = True

                if all(has_emitted):
                    all_ready[0] = True
                    _drain_initial_buffers()
            else:
                latest[stream_idx] = val
                tup = tuple(latest)
                out_stream.emit(tup)

        for idx, s in enumerate(streams):
            s.sink(partial(_on_emit, idx))

        return out_stream


CombineOp = CombineLatestOp


