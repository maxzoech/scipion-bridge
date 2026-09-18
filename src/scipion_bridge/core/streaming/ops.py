from typing import Optional, List, Dict, Callable, Any, Type, TypeAlias, Union, Tuple, TypeVar, Mapping, Generic, cast, overload
from functools import partial, reduce
from pyrsistent import pdeque, PDeque

from scipion_bridge.core.struct import Struct
from scipion_bridge.core import struct

from .node import Node, FlushSignal, FLUSH, Stream
from .sink import Sink

_NodeT = TypeVar("_NodeT", bound=Node)

E = TypeVar("E")
S = TypeVar("S")

_NO_DEFAULT = object()


class Op(Node):
    """
    Intermediate operation node that allows chaining downstream operations.
    """

    def op(self, node: _NodeT) -> _NodeT:
        """Connect a downstream node to this op."""
        node.upstream.append(self)
        return node

    @overload
    def accumulate(
        self,
        func: Callable[[E, E], Tuple[E, Optional[E]]],
        start: object = _NO_DEFAULT,
    ) -> "AccumulateOp[E, E]": ...

    @overload
    def accumulate(
        self,
        func: Callable[[S, E], Tuple[S, Optional[E]]],
        start: S,
    ) -> "AccumulateOp[E, S]": ...

    def accumulate(
        self,
        func: Callable[[S, E], Tuple[S, Optional[E]]],
        start: Union[S, object] = _NO_DEFAULT,
    ) -> "AccumulateOp":
        """Fold stream items using func, emitting the running accumulated value on every item."""
        return self.op(AccumulateOp(func, start=start))

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

    def flatten(self) -> "FlattenOp":
        """Unroll lists, tuples, or struct.Set items into individual emissions."""
        return self.op(FlattenOp())

    def group_by(self, key: Union[Callable[[Any], Any], str, int]) -> "GroupByOp":
        """Group stream items by key, emitting (key, full_item) pairs."""
        return self.op(GroupByOp(key))

    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        start: Any = _NO_DEFAULT,
    ) -> "ReduceOp":
        """Reduce stream items using func into a single value, emitted upon FlushSignal."""
        return self.op(ReduceOp(func, start=start))

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


_AccumulatorState: TypeAlias = Union[S, object]

class AccumulateOp(Op, Generic[E, S]):
    """
    Folds stream elements using an accumulator function, emitting the running accumulator on every item.
    """

    def __init__(
        self,
        func: Callable[[S, E], Tuple[S, Optional[E]]],
        start: _AccumulatorState = _NO_DEFAULT,
        upstream: Optional[List[Node]] = None,
    ):
        super().__init__(upstream=upstream)

        self.func = func
        self.start = start

    def _accumulate_step(
        self,
        state: S,
        new_val: E,
    ) -> Tuple[_AccumulatorState, List[Union[Optional[E], FlushSignal]]]:
        
        match (state, new_val):

            case (state, new_val) if state == _NO_DEFAULT:
                return cast(S, new_val), [new_val]

            case (_, FlushSignal()):
                return self.start, [FLUSH]

            case (state, new_val):
                next_state, next_val = self.func(state, new_val)
                return next_state, [next_val]

    def transform(self, *streams: Stream) -> Stream:

        return (
                streams[0].accumulate(
                self._accumulate_step,
                start=self.start,
                returns_state=True,
            )
            .flatten()
            .filter(lambda x: x is not None)
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
                acc[k] = struct.concat([acc[k], value])
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
                    else struct.concat(queue)
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
                else struct.concat(accumulated)
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
                    queue[0] if len(queue) == 1 else struct.concat(queue)
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
                queue[0] if len(queue) == 1 else struct.concat(queue)
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
                    queue[0] if len(queue) == 1 else struct.concat(queue)
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
                queue[0] if len(queue) == 1 else struct.concat(queue)
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


class FlattenOp(Op):
    """
    Unrolls iterable containers (list, tuple, struct.Set, etc.) into individual emissions.
    Passes FlushSignal through as [FLUSH] to preserve pipeline lifecycle.
    """

    def __init__(self, upstream: Optional[List[Node]] = None):
        super().__init__(upstream=upstream)

    def _prepare_unroll(self, x: Any) -> Any:
        if isinstance(x, FlushSignal):
            return [FLUSH]

        if isinstance(x, (list, tuple, set, struct.Set)):
            return x

        # 4. If an unexpected non-iterable arrives, raise or wrap it
        raise TypeError(
            f"FlattenOp expected an iterable or struct.Set, got {type(x).__name__}"
        )

    def transform(self, *streams: Stream) -> Stream:
        return (
            streams[0]
            .map(self._prepare_unroll)
            .flatten()
        )


CombineOp = CombineLatestOp


class GroupedOp(Op):
    """
    Intermediate operation node for streams carrying (key, value) pairs.
    Provides key-aware operations (such as map, keyed chunk, and keyed reduce)
    that operate on values while preserving keys.
    """

    def map(self, func: Callable[[Any], Any]) -> "GroupedMapOp":
        """Apply func to each value in the (key, value) stream, emitting (key, func(value))."""
        return self.op(GroupedMapOp(func))

    def chunk(self, size: int) -> "KeyedChunkOp":
        """Buffer items per key until size is reached, emitting (key, chunk)."""
        return self.op(KeyedChunkOp(size))

    def reduce_by_key(
        self,
        func: Callable[[Any, Any], Any],
        start: Any = _NO_DEFAULT,
    ) -> "KeyedReduceOp":
        """Reduce elements independently for each key, emitting (key, reduced_value) upon FlushSignal."""
        return self.op(KeyedReduceOp(func, start=start))


class GroupByOp(GroupedOp):
    """
    Groups stream elements by a key, emitting (key, full_item) pairs.
    Preserves the full original element and propagates FlushSignal.
    """

    def __init__(
        self,
        key: Union[Callable[[Any], Any], str, int],
        upstream: Optional[List[Node]] = None,
    ):
        super().__init__(upstream=upstream)
        self.key = key

    def _extract_key(self, item: Any) -> Any:
        if callable(self.key):
            return self.key(item)
        if isinstance(self.key, int):
            return item[self.key]
        if isinstance(self.key, str):
            if isinstance(item, (dict, Mapping)) and self.key in item:
                return item[self.key]
            if hasattr(item, self.key):
                return getattr(item, self.key)
            if isinstance(item, (list, tuple)) and self.key.isdigit():
                return item[int(self.key)]
            if hasattr(item, "__getitem__"):
                try:
                    return item[self.key]
                except Exception:
                    pass
            raise KeyError(
                f"Field or attribute '{self.key}' not found on {type(item).__name__}."
            )
        raise TypeError(
            f"Unsupported key type '{type(self.key).__name__}'. Expected callable, str, or int."
        )

    def _group_func(self, x: Any) -> Any:
        if isinstance(x, FlushSignal):
            return x
        k = self._extract_key(x)
        return (k, x)

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].map(self._group_func)


class GroupedMapOp(MapOp, GroupedOp):
    """
    Applies a mapping function to the value in each (key, value) pair, emitting (key, func(value)).
    """

    def __init__(
        self,
        func: Callable[[Any], Any],
        upstream: Optional[List[Node]] = None,
    ):
        super().__init__(func=func)
        if upstream is not None:
            self.upstream = upstream

    def _map_func(self, x: Any) -> Any:
        if isinstance(x, FlushSignal):
            return x
        if not (isinstance(x, tuple) and len(x) == 2):
            raise TypeError(
                f"GroupedMapOp expects (key, value) pairs, got {type(x).__name__}"
            )
        k, v = x
        return (k, self.func(v))

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].map(self._map_func)


class ReduceOp(Op):
    """
    Folds stream elements using an accumulator function and emits the final reduced value upon FlushSignal.
    """

    def __init__(
        self,
        func: Callable[[Any, Any], Any],
        start: Any = _NO_DEFAULT,
        upstream: Optional[List[Node]] = None,
    ):
        super().__init__(upstream=upstream)
        self.func = func
        self.start = start

    def _reduce_step(
        self,
        state: Tuple[Any, bool, bool],
        new_val: Any,
    ) -> Tuple[Tuple[Any, bool, bool], List[Any]]:
        acc, has_val, has_start = state

        if isinstance(new_val, FlushSignal):
            emitted: List[Any] = []
            if has_val:
                emitted.append(acc)
            emitted.append(FLUSH)
            reset_acc = self.start if has_start else None
            return (reset_acc, has_start, has_start), emitted

        if not has_val:
            acc = new_val
            has_val = True
        else:
            acc = self.func(acc, new_val)

        return (acc, True, has_start), []

    def transform(self, *streams: Stream) -> Stream:
        has_start = self.start is not _NO_DEFAULT
        initial_acc = self.start if has_start else None
        return streams[0].accumulate(
            self._reduce_step,
            start=(initial_acc, has_start, has_start),
            returns_state=True,
        ).flatten()


class _KeyedChunkState:
    def __init__(self) -> None:
        self.set_queue: List[struct.Set] = []
        self.set_capacity: int = 0
        self.item_queue: List[Any] = []
        self.is_set: Optional[bool] = None


class KeyedChunkOp(ChunkOp, GroupedOp):
    """
    Buffers stream items per key until size is reached, emitting (key, chunk).
    If values are struct.Set or Struct, chunks are emitted as struct.Set[T].
    Otherwise, chunks are emitted as Python lists.
    """

    def __init__(self, size: int, upstream: Optional[List[Node]] = None):
        super().__init__(size=size, upstream=upstream)

    def _accumulate_keyed_chunks(
        self,
        states: Dict[Any, _KeyedChunkState],
        new_item: Any,
    ) -> Tuple[Dict[Any, _KeyedChunkState], List[Any]]:
        emitted: List[Any] = []

        if isinstance(new_item, FlushSignal):
            for k, state in list(states.items()):
                if state.is_set:
                    if state.set_queue:
                        leftover = (
                            state.set_queue[0]
                            if len(state.set_queue) == 1
                            else struct.concat(state.set_queue)
                        )
                        emitted.append((k, leftover))
                else:
                    if state.item_queue:
                        emitted.append((k, state.item_queue))
            states.clear()
            emitted.append(FLUSH)
            return states, emitted

        if not (isinstance(new_item, tuple) and len(new_item) == 2):
            raise TypeError(
                f"KeyedChunkOp expects (key, value) pairs, got {type(new_item).__name__}"
            )

        k, val = new_item
        if k not in states:
            states[k] = _KeyedChunkState()
        state = states[k]

        if isinstance(val, (struct.Set, Struct)):
            state.is_set = True
            if isinstance(val, Struct):
                set_cls: Any = struct.Set
                item_set: struct.Set = set_cls[type(val)]([val])
            else:
                item_set = val
            if len(item_set) > 0:
                state.set_queue.append(item_set)
                state.set_capacity += len(item_set)

            while state.set_capacity >= self.chunk_size:
                accumulated: List[struct.Set] = []
                needed = self.chunk_size

                while state.set_queue and needed > 0:
                    head = state.set_queue[0]
                    head_len = len(head)

                    if head_len <= needed:
                        accumulated.append(head)
                        needed -= head_len
                        state.set_queue.pop(0)
                    else:
                        accumulated.append(head[:needed])
                        state.set_queue[0] = head[needed:]
                        needed = 0

                chunk = (
                    accumulated[0]
                    if len(accumulated) == 1
                    else struct.concat(accumulated)
                )
                emitted.append((k, chunk))
                state.set_capacity -= self.chunk_size

        else:
            state.is_set = False
            state.item_queue.append(val)
            while len(state.item_queue) >= self.chunk_size:
                chunk_items = state.item_queue[: self.chunk_size]
                state.item_queue = state.item_queue[self.chunk_size :]
                emitted.append((k, chunk_items))

        return states, emitted

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].accumulate(
            self._accumulate_keyed_chunks,
            start={},
            returns_state=True,
        ).flatten()


class KeyedReduceOp(GroupedOp):
    """
    Reduces stream elements independently for each key, emitting (key, reduced_value) upon FlushSignal.
    """

    def __init__(
        self,
        func: Callable[[Any, Any], Any],
        start: Any = _NO_DEFAULT,
        upstream: Optional[List[Node]] = None,
    ):
        super().__init__(upstream=upstream)
        self.func = func
        self.start = start

    def _accumulate_keyed_reduce(
        self,
        states: Dict[Any, Tuple[Any, bool]],
        new_item: Any,
    ) -> Tuple[Dict[Any, Tuple[Any, bool]], List[Any]]:
        emitted: List[Any] = []
        has_start = self.start is not _NO_DEFAULT

        if isinstance(new_item, FlushSignal):
            for k, (acc, has_val) in list(states.items()):
                if has_val:
                    emitted.append((k, acc))
            states.clear()
            emitted.append(FLUSH)
            return states, emitted

        if not (isinstance(new_item, tuple) and len(new_item) == 2):
            raise TypeError(
                f"KeyedReduceOp expects (key, value) pairs, got {type(new_item).__name__}"
            )

        k, val = new_item
        if k not in states:
            if has_start:
                acc = self.func(self.start, val)
                states[k] = (acc, True)
            else:
                states[k] = (val, True)
        else:
            acc, _ = states[k]
            states[k] = (self.func(acc, val), True)

        return states, emitted

    def transform(self, *streams: Stream) -> Stream:
        return streams[0].accumulate(
            self._accumulate_keyed_reduce,
            start={},
            returns_state=True,
        ).flatten()



