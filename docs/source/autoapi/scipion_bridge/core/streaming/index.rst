scipion_bridge.core.streaming
=============================

.. py:module:: scipion_bridge.core.streaming


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/core/streaming/backend/index
   /autoapi/scipion_bridge/core/streaming/element_mapper/index
   /autoapi/scipion_bridge/core/streaming/ir/index
   /autoapi/scipion_bridge/core/streaming/keyed/index
   /autoapi/scipion_bridge/core/streaming/node/index
   /autoapi/scipion_bridge/core/streaming/ops/index
   /autoapi/scipion_bridge/core/streaming/pipeline/index
   /autoapi/scipion_bridge/core/streaming/sink/index
   /autoapi/scipion_bridge/core/streaming/sink_writer/index
   /autoapi/scipion_bridge/core/streaming/spill/index


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.streaming.FLUSH
   scipion_bridge.core.streaming.DEFAULT_GROUP_BY_WORKERS
   scipion_bridge.core.streaming.SpillStoreFactory


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.Node
   scipion_bridge.core.streaming.FlushSignal
   scipion_bridge.core.streaming.LoweringContext
   scipion_bridge.core.streaming.Op
   scipion_bridge.core.streaming.Source
   scipion_bridge.core.streaming.MapOp
   scipion_bridge.core.streaming.MapElementOp
   scipion_bridge.core.streaming.ChunkOp
   scipion_bridge.core.streaming.CollectOp
   scipion_bridge.core.streaming.FlattenOp
   scipion_bridge.core.streaming.CombineLatestOp
   scipion_bridge.core.streaming.KeyedOp
   scipion_bridge.core.streaming.ElementMapConfig
   scipion_bridge.core.streaming.Sink
   scipion_bridge.core.streaming.SinkWriter
   scipion_bridge.core.streaming.CallbackSinkWriter
   scipion_bridge.core.streaming.IROp
   scipion_bridge.core.streaming.IRSource
   scipion_bridge.core.streaming.IRMap
   scipion_bridge.core.streaming.IRAccumulate
   scipion_bridge.core.streaming.IRSink
   scipion_bridge.core.streaming.IRDemux
   scipion_bridge.core.streaming.Keyed
   scipion_bridge.core.streaming.Tagged
   scipion_bridge.core.streaming.WorkersFrom
   scipion_bridge.core.streaming.CompiledPipeline
   scipion_bridge.core.streaming.StageStats
   scipion_bridge.core.streaming.StreamingBackendProvider
   scipion_bridge.core.streaming.PickleSpillStore
   scipion_bridge.core.streaming.SpillStore
   scipion_bridge.core.streaming.Pipeline


Functions
---------

.. autoapisummary::

   scipion_bridge.core.streaming.lower
   scipion_bridge.core.streaming.replace_node
   scipion_bridge.core.streaming.clone_ir
   scipion_bridge.core.streaming.share_keys
   scipion_bridge.core.streaming.pickle_spill_store


Package Contents
----------------

.. py:class:: Node(upstream: Optional[List[Node]] = None)

   Base class for all nodes in the streaming computational graph.


   .. py:attribute:: upstream
      :type:  List[Node]


   .. py:attribute:: downstream
      :type:  List[Node]
      :value: []



   .. py:method:: lower(ctx: LoweringContext) -> scipion_bridge.core.streaming.ir.IROp
      :abstractmethod:


      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: FlushSignal

   Sentinel object emitted through the stream graph to trigger state flushing.


.. py:data:: FLUSH

.. py:function:: lower(nodes: List[Node]) -> List[scipion_bridge.core.streaming.ir.IROp]

   Lower one or more DAG root/sink nodes into lowered IR nodes.

   :returns: List of lowered IROp nodes corresponding to the input nodes.


.. py:function:: replace_node(old: Node, new: Node) -> None

   Rewire every downstream consumer of ``old`` to consume ``new`` instead.

   The position of ``old`` in each consumer's ``upstream`` list is preserved, so
   the argument order of multi-input nodes does not change. ``old`` is left
   without downstream nodes.


.. py:class:: LoweringContext

   Context coordinator that manages memoization and DAG wiring during lowering.


   .. py:attribute:: memo
      :type:  Dict[int, scipion_bridge.core.streaming.ir.IROp]


   .. py:attribute:: sources
      :type:  Dict[str, scipion_bridge.core.streaming.ir.IRSource]


   .. py:method:: lower_node(node: Node) -> scipion_bridge.core.streaming.ir.IROp

      Recursively lowers a Node using polymorphism and wires DAG dependencies.



.. py:class:: Op(upstream: Optional[List[Node]] = None)

   Bases: :py:obj:`scipion_bridge.core.streaming.node.Node`


   Intermediate operation node that allows chaining downstream operations.


   .. py:method:: op(node: _NodeT) -> _NodeT

      Connect a downstream node to this op.



   .. py:method:: map_batch(func: Callable[[Any], Any], *, cpu_only: bool = False) -> MapOp

      Transform entire incoming stream item / batch (1:1).

      Every map is executed as its own pipeline stage, and consecutive stages
      process different items concurrently. Splitting a step into separate
      maps therefore overlaps its parts, e.g. CPU post-processing of one
      batch with the GPU forward pass of the next::

          particles.chunk(256).map(forward).map(build_metadata)

      :param func: Function applied to each item.
      :param cpu_only: Run without the GPUs of the protocol's compute
                       resources (``@resources``), e.g. for cheap bookkeeping maps
                       that should not wait for a GPU.



   .. py:method:: map(func: Callable[[Any], Any], *, cpu_only: bool = False) -> MapOp

      Alias for map_batch.



   .. py:method:: map_element(func: Callable[[Any], Any], *, workers: scipion_bridge.core.streaming.element_mapper.Workers = 'auto', executor: scipion_bridge.core.streaming.element_mapper.Executor = 'thread', start_method: Optional[scipion_bridge.core.streaming.element_mapper.StartMethod] = None, chunksize: Optional[int] = None, cpu_only: bool = False) -> MapElementOp

      Apply ``func`` to every element of the incoming collections, in parallel.

      For each collection ``col`` (any collection with ``__len__``,
      ``__getitem__`` and ``__setitem__``) this runs
      ``col[i] = func(col[i])`` for all ``i`` on a thread pool and forwards
      the same, modified collection. ``func`` may modify the element in
      place (e.g. a Set row view) or return a new value. Every index is
      processed by exactly one worker.

      Use it after ``.chunk(n)`` to preprocess elements on the CPU in a
      stage of its own::

          particles.chunk(256).map_element(preprocess).map(forward)

      :param func: Function applied to each element.
      :param workers: Number of workers, or ``"auto"`` for one worker per
                      element capped at the available CPUs.
      :param executor: ``"thread"`` (default) or ``"process"``. Threads are safe
                       next to CUDA/JAX and scale when ``func`` releases the GIL
                       (NumPy, PyTorch, OpenCV); processes help for pure-Python work.
      :param start_method: Process start method (``"spawn"`` by default; only
                           with ``executor="process"``). Avoid ``"fork"`` in processes that
                           initialized CUDA or JAX.
      :param chunksize: Consecutive indices per pool task.
      :param cpu_only: Run without the GPUs of the protocol's compute
                       resources (``@resources``).



   .. py:method:: chunk(n: int, drop_last: bool = False) -> ChunkOp

      Accumulate Set[T] instances into batches of target size `n`.

      :param n: Target chunk size (number of elements in the output Set). Must be > 0.
      :param drop_last: If True, any partial remainder Set smaller than `n` upon
                        stream completion (FlushSignal) is dropped. This prevents downstream
                        JIT-compiled models from triggering recompilations for a non-standard
                        batch size.



   .. py:method:: collect(n: int) -> CollectOp

      Collect the first `n` elements of a stream of Set[T] into a single Set.

      The collected Set is emitted once, as soon as `n` elements have arrived;
      all later items are ignored. If the stream ends (FlushSignal) before `n`
      elements arrived, the elements collected so far are emitted instead.

      Use it to train a model on an initial sample of the stream::

          model = particles.collect(5_000).map(train)

      :param n: Number of elements to collect. Must be > 0.



   .. py:method:: flatten() -> FlattenOp

      Emit every element of incoming iterables as a separate item (1:N).

      Accepts any iterable, e.g. lists, tuples, generators or a Collection,
      which yields its initialized items in index order. Strings, bytes and
      mappings are rejected, as iterating them yields characters or keys.

      A Set is a batch of rows, not an iterable, and is rejected as well:
      sending its rows as separate items would be much more expensive than
      sending the batch. Use ``chunk`` to change batch sizes instead.



   .. py:method:: combine_latest(other: Op) -> CombineLatestOp

      Pair every item of this stream with the latest item of `other`.

      Emits `(item, latest)` for every item of this stream, where `latest` is
      the most recent item of `other`. Items arriving before `other` produced
      its first item are buffered and emitted once it has. Items of `other`
      only update `latest` and emit nothing themselves. Items still buffered
      at the end of the stream (FlushSignal) are dropped, as `other` never
      produced an item to pair them with.

      Use it to apply a model trained on a sample of the stream to the whole
      stream::

          model = particles.collect(5_000).map(train)
          particles.chunk(256).combine_latest(model).map(predict)

      :param other: Stream providing the latest value. Must be a different stream
                    than this one.



   .. py:method:: group_by(key: int | str, pipeline: Callable[[Op], Op], *, max_keys: Optional[int] = None, workers: scipion_bridge.core.streaming.ir.GroupByWorkers = WorkersFrom.BACKEND) -> KeyedOp[Any]
                  group_by(key: Callable[[Any], K], pipeline: Callable[[Op], Op], *, max_keys: Optional[int] = None, workers: scipion_bridge.core.streaming.ir.GroupByWorkers = WorkersFrom.BACKEND) -> KeyedOp[K]

      Run ``pipeline`` separately on the items of every key (demux).

      Every item is routed by its key into the pipeline, so that stateful
      operations (``chunk``, ``collect``, ``combine_latest``) only see the
      items of one key. The keys share a few copies of the pipeline
      (``workers``): every key is assigned to one when its first item
      arrives, in turn, and the stages of a copy keep a state per key. A
      slow key delays the other keys of its copy. Results are emitted as
      ``Keyed(key, result)``; call ``unkey()`` to continue with the merged
      stream::

          classes.flatten()
              .group_by(lambda cls: cls.class_id, pipeline=refine)
              .unkey()
              .map(write_class)

      :param key: Index or field name selecting the key of an item
                  (``item[key]``), or a function computing it. Keys must be
                  hashable.
      :param pipeline: Builds the pipeline of one key from its input stream.
                       It may only consume that input, and every branch must lead to
                       the stream it returns.
      :param max_keys: Maximum number of keys; a further key fails the
                       pipeline.
      :param workers: Number of copies of the pipeline the keys share, each
                      with stages (processes) of its own. ``None`` gives every key
                      a copy of its own. Defaults to the backend's setting.



   .. py:method:: write_to(writer: scipion_bridge.core.streaming.sink_writer.SinkWriter) -> scipion_bridge.core.streaming.sink.Sink

      Attach a terminal SinkWriter.



   .. py:method:: checkpoint(writer: scipion_bridge.core.streaming.sink_writer.SinkWriter) -> Op

      Attach an asynchronous persistence checkpoint without cutting off the stream.



   .. py:method:: sink(callback: Callable[[Any], Any]) -> scipion_bridge.core.streaming.sink.Sink

      Attach a callback-based sink (convenience for testing/debugging).



.. py:class:: Source(name: str)

   Bases: :py:obj:`Op`


   Entry point input stream node.


   .. py:attribute:: name


   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: MapOp(func: Callable[[Any], Any], *, cpu_only: bool = False)

   Bases: :py:obj:`Op`


   1:1 batch/element mapping operation node.


   .. py:attribute:: func


   .. py:attribute:: cpu_only
      :value: False



   .. py:attribute:: compute
      :type:  Optional[scipion_bridge.core.environment.compute.ComputeAssignment]
      :value: None



   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: MapElementOp(func: Callable[[Any], Any], config: scipion_bridge.core.streaming.element_mapper.ElementMapConfig, *, cpu_only: bool = False)

   Bases: :py:obj:`Op`


   Operation node applying a function to every element of a collection in parallel.


   .. py:attribute:: func


   .. py:attribute:: config


   .. py:attribute:: cpu_only
      :value: False



   .. py:attribute:: compute
      :type:  Optional[scipion_bridge.core.environment.compute.ComputeAssignment]
      :value: None



   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: ChunkOp(n: int, drop_last: bool = False)

   Bases: :py:obj:`Op`


   Operation node that accumulates Set[T] instances into fixed-size batches.


   .. py:attribute:: n


   .. py:attribute:: drop_last
      :value: False



   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: CollectOp(n: int)

   Bases: :py:obj:`Op`


   Operation node that collects the first `n` elements of a stream into one Set.


   .. py:attribute:: n


   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: FlattenOp

   Bases: :py:obj:`Op`


   Operation node emitting every element of incoming iterables (1:N).


   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: CombineLatestOp

   Bases: :py:obj:`Op`


   Operation node pairing every item of its first input with the latest of its second.


   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: KeyedOp(key_fn: Callable[[Any], K], entry: Source, exit_: Op, max_keys: Optional[int], workers: scipion_bridge.core.streaming.ir.GroupByWorkers)

   Bases: :py:obj:`Op`, :py:obj:`Generic`\ [\ :py:obj:`K`\ ]


   Operation node routing items into a pipeline per key (``group_by``).

   The pipeline is kept as a template, outside of the enclosing graph, and
   lowered on its own.


   .. py:attribute:: key_fn


   .. py:attribute:: entry


   .. py:attribute:: exit_


   .. py:attribute:: max_keys


   .. py:attribute:: workers


   .. py:method:: unkey() -> Op

      End the keyed region; the stream carries ``Keyed(key, result)`` items.



   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: ElementMapConfig

   Execution options of ``Op.map_element``.

   .. attribute:: workers

      Number of parallel workers. ``"auto"`` uses one worker per
      element of the collection, capped at the CPUs available to the
      process (the reserved cores if the backend reserved some); after
      ``.chunk(n)`` this is ``min(n, CPUs)``.

   .. attribute:: executor

      ``"thread"`` runs ``func`` in the thread pool;
      ``"process"`` runs it in a process pool (for pure-Python work that
      holds the GIL).

   .. attribute:: start_method

      Start method of the process pool; only valid with
      ``executor="process"``. Defaults to ``"spawn"``.

   .. attribute:: chunksize

      Number of consecutive indices per pool task. Defaults to 1
      for ``workers="auto"``, else ``ceil(len(col) / (4 * workers))``.


   .. py:attribute:: workers
      :type:  Workers
      :value: 'auto'



   .. py:attribute:: executor
      :type:  Executor
      :value: 'thread'



   .. py:attribute:: start_method
      :type:  Optional[StartMethod]
      :value: None



   .. py:attribute:: chunksize
      :type:  Optional[int]
      :value: None



.. py:class:: Sink(writer: Union[scipion_bridge.core.streaming.sink_writer.SinkWriter, Callable[[Any], Any]], upstream: Optional[List[scipion_bridge.core.streaming.node.Node]] = None)

   Bases: :py:obj:`scipion_bridge.core.streaming.node.Node`


   Terminal output or checkpoint node backed by an async SinkWriter.


   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:class:: SinkWriter

   Bases: :py:obj:`Protocol`


   Protocol for structured asynchronous streaming output.

   Implementations write SchemaConvertible containers (Struct, Set, Collection)
   atomically to a persistence target (Zarr, PostgreSQL, etc.).


   .. py:method:: write(item: scipion_bridge.core.struct.schema.SchemaConvertible) -> None
      :async:


      Asynchronously write a SchemaConvertible item and ensure durable commit.



   .. py:method:: finalize() -> None
      :async:


      Asynchronously flush pending writes and release resources.



.. py:class:: CallbackSinkWriter(callback: Callable[[Any], Any])

   Async adapter wrapping a callable as a SinkWriter for testing/debugging.

   The callback runs in a thread, so that a slow callback does not block the
   event loop of the stage writing, which keeps accepting items meanwhile.


   .. py:attribute:: callback


   .. py:method:: write(item: Any) -> None
      :async:



   .. py:method:: finalize() -> None
      :async:



.. py:class:: IROp

   Base class for all IR primitives with DAG edge management.


   .. py:attribute:: downstream
      :type:  List[IROp]
      :value: []



   .. py:attribute:: upstream
      :type:  List[IROp]
      :value: []



   .. py:method:: add_downstream(child: IROp) -> None

      Wire a downstream edge and reciprocal upstream edge.



.. py:class:: IRSource

   Bases: :py:obj:`IROp`


   Ingestion entry point.


   .. py:attribute:: name
      :type:  str
      :value: ''



.. py:class:: IRMap

   Bases: :py:obj:`IROp`


   Stateless 1:1 batch/element transformation.


   .. py:attribute:: func
      :type:  Callable[[Any], Any]


   .. py:attribute:: name
      :type:  Optional[str]
      :value: None



   .. py:attribute:: compute
      :type:  Optional[scipion_bridge.core.environment.compute.ComputeAssignment]
      :value: None



.. py:class:: IRAccumulate

   Bases: :py:obj:`IROp`


   Stateful stream accumulation primitive.

   Maintains internal state across incoming items and flushes,
   emitting zero or more output items downstream.

   A pipeline shared by the keys of a ``group_by`` creates the state of a key
   with its first item, and flushes only the keys it has seen. ``flush_fn``
   must therefore emit nothing on the initial state.


   .. py:attribute:: accumulate_fn
      :type:  Callable[[Any, Any], Tuple[Any, List[Any]]]


   .. py:attribute:: initial_state_fn
      :type:  Callable[[], Any]


   .. py:attribute:: flush_fn
      :type:  Optional[Callable[[Any], Tuple[Any, List[Any]]]]
      :value: None



   .. py:attribute:: name
      :type:  str
      :value: 'accumulate'



   .. py:attribute:: tag_inputs
      :type:  bool
      :value: False



.. py:class:: IRSink

   Bases: :py:obj:`IROp`


   Terminal/Checkpoint node delegating to an async SinkWriter.


   .. py:attribute:: writer
      :type:  Optional[scipion_bridge.core.streaming.sink_writer.SinkWriter]
      :value: None



.. py:class:: IRDemux

   Bases: :py:obj:`IROp`


   Routing of items into a child pipeline per key (``group_by``).

   ``template`` is the exit node of the child pipeline, lowered on its own; it
   is not wired into the enclosing DAG. Its only source is ``source_name``.
   The backend runs clones of it, one per key or shared by several keys
   (``workers``), and emits the results of every key as ``Keyed(key, result)``.

   In a pipeline shared by several keys (see ``share_keys``), the items arrive
   as ``Keyed(outer, item)`` (``outer_keyed``); the results are emitted as
   ``Keyed(outer, Keyed(key, result))``.


   .. py:attribute:: key_fn
      :type:  Callable[[Any], Any]


   .. py:attribute:: template
      :type:  Optional[IROp]
      :value: None



   .. py:attribute:: source_name
      :type:  str
      :value: ''



   .. py:attribute:: max_keys
      :type:  Optional[int]
      :value: None



   .. py:attribute:: workers
      :type:  GroupByWorkers


   .. py:attribute:: outer_keyed
      :type:  bool
      :value: False



   .. py:attribute:: name
      :type:  str
      :value: 'group_by'



.. py:class:: Keyed

   Bases: :py:obj:`NamedTuple`


   Result of a ``group_by`` pipeline, together with the key of its group.


   .. py:attribute:: key
      :type:  Any


   .. py:attribute:: value
      :type:  Any


.. py:class:: Tagged

   Item arriving on a stage with several inputs, tagged with its input port.

   The port is the position of the sending stage among the upstream stages
   of the receiving stage.


   .. py:attribute:: port
      :type:  int


   .. py:attribute:: item
      :type:  Any


.. py:class:: WorkersFrom(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Number of workers of a ``group_by`` that its backend decides.


   .. py:attribute:: BACKEND


.. py:data:: DEFAULT_GROUP_BY_WORKERS
   :value: 4


.. py:function:: clone_ir(sinks: Sequence[IROp], changes: Callable[[IROp], Mapping[str, Any]] = _no_changes) -> List[IROp]

   Copy the IR DAG reachable upstream from ``sinks``.

   Every node is copied with fresh edges, so the copy can be wired (e.g. to
   a new sink) and compiled without touching the original. Upstream edges
   keep their order, which preserves the input ports of multi-input stages.
   Functions, writers and other attributes are shared, not copied.

   :param sinks: Exit nodes of the DAG to copy.
   :param changes: Fields to replace in the copy of a node.

   :returns: The copies of ``sinks``, in the same order.


.. py:function:: share_keys(exit_: scipion_bridge.core.streaming.ir.IROp) -> scipion_bridge.core.streaming.ir.IROp

   Copy of a ``group_by`` pipeline carrying ``Keyed`` items of many keys.

   :param exit_: Exit node of the pipeline (the template of an ``IRDemux``).

   :returns: The exit node of the copy.


.. py:class:: CompiledPipeline

   Bases: :py:obj:`abc.ABC`


   Handle to a compiled, runnable streaming pipeline.


   .. py:method:: send(source_name: str, value: Any) -> None
      :abstractmethod:


      Push an item into the named source.



   .. py:method:: flush() -> None
      :abstractmethod:


      Drain in-flight tasks and finalize sinks.



   .. py:method:: stats() -> Dict[str, StageStats]
      :abstractmethod:


      Return execution metrics per stage, keyed by a readable stage label.



   .. py:method:: close() -> None

      Terminate any resources or actors allocated for this pipeline.



.. py:class:: StageStats

   Execution metrics of a single pipeline stage.

   .. attribute:: items_in

      Data items processed by the stage (excluding FLUSH).

   .. attribute:: items_out

      Data items forwarded downstream.

   .. attribute:: idle_s

      Time spent waiting for input. High values mean the stage is
      starved by its upstream.

   .. attribute:: process_s

      Time spent in the stage logic.

   .. attribute:: blocked_s

      Time spent waiting for room in the outbox. High values mean
      that the maps on the routes of the stage, which run at most
      ``max_in_flight`` items at a time, are throttling it.

   .. attribute:: emit_s

      Time spent forwarding items downstream, including
      serialization.

   .. attribute:: buffered_peak

      Most items waiting at once for the stage. High values
      mean the stage is slower than its upstream.

   .. attribute:: spilled

      Items that overflowed the buffer into the spill store.

   .. attribute:: fetch_s

      Time spent reading waiting items, from the object store or
      the spill store.

   .. attribute:: spill_s

      Time spent writing items to the spill store.


   .. py:attribute:: items_in
      :type:  int
      :value: 0



   .. py:attribute:: items_out
      :type:  int
      :value: 0



   .. py:attribute:: idle_s
      :type:  float
      :value: 0.0



   .. py:attribute:: process_s
      :type:  float
      :value: 0.0



   .. py:attribute:: blocked_s
      :type:  float
      :value: 0.0



   .. py:attribute:: emit_s
      :type:  float
      :value: 0.0



   .. py:attribute:: buffered_peak
      :type:  int
      :value: 0



   .. py:attribute:: spilled
      :type:  int
      :value: 0



   .. py:attribute:: fetch_s
      :type:  float
      :value: 0.0



   .. py:attribute:: spill_s
      :type:  float
      :value: 0.0



.. py:class:: StreamingBackendProvider

   Bases: :py:obj:`abc.ABC`


   Abstract base for streaming execution backends.


   .. py:method:: compile(ir_sinks: List[scipion_bridge.core.streaming.ir.IROp]) -> CompiledPipeline
      :abstractmethod:


      Compile an IR DAG into a runnable CompiledPipeline.



.. py:class:: PickleSpillStore(directory: pathlib.Path)

   Bases: :py:obj:`SpillStore`\ [\ :py:obj:`str`\ ]


   Pickles every item into a file of its own inside ``directory``.

   The directory is created on the first item, on the node of the stage.


   .. py:attribute:: directory


   .. py:method:: put(item: Any) -> str

      Store ``item`` and return the handle to read it back.



   .. py:method:: get(handle: str) -> Any

      Reconstruct the item stored under ``handle``.



   .. py:method:: discard(handle: str) -> None

      Free the storage of a consumed item.



.. py:class:: SpillStore

   Bases: :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`H`\ ]


   Stores the items that overflow the buffer of a pipeline stage.

   Every item is stored once, read back once and then discarded. Handles are
   small and picklable; the items themselves never pass through them.


   .. py:method:: put(item: Any) -> H
      :abstractmethod:


      Store ``item`` and return the handle to read it back.



   .. py:method:: get(handle: H) -> Any
      :abstractmethod:


      Reconstruct the item stored under ``handle``.



   .. py:method:: discard(handle: H) -> None
      :abstractmethod:


      Free the storage of a consumed item.



.. py:data:: SpillStoreFactory

.. py:function:: pickle_spill_store(root: pathlib.Path) -> SpillStoreFactory

   Factory of ``PickleSpillStore``s writing into a directory per stage under ``root``.


.. py:class:: Pipeline(compiled: scipion_bridge.core.streaming.backend.CompiledPipeline)

   Compiled streaming engine backed by a StreamingBackendProvider (e.g. Ray).


   .. py:method:: from_sink(*nodes: scipion_bridge.core.streaming.node.Node, backend: Optional[scipion_bridge.core.streaming.backend.StreamingBackendProvider] = Provide['streaming_backend']) -> Pipeline
      :classmethod:


      Factory method: Lowers the DAG starting from target nodes and compiles
      it into an executable pipeline using the configured streaming backend.



   .. py:method:: send(**kwargs: Any) -> None

      Submit data into the compiled stream using named keyword arguments.

      Usage:
          pipeline.send(particles=particle_set)



   .. py:method:: flush() -> None

      Flush all stateful operations in the pipeline by sending a FLUSH sentinel to all input sources.



   .. py:method:: stats() -> Dict[str, scipion_bridge.core.streaming.backend.StageStats]

      Return execution metrics per stage of the compiled pipeline.



   .. py:method:: close() -> None

      Terminate backend resources and actors allocated for this pipeline.



