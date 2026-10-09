scipion_bridge.core.streaming.ops
=================================

.. py:module:: scipion_bridge.core.streaming.ops

.. autoapi-nested-parse::

   Declarative streaming operations and DAG nodes.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.streaming.ops.K


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.ops.Op
   scipion_bridge.core.streaming.ops.Source
   scipion_bridge.core.streaming.ops.MapOp
   scipion_bridge.core.streaming.ops.MapElementOp
   scipion_bridge.core.streaming.ops.ChunkOp
   scipion_bridge.core.streaming.ops.CollectOp
   scipion_bridge.core.streaming.ops.FlattenOp
   scipion_bridge.core.streaming.ops.CombineLatestOp
   scipion_bridge.core.streaming.ops.KeyedOp


Functions
---------

.. autoapisummary::

   scipion_bridge.core.streaming.ops.lineage


Module Contents
---------------

.. py:data:: K

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



   .. py:method:: group_by(key: int | str, pipeline: Callable[[Op], Op], *, max_keys: Optional[int] = None) -> KeyedOp[Any]
                  group_by(key: Callable[[Any], K], pipeline: Callable[[Op], Op], *, max_keys: Optional[int] = None) -> KeyedOp[K]

      Run ``pipeline`` separately on the items of every key (demux).

      Every item is routed by its key into a pipeline of its own, so that
      stateful operations (``chunk``, ``collect``, ``combine_latest``) only
      see the items of one key. The pipeline of a key is created when its
      first item arrives. Results are emitted as ``Keyed(key, result)``;
      call ``unkey()`` to continue with the merged stream::

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
                       pipeline. Every key allocates the stages of its own pipeline.



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



.. py:class:: KeyedOp(key_fn: Callable[[Any], K], entry: Source, exit_: Op, max_keys: Optional[int])

   Bases: :py:obj:`Op`, :py:obj:`Generic`\ [\ :py:obj:`K`\ ]


   Operation node routing items into a pipeline per key (``group_by``).

   The pipeline is kept as a template, outside of the enclosing graph, and
   lowered on its own.


   .. py:attribute:: key_fn


   .. py:attribute:: entry


   .. py:attribute:: exit_


   .. py:attribute:: max_keys


   .. py:method:: unkey() -> Op

      End the keyed region; the stream carries ``Keyed(key, result)`` items.



   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



.. py:function:: lineage(node: scipion_bridge.core.streaming.node.Node) -> Iterator[scipion_bridge.core.streaming.node.Node]

   Yield ``node`` and every node upstream of it, each once.

   The pipelines of ``group_by`` ops belong to the lineage of the op.


