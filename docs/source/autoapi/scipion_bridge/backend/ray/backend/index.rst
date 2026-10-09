scipion_bridge.backend.ray.backend
==================================

.. py:module:: scipion_bridge.backend.ray.backend


Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.ray.backend.logger
   scipion_bridge.backend.ray.backend.DEFAULT_BUFFER_SIZE
   scipion_bridge.backend.ray.backend.DEFAULT_SPILL_THRESHOLD


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.backend.RaySourceActor
   scipion_bridge.backend.ray.backend.RayMergeActor
   scipion_bridge.backend.ray.backend.RayAccumulatorActor
   scipion_bridge.backend.ray.backend.RayComputeExecutor
   scipion_bridge.backend.ray.backend.RayComputeGroup
   scipion_bridge.backend.ray.backend.RaySinkActor
   scipion_bridge.backend.ray.backend.RayDemuxActor
   scipion_bridge.backend.ray.backend.RayCompiledPipeline
   scipion_bridge.backend.ray.backend.RayBackend


Module Contents
---------------

.. py:data:: logger

.. py:data:: DEFAULT_BUFFER_SIZE
   :value: 4


.. py:data:: DEFAULT_SPILL_THRESHOLD
   :value: 4


.. py:class:: RaySourceActor(name: str, config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str)

   Bases: :py:obj:`_PipelinedStage`


   Input source of the streaming pipeline, forwarding items unchanged.


   .. py:attribute:: name


   .. py:method:: process(item: Any, port: int) -> List[Any]
      :async:


      Process an item arriving on ``port`` and return the items to emit.



.. py:class:: RayMergeActor(config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str)

   Bases: :py:obj:`_PipelinedStage`


   Joins the inputs of a map with several upstream stages.

   The items pass unchanged; the map is applied on the route of the merged
   stream, after the FLUSH of every input arrived.


   .. py:method:: process(item: Any, port: int) -> List[Any]
      :async:


      Process an item arriving on ``port`` and return the items to emit.



.. py:class:: RayAccumulatorActor(accumulate_fn: Callable[[Any, Any], tuple[Any, List[Any]]], initial_state_fn: Callable[[], Any], config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str, parameters: Mapping[str, Any], flush_fn: Optional[Callable[[Any], tuple[Any, List[Any]]]] = None, tag_inputs: bool = False)

   Bases: :py:obj:`_PipelinedStage`


   Stateful accumulation stage (IRAccumulate) of the streaming pipeline.

   With ``tag_inputs``, every item is passed to ``accumulate_fn`` as
   ``Tagged(port, item)`` so that it can tell its inputs apart.


   .. py:attribute:: accumulate_fn


   .. py:attribute:: flush_fn
      :value: None



   .. py:attribute:: tag_inputs
      :value: False



   .. py:attribute:: state


   .. py:method:: process(item: Any, port: int) -> List[Any]
      :async:


      Process an item arriving on ``port`` and return the items to emit.



   .. py:method:: on_flush() -> List[Any]
      :async:


      Finalize the stage on FLUSH and return the items to emit downstream.



.. py:class:: RayComputeExecutor(parameters: Mapping[str, Any], group: str, profile: Optional[scipion_bridge.backend.ray.profiling.ProfileConfig])

   Process holding the resources of a long-running group of maps.

   Every map of the group runs its calls here, so the maps share the
   resources and the process-scope resources of the protocol (e.g. a model
   on the GPU) are built once. A threaded actor: the maps call concurrently.

   Every map has ``_EXECUTOR_CALLS_IN_FLIGHT`` calls in flight, but its
   function runs one call at a time. Ray reads the argument of a call before
   it starts and writes its result after it returns, so these overlap with
   the computation of the other call (double buffering).


   .. py:method:: call(label: str, func: Callable[[Any], Any], item: Any) -> Any

      Run ``func`` of the map ``label`` on ``item``.



   .. py:method:: flush_profile() -> None

      Wait until the collector wrote the profiling events of the executor.



.. py:class:: RayComputeGroup(options: Dict[str, Any], members: int, parameters: Mapping[str, Any], group: str, profile: Optional[scipion_bridge.backend.ray.profiling.ProfileConfig])

   Coordinator of the executor of a long-running group of maps.

   The executor is created on the first ``acquire`` and killed, releasing
   its resources, once every map of the group has passed FLUSH. Items
   arriving after a flush create a new executor.


   .. py:method:: acquire() -> Any
      :async:


      Return the executor of the group, creating it if necessary.



   .. py:method:: flush_profile() -> None
      :async:


      Wait until the collector wrote the profiling events of the group.



   .. py:method:: release() -> None
      :async:


      Mark a map as flushed; kill the executor once all are flushed.



.. py:class:: RaySinkActor(writer: scipion_bridge.core.streaming.sink_writer.SinkWriter, config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str)

   Bases: :py:obj:`_PipelinedStage`


   Terminal stage wrapping a SinkWriter, finalized on FLUSH.


   .. py:attribute:: writer


   .. py:method:: process(item: Any, port: int) -> List[Any]
      :async:


      Process an item arriving on ``port`` and return the items to emit.



   .. py:method:: on_flush() -> List[Any]
      :async:


      Finalize the stage on FLUSH and return the items to emit downstream.



.. py:class:: RayDemuxActor(key_fn: Callable[[Any], Any], template: scipion_bridge.core.streaming.ir.IROp, source_name: str, max_keys: Optional[int], workers: Optional[int], outer_keyed: bool, group_by_workers: Optional[int], prefix: str, config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str, parameters: Mapping[str, Any], gpu_memory: Optional[Tuple[float, Ellipsis]] = None)

   Bases: :py:obj:`_PipelinedStage`


   Router of a ``group_by`` (IRDemux) into child pipelines.

   With ``workers=None``, every key has a child pipeline of its own, compiled
   from a clone of the template when the first item of the key arrives. With
   ``workers=n``, the keys share up to ``n`` children compiled from
   ``share_keys(template)``: every new key is assigned to the next child in
   turn, a child is started with its first key, and the items carry their
   key through it as ``Keyed(key, item)``.

   Nested in a shared child (``outer_keyed``), the items arrive as
   ``Keyed(outer, item)``; the router keys its children by ``(outer, key)``
   and emits ``Keyed(outer, Keyed(key, result))``.

   The sink of a child hands every result back to ``emit``, which forwards it
   as ``Keyed(key, result)``:

   - The router does not wait for a child to start: the items of the key
     wait in the queue of the child meanwhile, and the items of other keys
     are routed on. The children of the keys of a burst start in parallel.
     The queue holds up to ``buffer_size`` items; beyond it, the router
     waits, which propagates backpressure upstream.
   - The compute task queues items for a child, whose sink waits for room in
     the outbox of the router. The outbox is drained by the emitter task,
     independently of the compute task, so the two never wait on each other.
   - On FLUSH, the router flushes every child, after the items queued for
     it, before it forwards the FLUSH. A child returns from its flush only
     after its sink handed back all results, so they precede the FLUSH
     downstream.
   - The children abort the abort targets of the router, so that a failing
     child stops the enclosing pipeline.

   The children are owned by the router and terminated together with it.


   .. py:attribute:: gpu_memory
      :value: None



   .. py:attribute:: key_fn


   .. py:attribute:: template


   .. py:attribute:: source_name


   .. py:attribute:: max_keys


   .. py:attribute:: workers


   .. py:attribute:: outer_keyed


   .. py:attribute:: group_by_workers


   .. py:attribute:: prefix


   .. py:attribute:: config


   .. py:attribute:: parameters


   .. py:method:: process(item: Any, port: int) -> List[Any]
      :async:


      Process an item arriving on ``port`` and return the items to emit.



   .. py:method:: on_flush() -> List[Any]
      :async:


      Finalize the stage on FLUSH and return the items to emit downstream.



   .. py:method:: emit(key: Any, item: Any) -> None
      :async:


      Forward a result of the child pipeline of ``key``.



   .. py:method:: children_stats() -> Dict[str, scipion_bridge.core.streaming.backend.StageStats]
      :async:


      Return the execution metrics of every stage of the started children.



   .. py:method:: flush_profile() -> None
      :async:


      Wait until the collector wrote the events of the router and its children.



.. py:class:: RayCompiledPipeline(sources: Dict[str, Any], stages: Dict[str, Tuple[scipion_bridge.core.streaming.ir.IROp, Any]], demuxes: Sequence[Any] = (), groups: Sequence[Any] = (), spill_root: Optional[pathlib.Path] = None, profile: Optional[scipion_bridge.backend.ray.profiling.ProfileConfig] = None, owns_profile: bool = False)

   Bases: :py:obj:`scipion_bridge.core.streaming.backend.CompiledPipeline`


   Executable compiled streaming pipeline running on Ray.


   .. py:method:: source_handle(source_name: str) -> Any

      Return the actor handle of a named input source.

      Lets an async actor push into the pipeline without blocking its event
      loop: ``await pipeline.source_handle(name).push.remote(item)``.



   .. py:method:: send(source_name: str, value: Any) -> None

      Push an item into a named input source.

      Returns once the source has buffered the item. While the buffer of
      the source is full, because the stages downstream of it are, it waits
      for room.



   .. py:method:: flush() -> None

      Flush all sources in parallel and wait for pipeline completion.



   .. py:method:: flush_profile() -> None

      Wait until the collector wrote the profiling events of every process.

      A stage sends its events at every FLUSH; this sends the events since,
      e.g. of a failed pipeline.



   .. py:method:: stats() -> Dict[str, scipion_bridge.core.streaming.backend.StageStats]

      Return execution metrics per stage, in topological order.

      The stages of ``group_by`` children follow, labelled with their key.



   .. py:method:: close() -> None

      Terminate all actors allocated for this pipeline.

      The children of ``group_by`` routers terminate with their router, and
      the executors of compute groups with their coordinator. A profiled
      pipeline completes its trace first.



.. py:class:: RayBackend(init_ray: bool = True, queue_size: int = 2, parameters: Optional[Mapping[str, Any]] = None, gpu_memory: Optional[Sequence[float]] = None, max_in_flight: Optional[int] = None, buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE, spill_threshold: Optional[int] = DEFAULT_SPILL_THRESHOLD, spill_store: Optional[scipion_bridge.core.streaming.spill.SpillStoreFactory] = None, profile: Union[None, bool, str, pathlib.Path, scipion_bridge.backend.ray.profiling.ProfileConfig] = None, profile_log_level: int = logging.INFO, group_by_workers: Optional[int] = DEFAULT_GROUP_BY_WORKERS)

   Bases: :py:obj:`scipion_bridge.core.streaming.backend.StreamingBackendProvider`


   Streaming backend that compiles IR DAGs into persistent Ray actors with direct P2P messaging.


   .. py:attribute:: queue_size
      :value: 2



   .. py:attribute:: max_in_flight
      :value: 16



   .. py:attribute:: buffer_size
      :value: 4



   .. py:attribute:: spill_threshold
      :value: 4



   .. py:attribute:: spill_store
      :value: None



   .. py:attribute:: parameters


   .. py:attribute:: profile


   .. py:attribute:: profile_log_level
      :value: 20



   .. py:attribute:: group_by_workers
      :value: 4



   .. py:method:: compile(ir_sinks: List[scipion_bridge.core.streaming.ir.IROp], *, name: str = '', abort_targets: Sequence[Any] = ()) -> RayCompiledPipeline

      Compile a list of IR sink nodes into an executable RayCompiledPipeline.

      :param ir_sinks: Terminal IR nodes of the pipeline.
      :param name: Prefix of the stage labels and actor names, identifying a
                   nested pipeline (e.g. a ``group_by`` child) in stats and the
                   Ray dashboard.
      :param abort_targets: Further actor handles aborted when a stage of this
                            pipeline fails, in addition to its own sources. A nested
                            pipeline passes the sources of its parent, so that a failure
                            stops the parent even if it never sends to the failed stage
                            again.



   .. py:method:: compile_async(ir_sinks: List[scipion_bridge.core.streaming.ir.IROp], *, name: str = '', abort_targets: Sequence[Any] = ()) -> RayCompiledPipeline
      :async:


      Compile like ``compile``, without blocking the event loop.

      Starting the actors of a pipeline takes seconds; an async actor (the
      router of a ``group_by``) awaits it while it keeps working, and
      compiles several pipelines at once.



