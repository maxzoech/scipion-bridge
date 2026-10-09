scipion_bridge.backend.ray
==========================

.. py:module:: scipion_bridge.backend.ray


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/backend/ray/backend/index
   /autoapi/scipion_bridge/backend/ray/container/index
   /autoapi/scipion_bridge/backend/ray/mailbox/index
   /autoapi/scipion_bridge/backend/ray/profiling/index
   /autoapi/scipion_bridge/backend/ray/ray_protocol_runner/index
   /autoapi/scipion_bridge/backend/ray/resource_provider/index
   /autoapi/scipion_bridge/backend/ray/sinks/index


Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.ray.configure_ray_container


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.RayBackend
   scipion_bridge.backend.ray.RayCompiledPipeline
   scipion_bridge.backend.ray.RaySourceActor
   scipion_bridge.backend.ray.RayAccumulatorActor
   scipion_bridge.backend.ray.RaySinkActor
   scipion_bridge.backend.ray.RayDemuxActor
   scipion_bridge.backend.ray.RayContainer
   scipion_bridge.backend.ray.RayPipelineRunner
   scipion_bridge.backend.ray.RayResourceProvider
   scipion_bridge.backend.ray.RayResourceCoordinator


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.ray.configure_ray_env


Package Contents
----------------

.. py:class:: RayBackend(init_ray: bool = True, queue_size: int = 2, parameters: Optional[Mapping[str, Any]] = None, gpu_memory: Optional[Sequence[float]] = None, max_in_flight: Optional[int] = None, buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE, spill_threshold: Optional[int] = DEFAULT_SPILL_THRESHOLD, spill_store: Optional[scipion_bridge.core.streaming.spill.SpillStoreFactory] = None, profile: Union[None, bool, str, pathlib.Path, scipion_bridge.backend.ray.profiling.ProfileConfig] = None, profile_log_level: int = logging.INFO)

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



.. py:class:: RaySourceActor(name: str, config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str)

   Bases: :py:obj:`_PipelinedStage`


   Input source of the streaming pipeline, forwarding items unchanged.


   .. py:attribute:: name


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



.. py:class:: RayDemuxActor(key_fn: Callable[[Any], Any], template: scipion_bridge.core.streaming.ir.IROp, source_name: str, max_keys: Optional[int], prefix: str, config: _StageConfig, spill: Optional[scipion_bridge.backend.ray.mailbox.SpillPolicy], label: str, parameters: Mapping[str, Any], gpu_memory: Optional[Tuple[float, Ellipsis]] = None)

   Bases: :py:obj:`_PipelinedStage`


   Router of a ``group_by`` (IRDemux) into a child pipeline per key.

   The child pipeline of a key is compiled from a clone of the template when
   the first item of the key arrives. Its sink hands every result back to
   ``emit``, which forwards it as ``Keyed(key, result)``:

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



.. py:class:: RayContainer

   Bases: :py:obj:`dependency_injector.containers.DeclarativeContainer`


   .. py:attribute:: config


   .. py:attribute:: shell_exec


   .. py:attribute:: temp_file_provider


   .. py:attribute:: storage_provider


   .. py:attribute:: parameters


   .. py:attribute:: protocol_config_provider


   .. py:attribute:: resource_provider


   .. py:attribute:: streaming_backend


.. py:data:: configure_ray_container

.. py:function:: configure_ray_env(modules=None, packages=None, parameters: Optional[Mapping[str, Any]] = None)

   Configure and wire RayContainer.

   :param parameters: Values of the protocol parameters, served to ``Field.value``.


.. py:class:: RayPipelineRunner(protocol: scipion_bridge.core.protocol.protocol_base.Protocol, *, parameters: Optional[Mapping[str, Any]] = None, origin_types: Optional[Dict[str, Type]] = None, sink: Optional[Union[scipion_bridge.core.streaming.sink.Sink, scipion_bridge.core.streaming.sink_writer.SinkWriter, Callable[[Any], Any]]] = None, queue_size: int = 2, max_in_flight: Optional[int] = None, buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE, spill_threshold: Optional[int] = DEFAULT_SPILL_THRESHOLD, spill_store: Optional[scipion_bridge.core.streaming.spill.SpillStoreFactory] = None, profile: Union[None, bool, str, pathlib.Path] = None, profile_log_level: int = logging.INFO)

   Ray pipeline runner for Scipion Bridge protocols.

   Precomputes resolvers for protocol inputs upon initialization,
   compiles the protocol streaming DAG into a Ray pipeline,
   and executes it on input chunks.


   .. py:attribute:: protocol


   .. py:attribute:: parameters


   .. py:attribute:: origin_types


   .. py:attribute:: queue_size


   .. py:attribute:: max_in_flight
      :value: None



   .. py:attribute:: buffer_size
      :value: 4



   .. py:attribute:: spill_threshold
      :value: 4



   .. py:attribute:: spill_store
      :value: None



   .. py:attribute:: profile
      :value: None



   .. py:attribute:: profile_log_level
      :value: 20



   .. py:attribute:: sink
      :value: None



   .. py:property:: pipeline
      :type: scipion_bridge.core.streaming.pipeline.Pipeline


      Return the streaming Pipeline, compiling it on first access.


   .. py:method:: set_parameters(parameters: Mapping[str, Any]) -> None

      Replace the parameter values. Only possible before the pipeline is compiled.



   .. py:method:: set_profile(profile: Union[bool, str, pathlib.Path]) -> None

      Profile the pipeline. Only possible before the pipeline is compiled.



   .. py:property:: input_resolvers
      :type: Dict[str, scipion_bridge.core.typed.resolve.ComposedResolver]


      Return the precomputed input resolvers.


   .. py:method:: run(inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None

      Iterate over resolved inputs in lockstep interleaving and execute the compiled streaming pipeline.



   .. py:method:: close() -> None

      Terminate all actors allocated for the pipeline.



   .. py:method:: launch_as_terminal_application()

      CLI entry point for running the protocol from terminal arguments.



.. py:class:: RayResourceProvider

   Bases: :py:obj:`scipion_bridge.core.environment.resource_provider.ResourceProvider`


   Ray-aware resource provider supporting worker-local and cluster-shared resources.


   .. py:method:: clear() -> None

      Clear the worker-local resource cache.



   .. py:method:: reset_coordinator() -> None
      :classmethod:


      Reset the cluster coordinator's cached object references if active.



   .. py:method:: get_resource(name: str, builder: Callable[[Any], Any], instance: Any, scope: scipion_bridge.core.environment.resource_provider.ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None) -> Any

      Retrieve or build a resource for a given protocol instance.



.. py:class:: RayResourceCoordinator

   Cluster-wide coordinator managing shared ObjectRefs in Ray Plasma store.

   Every resource is built once, by a task holding a share of the cluster's
   CPU cores only while it builds; the coordinator itself holds none.


   .. py:method:: get_or_build(key: str, builder: Callable[[Any], Any], instance: Any) -> ray.ObjectRef


   .. py:method:: clear() -> None


