scipion_bridge.backend.ray.ray_protocol_runner
==============================================

.. py:module:: scipion_bridge.backend.ray.ray_protocol_runner


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.ray_protocol_runner.RayPipelineRunner


Module Contents
---------------

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



