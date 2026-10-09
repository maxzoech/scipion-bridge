scipion_bridge.core.streaming.pipeline
======================================

.. py:module:: scipion_bridge.core.streaming.pipeline


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.pipeline.Pipeline


Module Contents
---------------

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



