scipion_bridge.core.streaming.backend
=====================================

.. py:module:: scipion_bridge.core.streaming.backend

.. autoapi-nested-parse::

   Streaming backend provider ABCs.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.backend.StageStats
   scipion_bridge.core.streaming.backend.CompiledPipeline
   scipion_bridge.core.streaming.backend.StreamingBackendProvider


Module Contents
---------------

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



.. py:class:: StreamingBackendProvider

   Bases: :py:obj:`abc.ABC`


   Abstract base for streaming execution backends.


   .. py:method:: compile(ir_sinks: List[scipion_bridge.core.streaming.ir.IROp]) -> CompiledPipeline
      :abstractmethod:


      Compile an IR DAG into a runnable CompiledPipeline.



