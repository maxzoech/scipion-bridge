scipion_bridge.core.streaming.sink
==================================

.. py:module:: scipion_bridge.core.streaming.sink

.. autoapi-nested-parse::

   Terminal output and checkpoint node for streaming pipelines.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.sink.Sink


Module Contents
---------------

.. py:class:: Sink(writer: Union[scipion_bridge.core.streaming.sink_writer.SinkWriter, Callable[[Any], Any]], upstream: Optional[List[scipion_bridge.core.streaming.node.Node]] = None)

   Bases: :py:obj:`scipion_bridge.core.streaming.node.Node`


   Terminal output or checkpoint node backed by an async SinkWriter.


   .. py:method:: lower(ctx: scipion_bridge.core.streaming.node.LoweringContext) -> scipion_bridge.core.streaming.ir.IROp

      Polymorphically lower this surface Node to its low-level IR representation.



