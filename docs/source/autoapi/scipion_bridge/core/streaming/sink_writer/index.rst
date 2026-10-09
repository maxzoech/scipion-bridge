scipion_bridge.core.streaming.sink_writer
=========================================

.. py:module:: scipion_bridge.core.streaming.sink_writer

.. autoapi-nested-parse::

   Generic async sink writer protocol for streaming pipeline output.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.sink_writer.SinkWriter
   scipion_bridge.core.streaming.sink_writer.CallbackSinkWriter


Module Contents
---------------

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



