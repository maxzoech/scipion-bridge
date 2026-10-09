scipion_bridge.backend.ray.sinks
================================

.. py:module:: scipion_bridge.backend.ray.sinks


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/backend/ray/sinks/tensorstore_sink/index


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.sinks.TensorStoreSinkWriter


Package Contents
----------------

.. py:class:: TensorStoreSinkWriter(path: Union[str, pathlib.Path])

   Bases: :py:obj:`scipion_bridge.core.streaming.sink_writer.SinkWriter`


   Asynchronous SinkWriter persisting SchemaConvertible items to Zarr groups via TensorStore.

   Implements the core reduction algebra:
   - SetEntryBase (ArraySetEntry) -> Concat / Append along axis 0.
   - ArrayEntry -> Last-Writer-Wins overwrite.


   .. py:attribute:: path


   .. py:method:: write(item: scipion_bridge.core.struct.schema.SchemaConvertible) -> None
      :async:


      Write an incoming batch/item into TensorStore Zarr datasets.



   .. py:method:: finalize() -> None
      :async:


      Finalize all open stores.



