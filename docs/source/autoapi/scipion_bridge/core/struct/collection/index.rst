scipion_bridge.core.struct.collection
=====================================

.. py:module:: scipion_bridge.core.struct.collection

.. autoapi-nested-parse::

   Statically indexed sequence container for Struct items backed by columnar storage.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.collection.T


Classes
-------

.. autoapisummary::

   scipion_bridge.core.struct.collection.Collection


Module Contents
---------------

.. py:data:: T

.. py:class:: Collection(size: int, items: Optional[Union[Sequence[T], Dict[int, T]]] = None, storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, dtype: Optional[Any] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Statically indexed sequence container for Struct items backed by columnar storage.


   .. py:attribute:: size


   .. py:property:: storage
      :type: scipion_bridge.core.struct.storage._BaseStorage



   .. py:method:: is_initialized(index: int) -> bool


   .. py:method:: initialized_indices() -> list[int]


   .. py:method:: default() -> scipion_bridge.core.struct.schema.SchemaConvertible
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: item_type() -> Type[scipion_bridge.core.struct.struct.Struct]
      :classmethod:


      Return the element Struct type of the Collection.



   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:

      :abstractmethod:



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



   .. py:method:: items() -> Iterator[Tuple[int, T]]


   .. py:method:: keys() -> list[int]


   .. py:method:: values() -> list[T]


   .. py:method:: to_set() -> scipion_bridge.core.struct.set.Set[T]


   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the Collection to a RecordBatch with one row per slot.

      Uninitialized slots, and fields not initialized in a slot, are null.
      The size is stored in the batch metadata.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct a Collection from a RecordBatch produced by :meth:`to_arrow`.



