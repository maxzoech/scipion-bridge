scipion_bridge.core.struct.set
==============================

.. py:module:: scipion_bridge.core.struct.set

.. autoapi-nested-parse::

   Set container implementation.

   A Set is a sequence container for Struct items, stored as flattened
   N-dimensional arrays across the outer batch dimension.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.set.T


Classes
-------

.. autoapisummary::

   scipion_bridge.core.struct.set.Set


Functions
---------

.. autoapisummary::

   scipion_bridge.core.struct.set.concat


Module Contents
---------------

.. py:data:: T

.. py:class:: Set(items: Sequence[T] = (), capacity: Optional[Union[int, scipion_bridge.core.struct.struct.Arg]] = None, storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Sequence container for Struct instances backed by Apache Arrow columnar storage.


   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:



   .. py:method:: item_type() -> Type[scipion_bridge.core.struct.struct.Struct]
      :classmethod:


      Return the element Struct type of the Set.



   .. py:property:: capacity
      :type: Optional[int]



   .. py:property:: storage
      :type: scipion_bridge.core.struct.storage._BaseStorage



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



   .. py:method:: default() -> Set[T]
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the visible rows of the Set to an Apache Arrow RecordBatch.

      Every top-level field becomes a column; nested Structs and Sets become
      StructArray columns. Uninitialized fields are omitted. For a view, only
      the rows of the view are exported, and contiguous buffers are wrapped
      without copying.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct a Set from a RecordBatch produced by :meth:`to_arrow`.

      Column buffers are adopted without copying where Arrow allows it; such
      buffers may be read-only and are copied on the first in-place write.



   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


.. py:function:: concat(sets: Sequence[Set[T]]) -> Set[T]

   Concatenate multiple Sets with identical schemas along axis 0.

   :param sets: A non-empty sequence of Set instances to concatenate.

   :returns: A new Set containing the concatenated data.


