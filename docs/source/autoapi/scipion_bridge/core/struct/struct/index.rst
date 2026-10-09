scipion_bridge.core.struct.struct
=================================

.. py:module:: scipion_bridge.core.struct.struct

.. autoapi-nested-parse::

   Descriptor and metaprogramming layer for Struct definition and materialization.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.struct.Dim
   scipion_bridge.core.struct.struct.T


Classes
-------

.. autoapisummary::

   scipion_bridge.core.struct.struct.Arg
   scipion_bridge.core.struct.struct.Array
   scipion_bridge.core.struct.struct.Trait
   scipion_bridge.core.struct.struct.Struct


Functions
---------

.. autoapisummary::

   scipion_bridge.core.struct.struct.struct_leaf_arrays
   scipion_bridge.core.struct.struct.leaf_columns
   scipion_bridge.core.struct.struct.write_struct_row


Module Contents
---------------

.. py:class:: Arg(value: Optional[Union[int, Arg]] = None, *, name: Optional[str] = None)

   Class-level dimension specification or named parameter.


   .. py:attribute:: name
      :value: None



   .. py:method:: new(value: Optional[Union[Arg, int]] = None, *, name: Optional[str] = None) -> Arg
      :classmethod:



   .. py:property:: value
      :type: Optional[int]



   .. py:method:: validate(other: Any) -> None


   .. py:property:: is_static
      :type: bool



.. py:type:: Dim
   :canonical: Arg


.. py:data:: T

.. py:class:: Array(dtype: Optional[Union[numpy.dtype, type, str]] = None, *, shape: Optional[Union[Tuple[Union[Dim, int, None], Ellipsis], list]] = None, is_scalar: bool = False)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Descriptor and schema representation for array attributes on Struct classes.


   .. py:attribute:: is_scalar
      :value: False



   .. py:attribute:: shape_spec
      :type:  Tuple[Dim, Ellipsis]
      :value: ()



   .. py:property:: shape
      :type: Tuple[Optional[int], Ellipsis]



   .. py:property:: dtype
      :type: Optional[numpy.dtype]



   .. py:method:: default() -> scipion_bridge.core.struct.schema.SchemaConvertible
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:

      :abstractmethod:



   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the initialized data to an Apache Arrow RecordBatch.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct an instance from a RecordBatch produced by :meth:`to_arrow`.



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



.. py:class:: Trait(*args: Any, **kwargs: Any)

   Specification layer: accumulates fields, dimensions, and shape descriptors.


.. py:class:: Struct(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`Trait`, :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:method:: default() -> Struct
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:



   .. py:property:: storage
      :type: scipion_bridge.core.struct.storage._BaseStorage



   .. py:method:: is_initialized(field_name: str) -> bool

      Check if a field has been initialized in storage.



   .. py:method:: initialized_fields() -> list[str]

      Return the names of all fields that have been initialized.



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the initialized fields to a RecordBatch with a single row.

      For a view (a Set element, a Collection item or a nested field), only
      the data of the view is exported. Nested Sets become nested list
      columns.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct a Struct owning its storage from a batch of :meth:`to_arrow`.



.. py:function:: struct_leaf_arrays(storage: scipion_bridge.core.struct.storage._BaseStorage, schema: scipion_bridge.core.struct.schema.Schema) -> Dict[scipion_bridge.core.struct.key_path.KeyPath, pyarrow.Array]

   Export the initialized leaves of the Struct at ``storage.root`` as 1-row arrays.


.. py:function:: leaf_columns(batch: pyarrow.RecordBatch, schema: scipion_bridge.core.struct.schema.Schema) -> Dict[scipion_bridge.core.struct.key_path.KeyPath, _LeafColumn]

   Convert the leaf columns of ``batch`` into buffers, with their validity per row.

   Every column is converted as a whole: Awkward ignores the offset of sliced
   Arrow arrays with a validity bitmap, so rows are indexed afterwards.


.. py:function:: write_struct_row(storage: scipion_bridge.core.struct.storage._BaseStorage, columns: Dict[scipion_bridge.core.struct.key_path.KeyPath, _LeafColumn], row: int) -> None

   Write row ``row`` of the leaf columns into the Struct at ``storage.root``.

   Leaves that are missing from the columns or null in the row stay
   uninitialized.


