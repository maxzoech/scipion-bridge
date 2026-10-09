scipion_bridge.core.struct.schema
=================================

.. py:module:: scipion_bridge.core.struct.schema

.. autoapi-nested-parse::

   Schema definitions for Struct and Set data structures.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.struct.schema.SchemaConvertible
   scipion_bridge.core.struct.schema.Entry
   scipion_bridge.core.struct.schema.ArrayEntryBase
   scipion_bridge.core.struct.schema.SetEntryBase
   scipion_bridge.core.struct.schema.ArrayEntry
   scipion_bridge.core.struct.schema.ArraySetEntry
   scipion_bridge.core.struct.schema.RaggedArraySetEntry
   scipion_bridge.core.struct.schema.SchemaEntry
   scipion_bridge.core.struct.schema.SchemaSetEntry
   scipion_bridge.core.struct.schema.CollectionEntry
   scipion_bridge.core.struct.schema.Schema


Module Contents
---------------

.. py:class:: SchemaConvertible

   Abstract base for classes or objects convertible to a schema representation.


   .. py:method:: schema() -> Schema
      :classmethod:

      :abstractmethod:



   .. py:method:: default() -> SchemaConvertible
      :classmethod:

      :abstractmethod:


      Create a default, unspecialized instance from the type.



   .. py:property:: entry
      :type: Entry



   .. py:method:: convert_to_entry() -> Entry
      :abstractmethod:


      Convert this instance into a schema Entry tree representation.



   .. py:method:: print_schema() -> None
      :classmethod:



   .. py:property:: is_descriptor
      :type: bool

      :abstractmethod:


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch
      :abstractmethod:


      Export the initialized data to an Apache Arrow RecordBatch.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> SchemaConvertible
      :classmethod:

      :abstractmethod:


      Construct an instance from a RecordBatch produced by :meth:`to_arrow`.



.. py:class:: Entry

   Abstract base for all schema field entries.


   .. py:property:: is_static
      :type: bool

      :abstractmethod:


      True when the entry's shape is fully known at schema-creation time.


   .. py:method:: format_entry(name: str) -> str
      :abstractmethod:


      Return a human-readable label for *name* used by ``print_tree``.



   .. py:property:: children
      :type: Optional[Schema]


      Return the nested schema if this entry contains children, else None.


   .. py:attribute:: capacity
      :type:  Optional[int]
      :value: None



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry
      :abstractmethod:


      Transform this entry into its Set-vectorized entry representation.



.. py:class:: ArrayEntryBase

   Bases: :py:obj:`Entry`


   Shared behaviour for all array-backed entry types.


   .. py:attribute:: dtype
      :type:  numpy.dtype


   .. py:attribute:: shape
      :type:  Tuple[Optional[int], Ellipsis]


   .. py:property:: entry_name
      :type: str

      :abstractmethod:



   .. py:property:: is_static
      :type: bool


      Static if all dimensions are defined integers >= 0.


   .. py:method:: format_entry(name: str) -> str

      Return a human-readable label for *name* used by ``print_tree``.



.. py:class:: SetEntryBase

   Bases: :py:obj:`Entry`


   Abstract base for all schema field entries.


   .. py:attribute:: capacity
      :type:  Optional[int]
      :value: None



.. py:class:: ArrayEntry

   Bases: :py:obj:`ArrayEntryBase`


   An array field whose shape may or may not be fully static.


   .. py:property:: entry_name
      :type: str



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry

      Transform this entry into its Set-vectorized entry representation.



.. py:class:: ArraySetEntry

   Bases: :py:obj:`ArrayEntryBase`, :py:obj:`SetEntryBase`


   A fixed-shape array field inside a Set context.


   .. py:attribute:: capacity
      :type:  Optional[int]
      :value: None



   .. py:property:: is_static
      :type: bool


      Static if all dimensions are defined integers >= 0.


   .. py:property:: entry_name
      :type: str



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry

      Transform this entry into its Set-vectorized entry representation.



.. py:class:: RaggedArraySetEntry

   Bases: :py:obj:`ArrayEntryBase`, :py:obj:`SetEntryBase`


   A variable-shape array field inside a Set.


   .. py:attribute:: capacity
      :type:  Optional[int]
      :value: None



   .. py:property:: is_static
      :type: bool


      Static if all dimensions are defined integers >= 0.


   .. py:property:: entry_name
      :type: str



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry

      Transform this entry into its Set-vectorized entry representation.



.. py:class:: SchemaEntry

   Bases: :py:obj:`Entry`


   Wraps a nested struct type and its schema for record instantiation.


   .. py:attribute:: schema
      :type:  Schema


   .. py:property:: is_static
      :type: bool


      True when the entry's shape is fully known at schema-creation time.


   .. py:property:: children
      :type: Optional[Schema]


      Return the nested schema if this entry contains children, else None.


   .. py:method:: format_entry(name: str) -> str

      Return a human-readable label for *name* used by ``print_tree``.



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry

      Transform this entry into its Set-vectorized entry representation.



.. py:class:: SchemaSetEntry

   Bases: :py:obj:`SchemaEntry`, :py:obj:`SetEntryBase`


   Wraps a Set[Foo] container entry capable of instantiating Foo elements.


   .. py:attribute:: capacity
      :type:  Optional[int]
      :value: None



   .. py:property:: is_static
      :type: bool


      True when the entry's shape is fully known at schema-creation time.


   .. py:method:: format_entry(name: str) -> str

      Return a human-readable label for *name* used by ``print_tree``.



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry

      Transform this entry into its Set-vectorized entry representation.



.. py:class:: CollectionEntry

   Bases: :py:obj:`Entry`


   Wraps a statically-sized Collection container entry.


   .. py:attribute:: element_entry
      :type:  Entry


   .. py:attribute:: size
      :type:  int


   .. py:property:: is_static
      :type: bool


      True when the entry's shape is fully known at schema-creation time.


   .. py:property:: children
      :type: Optional[Schema]


      Return the nested schema if this entry contains children, else None.


   .. py:method:: format_entry(name: str) -> str

      Return a human-readable label for *name* used by ``print_tree``.



   .. py:method:: to_set_entry(capacity: Optional[int] = None) -> Entry

      Transform this entry into its Set-vectorized entry representation.



.. py:class:: Schema

   A tree of :class:`Entry` objects describing the storage layout of a Struct.


   .. py:attribute:: dtype
      :type:  Optional[Type]


   .. py:attribute:: fields
      :type:  Dict[str, Entry]


   .. py:attribute:: capacity
      :type:  Optional[int]
      :value: None



   .. py:method:: to_set_schema(capacity: Optional[int] = None) -> Schema

      Transform this schema into its Set-vectorized representation.



   .. py:property:: is_static
      :type: bool


      True when every field in the schema has a fixed shape.


   .. py:method:: tree_iter(*others: Schema, root: scipion_bridge.core.struct.key_path.KeyPath = KeyPath(root=())) -> Iterator[Tuple[Any, Ellipsis]]

      Yield (path, *entries) for all leaf array entries across this and optional other schemas.

      Strictly validates that all schemas have matching field keys and compatible hierarchy
      at every level.

      :param \*others: Additional Schema instances to traverse in parallel.
      :param root: Base KeyPath prefix for relative path accumulation.

      :raises ValueError: If field names do not match across schemas at any level.
      :raises TypeError: If a field is a nested branch in one schema but a leaf in another,
          or if a leaf entry is not an ArrayEntryBase.



   .. py:method:: print_tree(typename: Optional[str] = None) -> None

      Print the schema in a hierarchical tree format.



