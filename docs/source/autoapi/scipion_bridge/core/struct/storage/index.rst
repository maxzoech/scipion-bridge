scipion_bridge.core.struct.storage
==================================

.. py:module:: scipion_bridge.core.struct.storage

.. autoapi-nested-parse::

   Storage engines for Struct and Set data structures.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.storage.FieldKey


Classes
-------

.. autoapisummary::

   scipion_bridge.core.struct.storage.StorageView
   scipion_bridge.core.struct.storage.RootEngine


Module Contents
---------------

.. py:data:: FieldKey

.. py:class:: StorageView(root: scipion_bridge.core.struct.key_path.KeyPath = KeyPath(), parent: Optional[_BaseStorage] = None)

   Bases: :py:obj:`_BaseStorage`


   A reference to some other storage.

   Schema-based types like `Struct` or `Set` are internally backed by an array
   storage (e.g. numpy or Arrow).



   .. py:method:: is_initialized(key: scipion_bridge.core.struct.key_path.KeyPath) -> bool

      Return True if the field identified by key has been initialized in storage.



   .. py:method:: clear(key: Optional[scipion_bridge.core.struct.key_path.KeyPath] = None) -> None

      Clear all stored data under key or root.



   .. py:method:: get_length(key: Optional[scipion_bridge.core.struct.key_path.KeyPath] = None) -> Optional[int]

      Return active sequence length under key/root, or None if uninitialized.



   .. py:method:: read(key: scipion_bridge.core.struct.key_path.KeyPath, entry: scipion_bridge.core.struct.schema.Entry) -> Any

      Read data for the specified schema entry at this storage's index.



   .. py:method:: write(key: scipion_bridge.core.struct.key_path.KeyPath, entry: scipion_bridge.core.struct.schema.Entry, data: Any) -> None

      Write data for the specified schema entry at this storage's index.



.. py:class:: RootEngine(root: scipion_bridge.core.struct.key_path.KeyPath = KeyPath(), parent: Optional[_BaseStorage] = None)

   Bases: :py:obj:`_BaseStorage`


   Mutable in-memory storage engine backed by NumPy and Awkward Arrays.

   The root of a storage tree: it owns the data of a Struct or Set, and the
   StorageViews of nested fields and selections resolve to it.

   Every leaf field of the schema is stored as one column, keyed by its field
   path (without indices). A column lives either in ``_data`` as a single
   buffer, or, while a dynamic field is written element by element, in
   ``_chunks`` as a list of per-row buffers that is stacked into a buffer on
   the first read of the whole column.

   The type of a buffer follows from the data, not from the schema: it is a
   NumPy array whenever all rows are present and have the same shape, and an
   Awkward Array only when rows differ in shape (possible for dynamic fields,
   which have ``None`` dimensions) or some rows are missing. A buffer becomes
   Awkward only when the data requires it, e.g. a row of a different shape is
   written or a ragged column is concatenated; reads never convert it back.
   Where a column is built from parts (stacking rows, ``concat``, restoring
   from Arrow), regular data is built as NumPy directly.

   Concurrent element-wise access is supported as long as every thread reads
   and writes its own elements (e.g. ``Op.map_element``): writes into
   disjoint rows of an existing column touch disjoint memory and are
   lock-free. Transitions that replace a whole column object (allocation,
   copy-on-write of a read-only buffer, splitting a ragged column into rows,
   materializing pending rows) run once under a lock with double-checked
   locking. Whole-column writes and buffer expansion are not thread-safe.


   .. py:method:: is_initialized(key: scipion_bridge.core.struct.key_path.KeyPath) -> bool

      Return True if the field identified by key has been initialized in staging.



   .. py:method:: clear(key: Optional[scipion_bridge.core.struct.key_path.KeyPath] = None) -> None

      Clear all stored data under key or root.



   .. py:method:: get_length(key: Optional[scipion_bridge.core.struct.key_path.KeyPath] = None) -> Optional[int]

      Return the active sequence length of data stored under key or root.

      All columns below a container share their outer length, so the length
      is taken from the first column found. A missing element has length 0.



   .. py:method:: read(key: scipion_bridge.core.struct.key_path.KeyPath, entry: scipion_bridge.core.struct.schema.Entry) -> Any

      Read data for the specified schema entry at this storage's index.



   .. py:method:: write(key: scipion_bridge.core.struct.key_path.KeyPath, entry: scipion_bridge.core.struct.schema.Entry, data: Any) -> None

      Write data for the specified schema entry at this storage's index.



