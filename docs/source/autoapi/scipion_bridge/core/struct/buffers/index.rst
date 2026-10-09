scipion_bridge.core.struct.buffers
==================================

.. py:module:: scipion_bridge.core.struct.buffers

.. autoapi-nested-parse::

   Stateless helpers that build, reshape and fill the column buffers of storages.

   A column is a NumPy array whenever all of its rows are present and have the
   same shape. Only a column whose rows differ in shape, or which has missing
   rows, is an Awkward Array. This holds for static fields (whose element shape is
   known from the schema) and for dynamic fields (with ``None`` dimensions) alike;
   a dynamic field only *allows* rows of different shapes. While a dynamic column
   is written element by element, it is kept as a list of per-row buffers ("rows")
   and stacked into a single column when it is read as a whole.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.buffers.Column


Functions
---------

.. autoapisummary::

   scipion_bridge.core.struct.buffers.as_regular_numpy
   scipion_bridge.core.struct.buffers.check_castable
   scipion_bridge.core.struct.buffers.pad_to_capacity
   scipion_bridge.core.struct.buffers.has_numpy_layout
   scipion_bridge.core.struct.buffers.regularized
   scipion_bridge.core.struct.buffers.concat_columns
   scipion_bridge.core.struct.buffers.stack_rows
   scipion_bridge.core.struct.buffers.stack_sequence
   scipion_bridge.core.struct.buffers.split_rows
   scipion_bridge.core.struct.buffers.coerce_static_shape
   scipion_bridge.core.struct.buffers.broadcastable
   scipion_bridge.core.struct.buffers.expanded_shape
   scipion_bridge.core.struct.buffers.expand_to_fit
   scipion_bridge.core.struct.buffers.infer_outer_dims
   scipion_bridge.core.struct.buffers.nested_rows
   scipion_bridge.core.struct.buffers.assign_rows


Module Contents
---------------

.. py:data:: Column

.. py:function:: as_regular_numpy(data: Any, dtype: Any) -> Optional[numpy.ndarray]

   ``data`` as a NumPy array of ``dtype``, or None if it has no regular shape.

   Ragged nested sequences and values that are not numeric (strings aside)
   cannot be represented by a regular NumPy array; they are stored in Awkward
   Arrays instead.


.. py:function:: check_castable(source: numpy.dtype, target: Any, key: scipion_bridge.core.struct.key_path.KeyPath) -> None

   Raise if values of dtype ``source`` cannot be stored in a field of dtype ``target``.


.. py:function:: pad_to_capacity(column: numpy.ndarray, capacity: Optional[int], dtype: Any) -> numpy.ndarray
                 pad_to_capacity(column: awkward.Array, capacity: Optional[int], dtype: Any) -> awkward.Array

   Pad ``column`` along axis 0 to ``capacity`` rows.

   NumPy columns are padded with zeros, Awkward columns with missing rows.


.. py:function:: has_numpy_layout(column: awkward.Array) -> bool

   True if ``column`` is a NumPy array behind regular dimensions only.

   Such an array converts to NumPy without copying or inspecting the data.


.. py:function:: regularized(column: awkward.Array) -> Column

   ``column`` as a NumPy array if all its rows are present and of one shape.

   Used where a column is built from parts (rows, sequences); a column read
   back from storage is never checked again.


.. py:function:: concat_columns(columns: Sequence[Column]) -> Column

   Concatenate columns along axis 0.

   NumPy columns whose rows have one shape are joined with NumPy; any other
   mix (rows of different shapes, Awkward columns) gives an Awkward Array.
   Empty columns do not take part, so they never force the switch to Awkward.


.. py:function:: stack_rows(rows: List[Any]) -> Column

   Stack per-row buffers into a single column; None rows become missing rows.

   Rows of one shape give a NumPy array; rows of different shapes, or missing
   rows, give an Awkward Array.


.. py:function:: stack_sequence(rows: Sequence[Column]) -> Optional[Column]

   Stack a sequence of array rows, or None if some row is not an array.

   Rows of one shape give a NumPy array, otherwise an Awkward Array.


.. py:function:: split_rows(column: Column) -> List[Any]

   Split a column into a list of per-row buffers.

   Regular columns become writable NumPy row views (one copy at most, if the
   column is read-only); irregular or masked columns become Awkward rows,
   keeping missing rows as None.


.. py:function:: coerce_static_shape(arr: numpy.ndarray, entry_shape: Tuple[Optional[int], Ellipsis], key: scipion_bridge.core.struct.key_path.KeyPath, *, is_set: bool = False) -> numpy.ndarray

   Validate and adjust the shape of ``arr`` to the element shape of a static field.


.. py:function:: broadcastable(target: Tuple[int, Ellipsis], source: Tuple[int, Ellipsis]) -> bool

   True if arrays of the two shapes broadcast against each other.


.. py:function:: expanded_shape(shape: Tuple[int, Ellipsis], effective_idx: Tuple[scipion_bridge.core.struct.key_path.IndexType, Ellipsis]) -> Tuple[int, Ellipsis]

   Return the shape needed to hold ``effective_idx`` (``shape`` if it fits).


.. py:function:: expand_to_fit(buffer: numpy.ndarray, effective_idx: Tuple[scipion_bridge.core.struct.key_path.IndexType, Ellipsis]) -> numpy.ndarray

   Return ``buffer``, or a zero-padded copy large enough for ``effective_idx``.


.. py:function:: infer_outer_dims(entry: scipion_bridge.core.struct.schema.Entry, effective_idx: Tuple[scipion_bridge.core.struct.key_path.IndexType, Ellipsis], min_length: int = 0) -> Tuple[int, Ellipsis]

   Outer dimensions of a new column that has to hold ``effective_idx``.

   The first dimension is at least the capacity of a Set field and
   ``min_length`` (the length of the sibling columns).


.. py:function:: nested_rows(dims: Sequence[int]) -> List[Any]

   Nested lists of None rows with the given outer dimensions.


.. py:function:: assign_rows(rows: List[Any], effective_idx: Tuple[scipion_bridge.core.struct.key_path.IndexType, Ellipsis], data: Any) -> None

   Write ``data`` into the nested row lists at ``effective_idx``.

   Slice and index-array writes require one data element per selected row;
   the row lists never change their length here.


