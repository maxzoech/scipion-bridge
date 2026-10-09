scipion_bridge.core.struct.utils.arrow_utils
============================================

.. py:module:: scipion_bridge.core.struct.utils.arrow_utils

.. autoapi-nested-parse::

   Utilities for bridging scipion-bridge schemas/structures with Apache Arrow.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.utils.arrow_utils.pc


Functions
---------

.. autoapisummary::

   scipion_bridge.core.struct.utils.arrow_utils.schema_to_arrow_schema
   scipion_bridge.core.struct.utils.arrow_utils.build_tensor_array
   scipion_bridge.core.struct.utils.arrow_utils.build_ragged_array
   scipion_bridge.core.struct.utils.arrow_utils.build_multidim_ragged_array
   scipion_bridge.core.struct.utils.arrow_utils.is_regular_awkward
   scipion_bridge.core.struct.utils.arrow_utils.read_ragged
   scipion_bridge.core.struct.utils.arrow_utils.extract_field_from_arrow
   scipion_bridge.core.struct.utils.arrow_utils.get_nested_arrow_field
   scipion_bridge.core.struct.utils.arrow_utils.unwrap_extension_for_compute
   scipion_bridge.core.struct.utils.arrow_utils.rewrap_extension_after_compute
   scipion_bridge.core.struct.utils.arrow_utils.slice_arrow_array
   scipion_bridge.core.struct.utils.arrow_utils.arrow_to_numpy
   scipion_bridge.core.struct.utils.arrow_utils.leaf_to_arrow
   scipion_bridge.core.struct.utils.arrow_utils.leaf_from_arrow
   scipion_bridge.core.struct.utils.arrow_utils.regular_numpy_to_arrow
   scipion_bridge.core.struct.utils.arrow_utils.is_regular_arrow
   scipion_bridge.core.struct.utils.arrow_utils.regular_arrow_to_numpy
   scipion_bridge.core.struct.utils.arrow_utils.nest_columns
   scipion_bridge.core.struct.utils.arrow_utils.leaves_to_batch
   scipion_bridge.core.struct.utils.arrow_utils.null_row
   scipion_bridge.core.struct.utils.arrow_utils.find_nested_column


Module Contents
---------------

.. py:data:: pc
   :type:  Any

.. py:function:: schema_to_arrow_schema(schema: scipion_bridge.core.struct.schema.Schema) -> pyarrow.Schema

   Recursively convert a scipion-bridge Schema into a pyarrow.Schema.


.. py:function:: build_tensor_array(data: numpy.ndarray, shape: Tuple[int, Ellipsis], dtype: numpy.dtype, mask: Optional[Sequence[bool] | numpy.typing.NDArray] = None) -> pyarrow.ExtensionArray

   Compile a contiguous multidimensional NumPy array into an Arrow FixedShapeTensorArray with optional validity bitmask.


.. py:function:: build_ragged_array(chunks: Sequence[Optional[numpy.ndarray]], dtype: numpy.dtype) -> pyarrow.ListArray

   Compile a list of variable-length NumPy arrays into an Arrow ListArray using offsets and validity bitmask.


.. py:function:: build_multidim_ragged_array(chunks: Sequence[Optional[numpy.ndarray]], dtype: numpy.dtype) -> Union[pyarrow.ListArray, pyarrow.LargeListArray]

   Compile a sequence of 2D NumPy arrays into a nested Arrow ListArray[ListArray].

   Optimization:
   Constructing the Arrow ListArray hierarchy directly via contiguous NumPy buffers
   and offsets avoids Awkward Array's C++ `fromiter` traversal, which inspects
   every float scalar individually (~6 µs per scalar).


.. py:function:: is_regular_awkward(arr: awkward.Array) -> bool

   Check if an Awkward array has regular (non-jagged) dimensions at all depths.


.. py:function:: read_ragged(sliced: Any, entry: scipion_bridge.core.struct.schema.RaggedArraySetEntry, offset: Any) -> Any

   Resolve a read on a RaggedArraySetEntry from an Awkward array slice.


.. py:function:: extract_field_from_arrow(col: pyarrow.Array, field_name: str) -> pyarrow.Array

   Recursively traverse StructArray or ListArray layers to extract a named child field.


.. py:function:: get_nested_arrow_field(col: pyarrow.Array, path: scipion_bridge.core.struct.key_path.KeyPath) -> pyarrow.Array

   Traverse nested StructArray / ListArray layers along a KeyPath to resolve the leaf field array.


.. py:function:: unwrap_extension_for_compute(arr: pyarrow.Array) -> Tuple[pyarrow.Array, Optional[pyarrow.DataType]]

   Unwrap leaf FixedShapeTensorArray to storage array for PyArrow compute kernels.


.. py:function:: rewrap_extension_after_compute(arr: pyarrow.Array, ext_type: Optional[pyarrow.DataType]) -> pyarrow.Array

   Re-wrap storage array back into FixedShapeTensorArray after compute operations.


.. py:function:: slice_arrow_array(arr: pyarrow.Array, offset: Any) -> pyarrow.Array

   Apply multi-dimensional offset tuple using PyArrow slicing and pc.list_slice.


.. py:function:: arrow_to_numpy(col: pyarrow.Array, shape: Tuple[Optional[int], Ellipsis]) -> numpy.ndarray

   Convert an Arrow array (ExtensionArray, ListArray, or primitive) to an N-D NumPy array.


.. py:function:: leaf_to_arrow(data: Union[numpy.ndarray, awkward.Array], entry: scipion_bridge.core.struct.schema.ArrayEntryBase) -> pyarrow.Array

   Convert a leaf column buffer into an Arrow array.

   Axis 0 of ``data`` holds the rows and its last axes the shape of
   ``entry``. Axes in between belong to nested Sets; their length varies
   per row, so each becomes a list level instead of a tensor dimension.

   Static columns become FixedShapeTensorArrays (or primitive arrays for 1-D
   buffers). NumPy columns of dynamic fields become nested FixedSizeListArrays,
   which keep the row shape, and Awkward columns become (nested) list arrays.
   Contiguous NumPy buffers are wrapped without copying.


.. py:function:: leaf_from_arrow(column: pyarrow.Array, entry: scipion_bridge.core.struct.schema.ArrayEntryBase) -> Union[numpy.ndarray, awkward.Array]

   Convert an Arrow array produced by :func:`leaf_to_arrow` back into a buffer.

   Static columns are always restored as NumPy arrays. For dynamic fields, the
   Arrow type and null count decide: null-free nested fixed-size lists (regular
   rows) are restored as NumPy arrays without copying where Arrow allows it,
   null-free nested Sets level by level, and anything else (ragged or
   nullable columns) as an Awkward Array. Nested Sets of equal length in every
   row are restored as a NumPy array, otherwise as a ragged Awkward array.


.. py:function:: regular_numpy_to_arrow(data: numpy.ndarray) -> pyarrow.Array

   Wrap a NumPy column as nested FixedSizeListArrays, one level per row axis.

   The type matches what ``ak.to_arrow`` produces for a regular Awkward
   Array, so columns exported either way concatenate. A contiguous buffer is
   wrapped without copying.


.. py:function:: is_regular_arrow(column: pyarrow.Array) -> bool

   True if ``column`` is null-free nested fixed-size lists of numbers.


.. py:function:: regular_arrow_to_numpy(column: pyarrow.Array) -> numpy.ndarray

   Restore a column of :func:`is_regular_arrow` as an N-D NumPy array.

   ``flatten`` respects the offset of sliced arrays. Numeric values are
   adopted without copying; such arrays are read-only.


.. py:function:: nest_columns(schema: scipion_bridge.core.struct.schema.Schema, leaves: Dict[scipion_bridge.core.struct.key_path.KeyPath, pyarrow.Array], root: scipion_bridge.core.struct.key_path.KeyPath = KeyPath(root=())) -> Dict[str, pyarrow.Array]

   Assemble leaf arrays into top-level columns, nesting children as StructArrays.

   Fields without any initialized leaf are omitted.


.. py:function:: leaves_to_batch(schema: scipion_bridge.core.struct.schema.Schema, leaves: Dict[scipion_bridge.core.struct.key_path.KeyPath, pyarrow.Array], metadata: Dict[bytes, bytes]) -> pyarrow.RecordBatch

   Assemble leaf arrays of equal length into a RecordBatch of nested columns.


.. py:function:: null_row(example: pyarrow.Array) -> pyarrow.Array

   A single null row of the type of ``example`` with valid child values.

   ``pa.nulls`` fills the children of fixed-size lists with nulls, which
   violates non-nullable child fields; the child values of ``example`` are
   reused instead.


.. py:function:: find_nested_column(batch: pyarrow.RecordBatch, path: scipion_bridge.core.struct.key_path.KeyPath) -> Optional[pyarrow.Array]

   Resolve the leaf array at ``path`` in a batch built by :func:`nest_columns`.

   Returns None if the leaf was not initialized when the batch was built.


