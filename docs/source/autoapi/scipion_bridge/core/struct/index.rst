scipion_bridge.core.struct
==========================

.. py:module:: scipion_bridge.core.struct

.. autoapi-nested-parse::

   Core Struct and Set abstractions and Arrow storage engine.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/core/struct/buffers/index
   /autoapi/scipion_bridge/core/struct/collection/index
   /autoapi/scipion_bridge/core/struct/exceptions/index
   /autoapi/scipion_bridge/core/struct/key_path/index
   /autoapi/scipion_bridge/core/struct/schema/index
   /autoapi/scipion_bridge/core/struct/set/index
   /autoapi/scipion_bridge/core/struct/storage/index
   /autoapi/scipion_bridge/core/struct/struct/index
   /autoapi/scipion_bridge/core/struct/utils/index


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.struct.Dim
   scipion_bridge.core.struct.IndexType


Exceptions
----------

.. autoapisummary::

   scipion_bridge.core.struct.UninitializedFieldError


Classes
-------

.. autoapisummary::

   scipion_bridge.core.struct.Entry
   scipion_bridge.core.struct.Schema
   scipion_bridge.core.struct.ArrayEntryBase
   scipion_bridge.core.struct.ArrayEntry
   scipion_bridge.core.struct.ArraySetEntry
   scipion_bridge.core.struct.RaggedArraySetEntry
   scipion_bridge.core.struct.SchemaEntry
   scipion_bridge.core.struct.SchemaSetEntry
   scipion_bridge.core.struct.CollectionEntry
   scipion_bridge.core.struct.Struct
   scipion_bridge.core.struct.Array
   scipion_bridge.core.struct.Arg
   scipion_bridge.core.struct.Set
   scipion_bridge.core.struct.Collection
   scipion_bridge.core.struct.KeyPath
   scipion_bridge.core.struct.StorageView


Functions
---------

.. autoapisummary::

   scipion_bridge.core.struct.concat


Package Contents
----------------

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



.. py:class:: KeyPath(root: Sequence[Tuple[str, IndexType]] = (('root', slice(None)), ))

   Bases: :py:obj:`Sequence`\ [\ :py:obj:`Tuple`\ [\ :py:obj:`str`\ , :py:obj:`IndexType`\ ]\ ]


   Represents an immutable navigation path to nested data structures or array-backed storage.

   A `KeyPath` models hierarchical field traversal combined with positional slicing
   or scalar indexing. It allows programmatic querying, projection, and manipulation
   of nested records, structs, tabular columns, or multidimensional arrays.

   Each path segment is stored as a `(name, index)` tuple where:
       - `name`: The name in the corresponding schema.
       - `index`: An `IndexType` indicating the selection across that dimension.
         This is typically initialized as an unbounded slice (`slice(None)`),
         which can subsequently be narrowed to a subslice or a concrete integer index.

   The path functions as an immutable sequence of components, supporting operations
   like chaining, relative slice composition, and offset projection without requiring
   eager resolution of the underlying container length when bounds are statically determinable.

   .. attribute:: components

      The underlying sequence
      of component identifier and index pairs.

      :type: tuple[tuple[str, IndexType], ...]

   .. attribute:: path

      The ordered attribute/field names traversed by the path.

      :type: tuple[str, ...]

   .. attribute:: indices

      The active slices or scalar indices for
      each step of the path.

      :type: tuple[IndexType, ...]

   .. rubric:: Examples

   Constructing and narrowing an attribute path:

   >>> path = KeyPath().append("users").append("orders")
   >>> path.path
   ('root', 'users', 'orders')

   Narrowing ranges along the path:

   >>> narrowed = path.narrow_slice(slice(0, 10)).narrow_slice(slice(2, 5))
   >>> narrowed.indices[-1]
   slice(2, 5, 1)

   Pinning the tail component to a scalar record index:

   >>> scalar_item = narrowed.narrow_index(1)
   >>> scalar_item.indices[-1]
   3


   .. py:attribute:: components


   .. py:property:: path
      :type: Tuple[str, Ellipsis]


      Returns the ordered tuple of component names traversed by the path.


   .. py:property:: indices
      :type: Tuple[IndexType, Ellipsis]


      Returns the tuple of active index/slice components along the path.


   .. py:method:: extend(path: KeyPath) -> KeyPath

      Concatenates another `KeyPath` onto this key path.

      Appends all components from `path` to the end of `self.components`,
      returning a new `KeyPath` instance.

      :param path: The `KeyPath` whose components should be concatenated.

      :returns: A new `KeyPath` instance containing the combined components.
      :rtype: KeyPath

      .. rubric:: Examples

      >>> p1 = KeyPath().append("users")
      >>> p2 = KeyPath((("orders", slice(0, 5)),))
      >>> combined = p1.extend(p2)
      >>> combined.path
      ('root', 'users', 'orders')
      >>> combined.indices[-1]
      slice(0, 5, None)



   .. py:method:: append(name: str) -> KeyPath

      Appends an attribute or field name to the key path.

      Extends the path to target a child attribute on an object, a struct field,
      or a set column. The new component is initialized with an unbounded
      slice (`slice(None)`), representing the entire range or collection until
      further narrowed.

      :param name: The name of the attribute, field, or column to append.

      :returns:

                A new `KeyPath` instance containing the existing path
                    components followed by `(name, slice(None))`.
      :rtype: KeyPath

      .. rubric:: Examples

      >>> path = KeyPath().append("user")
      >>> path.components[-1]
      ('user', slice(None, None, None))

      >>> nested = path.append("address").append("city")
      >>> nested.path
      ('root', 'user', 'address', 'city')



   .. py:method:: narrow_index(index: int, length: Optional[int] = None) -> KeyPath

      Narrows the terminal component's slice or index array to a concrete integer index.

      Composes a relative scalar index with the component currently held at the tail
      of the key path, projecting the sub-index into the parent's coordinate frame.

      Resolution rules:
          1. **Slice with Known Span or Sequence Length**: When the parent slice has
             non-negative bounds (`[start:stop]`), `span = max(0, stop - start)`.
             Alternatively, if `length` is provided, `span` is resolved from `length`.
             Bounds-checks `index` against `[-span, span - 1]` (`IndexError`), normalizes
             negative indices (`offset = span + index if index < 0 else index`), and
             projects `start + offset`.
          2. **Open-Ended Slice (`[start:]`) without Length**: If `start >= 0`,
             `stop is None`, and `index >= 0`, projects to `start + index`. Negative
             indices cannot be resolved without sequence length and raise `ValueError`.
          3. **Right-Anchored Slice (`[:stop]`) without Length**: If `start is None`,
             `stop < 0`, and `index < 0`, projects relative to the negative right edge
             as `stop + index`. Positive indices cannot be resolved without sequence
             length and raise `ValueError`.
          4. **Existing Index Array (`NDArray`)**: When the terminal component is already
             an index array, performs a scalar take: bounds-checks `index` against
             `len(parent_arr)` (handling negative indices) and yields `int(parent_arr[offset])`.
             Raises `IndexError` if out of bounds.
          5. **Scalar Index (`int`)**: Cannot be narrowed further; raises `ValueError`.

      :param index: The relative integer index to select from the existing component.
                    Accepts both positive and negative values where determinable.
      :param length: Optional known sequence length of the active dimension, used
                     for span resolution and bounds checking.

      :returns:

                A new `KeyPath` instance whose terminal component has been
                    narrowed to an integer index.
      :rtype: KeyPath

      :raises IndexError: If `index` falls outside valid bounds for a known span or length.
      :raises ValueError: In any of the following conditions:
          - The terminal component cannot be indexed (e.g. already an integer index).
          - The combination of slice anchors and `index` sign cannot be
            resolved statically without knowing the sequence length.
      :raises NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

      .. rubric:: Examples

      >>> path = KeyPath().append("users")  # root.users[:]
      >>> path.narrow_index(3).indices[-1]
      3

      >>> path.narrow_slice(slice(10, 20)).narrow_index(2).indices[-1]
      12

      >>> path.narrow_slice(slice(10, 20)).narrow_index(-1).indices[-1]
      19

      >>> path.narrow_indices([10, 20, 30]).narrow_index(1).indices[-1]
      20



   .. py:method:: narrow_slice(index: slice, length: Optional[int] = None) -> KeyPath

      Narrows the terminal component by composing it with a subslice.

      Projects a relative `slice` into the coordinate frame of the component
      currently held at the tail of the key path. Only slices with a step of 1
      are supported.

      Resolution rules:
          1. **Slice with Known Span or Sequence Length**: When the parent slice has
             non-negative bounds (`[start:stop]`), `span = max(0, stop - start)`.
             Alternatively, if `length` is provided, `span` is resolved from `length`.
             Composes the subslice via `index.indices(span)` and produces a canonical
             slice `slice(start + rel_start, start + rel_stop, 1)`. Clamps and normalizes
             negative or out-of-bounds indices automatically.
          2. **Unbounded / Symbolic Composition (without Length)**:
             - The new start is composed algebraically via `_add_bound(parent.start, index.start)`.
             - If `index.stop is None`, the new stop remains `parent.stop`.
             - If `index.stop >= 0`, the new stop is anchored to start: `_add_bound(parent.start, index.stop)`.
             - If `index.stop < 0`, the new stop is anchored to stop: `_add_bound(parent.stop, index.stop)`.
             - Mixed-sign composition (e.g. adding a negative offset to a non-negative anchor)
               raises `ValueError` when sequence length is unknown.
          3. **Existing Index Array (`NDArray`)**: Slices the index array in-place
             via `parent_arr[index]`.
          4. **Scalar Index (`int`)**: Cannot be sliced; raises `ValueError`.

      :param index: The relative subslice to compose with the existing slice or array.
                    Must have `step=1` (or `None`).
      :param length: Optional known sequence length of the active dimension, used
                     for span resolution and concrete slice computation.

      :returns: A new `KeyPath` instance with the narrowed terminal component.
      :rtype: KeyPath

      :raises ValueError: If `index.step != 1`, terminal component cannot be sliced,
          or mixed-sign composition cannot be resolved without sequence length.
      :raises NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

      .. rubric:: Examples

      >>> path = KeyPath().append("items")  # items[:]
      >>> path.narrow_slice(slice(2, 8)).indices[-1]
      slice(2, 8, 1)

      >>> path.narrow_slice(slice(2, 8)).narrow_slice(slice(1, 4)).indices[-1]
      slice(3, 6, 1)

      >>> path.narrow_slice(slice(5, None)).narrow_slice(slice(2, None)).indices[-1]
      slice(7, None, 1)

      >>> path.narrow_indices([10, 20, 30, 40, 50]).narrow_slice(slice(1, 4)).indices[-1]
      array([20, 30, 40])



   .. py:method:: narrow_indices(indices: Union[Sequence[int], numpy.typing.NDArray[numpy.integer]], length: Optional[int] = None) -> KeyPath

      Narrows the terminal component to an array of integer indices (gather/take).

      Composes relative integer indices with the component currently held at the tail
      of the key path, projecting sub-indices into the parent coordinate frame.

      Resolution rules:
          1. **Slice with Known Span or Sequence Length**: When the parent slice has
             non-negative bounds (`[start:stop]`), `span = max(0, stop - start)`.
             Alternatively, if `length` is provided, `span` is resolved from `length`.
             Checks all indices against `[-span, span - 1]` (`IndexError`), normalizes
             negative indices (`span + arr`), and projects `start + offset`.
          2. **Open-Ended Slice (`[start:]`) without Length**: When `start >= 0` and
             `stop is None`, non-negative indices are projected as `start + arr`. Any
             negative index raises `ValueError` because sequence length is required to
             compute negative offsets.
          3. **Slice with Negative Bounds without Length**: Cannot be resolved without
             sequence length; raises `ValueError`.
          4. **Existing Index Array (`NDArray`)**: Vectorized take/gather operation. Checks
             all indices against `[-len(parent_arr), len(parent_arr) - 1]` (`IndexError`),
             normalizes negative indices, and gathers from the parent array via
             `parent_arr[offset]`.
          5. **Scalar Index (`int`)**: Cannot be indexed into; raises `ValueError`.

      :param indices: Sequence or 1D integer array of relative indices to select.
      :param length: Optional known sequence length of the active dimension, used
                     for span resolution and bounds checking.

      :returns:

                A new `KeyPath` instance whose terminal component has been
                    narrowed to a 1D `np.int64` array of absolute indices.
      :rtype: KeyPath

      :raises IndexError: If any index falls outside valid bounds for a known span or length.
      :raises ValueError: In any of the following conditions:
          - The terminal component cannot be indexed (e.g. already a scalar index).
          - A negative index is provided for an open-ended slice when sequence
            length is unknown.
          - Slicing with negative bounds when sequence length is unknown.
      :raises NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

      .. rubric:: Examples

      >>> path = KeyPath().append("values")  # values[:]
      >>> path.narrow_indices([0, 2, 4], length=5).indices[-1]
      array([0, 2, 4])

      >>> path.narrow_slice(slice(10, 20)).narrow_indices([0, -1, 2]).indices[-1]
      array([10, 19, 12])

      >>> path.narrow_indices([10, 20, 30, 40]).narrow_indices([3, 1]).indices[-1]
      array([40, 20])



   .. py:method:: narrow_mask(mask: Union[Sequence[bool], numpy.typing.NDArray[numpy.bool_]], length: Optional[int] = None) -> KeyPath

      Narrows the terminal component using a boolean mask (filtering).

      Converts the boolean mask to active integer indices via `np.flatnonzero`
      and projects them into the parent coordinate frame.

      Resolution rules:
          1. **Identity Slice (`[:]`)**: If `length` is provided, validates
             `len(mask) == length` (raising `IndexError` on mismatch). Returns
             `np.flatnonzero(mask)`. If `length is None`, projects without length validation.
          2. **Known Fixed Span (`[start:stop]`)**: When both `start` and `stop` are
             non-negative, `span = max(0, stop - start)`. Validates `len(mask) == span`
             (`IndexError`) and projects `start + np.flatnonzero(mask)`.
          3. **Slice with Known Sequence Length**: When `length` is provided for an
             open-ended (`[start:]`) or negative-bound slice (e.g. `[:-2]`, `[-5:]`),
             `span` is resolved to `length`. Validates `len(mask) == span` (`IndexError`)
             and projects `start + np.flatnonzero(mask)`.
          4. **Open Right Bound (`[start:]`) without Length**: If `start >= 0` and
             `stop is None`, projects `start + np.flatnonzero(mask)` without length validation.
          5. **Negative Bounds without Length**: Raises `ValueError` because the span
             cannot be determined.
          6. **Existing Index Array (`NDArray`)**: Validates `len(mask) == len(parent_arr)`
             (`IndexError`) and filters the array in-place via `parent_arr[bool_mask]`.
          7. **Scalar Index (`int`)**: Cannot be masked; raises `ValueError`.

      :param mask: Sequence or 1D boolean array indicating elements to retain.
      :param length: Optional known sequence length of the active dimension, used
                     for span resolution and length validation.

      :returns:

                A new `KeyPath` instance whose terminal component has been
                    narrowed to a 1D `np.int64` array of absolute indices.
      :rtype: KeyPath

      :raises IndexError: If `len(mask)` does not match the known span or container
          length of the target component.
      :raises ValueError: In any of the following conditions:
          - The terminal component cannot be masked (e.g. a scalar index).
          - A mask is applied to a slice with negative bounds when sequence
            length is unknown.
      :raises NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

      .. rubric:: Examples

      >>> path = KeyPath().append("items")
      >>> path.narrow_mask([True, False, True]).indices[-1]
      array([0, 2])

      >>> slice_path = path.narrow_slice(slice(2, 7))  # items[2:7], span=5
      >>> slice_path.narrow_mask([True, False, False, True, False]).indices[-1]
      array([2, 5])

      >>> array_path = path.narrow_indices([10, 20, 30])
      >>> array_path.narrow_mask([True, False, True]).indices[-1]
      array([10, 30])



.. py:type:: IndexType
   :canonical: Union[slice, int, NDArray[np.int64]]


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



.. py:exception:: UninitializedFieldError

   Bases: :py:obj:`AttributeError`, :py:obj:`ValueError`


   Raised when attempting to access a field or element that has not been initialized or is null.

   Inherits from both AttributeError and ValueError so callers expecting either standard Python
   attribute lookup failure or Arrow null-value errors catch this consistently.


