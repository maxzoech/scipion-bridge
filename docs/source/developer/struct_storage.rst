.. _struct-storage:

Struct Storage
--------------

This article describes the storage engine behind the array types: how
containers translate attribute access and selections into reads and writes,
how columns are stored, and which invariants the engine maintains. It is the
place to start when investigating a bug in reading, writing, concatenating or
serializing container data. The user-facing behavior is described in
:doc:`/user_guide/structs/index`.

.. contents:: On this page
   :local:
   :depth: 2

Overview
^^^^^^^^

``Struct``, ``Set`` and ``Collection`` hold no arrays themselves. Each instance
holds a *schema* (the fields traced from the class annotations) and a *storage*
(``instance._storage``). Every attribute access, slice and column read is
translated into a ``read`` or ``write`` on that storage. The storage is
implemented in two modules:

* ``core/struct/storage.py``: the storage classes ``_BaseStorage``,
  ``StorageView`` and ``RootEngine``.
* ``core/struct/buffers.py``: stateless helpers that build, pad, split and
  stack column buffers. They do not depend on the storage classes and are
  tested directly in ``tests/scipion_bridge/struct/test_buffers.py``.

The storage tree
^^^^^^^^^^^^^^^^

A ``RootEngine`` owns the data. Everything derived from a container (a nested
field, a row, a slice, a mask selection) shares that data through a
``StorageView``. A view is only a ``KeyPath`` (its ``root``) and a parent. It
forwards every operation to ``root_storage``, the ``RootEngine`` at the top of
the chain:

.. code-block:: text

    particles = B.Set[Particle](...)    RootEngine          root
    particles[1:3]                      └─ StorageView      root[1:3]
    particles[1:3][0]                      └─ StorageView   root[1]
    particles[1:3][0].ctf                     └─ StorageView root[1].ctf

Views are created with ``append(name)`` for a field and with ``narrow_index``,
``narrow_slice``, ``narrow_indices`` and ``narrow_mask`` for a selection. The
narrowing composes with the index already on the path, so the view always holds
an absolute position in the root's columns: row 0 of ``particles[1:3]`` is
``root[1]``. A boolean mask is turned into an index array
(``particles[mask]`` has the root ``root[[0, 2]]``), so the engine never sees
masks.

``_BaseStorage`` defines the interface. ``read``, ``write``, ``clear``,
``get_length`` and ``is_initialized`` are abstract. ``__contains__``
(``key in storage``), the ``narrow_*`` methods and ``concat`` are implemented
on top of them. A new storage, including a test mock, has to implement all five
abstract methods; Python refuses to instantiate it otherwise.

Key paths and columns
^^^^^^^^^^^^^^^^^^^^^

A ``KeyPath`` is a sequence of ``(name, index)`` components that starts with
``root``. ``RootEngine`` splits it into two parts:

* the **field key**, the names without ``root``, which identifies the column;
* the **effective index**, the indices of the components with the no-op
  ``slice(None)`` entries removed, which selects the part of the column.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Access
     - Field key
     - Effective index
   * - ``particles["score"]``
     - ``("score",)``
     - ``()``
   * - ``particles[1].pixels``
     - ``("pixels",)``
     - ``(1,)``
   * - ``particles[1:3]["score"]``
     - ``("score",)``
     - ``(slice(1, 3),)``
   * - ``particles[mask]["score"]``
     - ``("score",)``
     - ``(array([0, 2]),)``
   * - ``classes[1].particles["pixels"]``
     - ``("particles", "pixels")``
     - ``(1,)``

Every leaf field of the schema is one column, stored under its field key.
Nested structs add their field names to the key. A nested ``Set`` adds a
dimension: in a ``Set[Class2D]``, the column ``("particles", "pixels")`` has
one entry per class, and each entry holds the pixels of all particles of that
class. A ``Collection`` stores each slot under its own key
(``("0", "pixels")``, ``("1", "pixels")``, ...), which makes it row-wise.

The type of a column follows from the **data**, not from the schema. This is
the central invariant of the engine:

    A column is an ``np.ndarray`` whenever all of its rows are present and have
    the same shape. It is an ``ak.Array`` only when rows differ in shape or
    some rows are missing.

The schema only decides what is *allowed*: a static field (shape fully known,
scalars) always has rows of one shape, while a dynamic field (a ``None``
dimension) may have rows of different shapes.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Data
     - Column
     - Example (3 elements)
   * - Static field
     - NumPy ``(*outer, *element_shape)``
     - ``score: float`` → ``(3, 1)``
   * - Dynamic field, all rows of one shape
     - NumPy ``(*outer, *row_shape)``
     - ``pixels``, all 64 × 64 → ``(3, 64, 64)``
   * - Dynamic field, rows of different shapes
     - Awkward Array
     - ``pixels``, 64 × 64 and 32 × 32 → ``3 * var * var * float32``
   * - Dynamic field, some rows missing
     - Awkward Array with option type
     - ``pixels``, row 1 unwritten → ``3 * option[var * 2 * float32]``
   * - Below nested ``Set`` s of one length
     - NumPy, with one axis for the nested ``Set``
     - ``Set[Class2D]`` with 2 particles per class → ``particles.score``:
       ``(2, 2, 1)``
   * - Below nested ``Set`` s of different lengths
     - Awkward Array (the nested dimension is ragged)
     - ``Set[Class2D]`` with 1 and 2 particles → ``particles.score``:
       ``2 * var * 1 * float64``

A column becomes Awkward only when the data requires it: a row of a different
shape is written, a ragged column is concatenated, or rows stay missing. The
switch is one-way: reads never convert an Awkward column back, because checking
the regularity of a large Awkward Array on every read would cost what the
invariant is meant to save. Where a column is *built* from parts, regular data
is built as NumPy directly: ``stack_rows`` and ``stack_sequence`` when rows are
joined, ``concat_columns`` in ``concat``, and ``leaf_from_arrow`` when a column
is restored from Arrow. A whole-column write of an Awkward Array that only wraps
a NumPy array (``RegularArray`` and ``NumpyArray`` layouts, checked by
``has_numpy_layout`` without touching the data) is unwrapped without copying.

The reason is speed. Selecting rows of an Awkward Array (``buffer[mask]``) goes
through Awkward's generic indexing, which for large image stacks is orders of
magnitude slower than NumPy's, and converting an Awkward column to NumPy can
copy it. Keeping regular columns in NumPy makes masks, ``concat`` and
serialization cost what they cost in NumPy.

All columns of one container share their outer length; the engine relies on
this to allocate new columns at full length and to answer ``get_length``.

The two states of a column
^^^^^^^^^^^^^^^^^^^^^^^^^^

A column is in one of two dictionaries of the engine:

* ``_data``: the column as one buffer (NumPy or Awkward, following the
  invariant above), its normal state;
* ``_chunks``: a *pending* column, a list with one buffer per row, where a
  row that was never written is ``None``.

Only dynamic columns become pending. A row written into a dynamic column may
have a different shape than the others, which neither a NumPy array nor an
(immutable) Awkward Array can take in place. Instead, the first element write
splits the column into rows, and later writes replace rows in the list. The
rows are joined into one buffer again when more than one element is read:

.. code-block:: text

                     element write to a dynamic field
                       (split_rows, assign_rows)
           ┌─────────────────────────────────────────────┐
           │                                             ▼
     _data: buffer                               _chunks: rows
           ▲                                             │
           └─────────────────────────────────────────────┘
                 read of a column, slice or index array
                         (stack_rows)

    whole-column write: replaces either state with a new buffer in _data

A loop of element writes therefore costs one split and one stack, not one
rebuild of the column per write. ``split_rows`` keeps regular columns as
writable NumPy row views (copying a read-only column once), and turns ragged
or masked columns into Awkward rows. ``stack_rows`` uses ``np.stack`` when all
rows are NumPy arrays of the same shape. Otherwise it joins them with
``ak.concatenate`` plus ``ak.unflatten``, masking the ``None`` rows; if no row
is missing and the result turns out regular (e.g. Awkward rows that all have
the same shape), it is converted to NumPy (``regularized``).

Static columns stay in ``_data`` and are written in place.

Reading
^^^^^^^

``RootEngine.read(key, entry)`` works in three steps:

1. If the column is pending and every index is an integer, the element is
   taken directly from the row list, without stacking the rows.
2. Otherwise a pending column is stacked into a buffer first
   (``_materialize_pending``).
3. The buffer is indexed with the effective index. Without an index, the
   whole buffer is returned.

A column that does not exist raises ``UninitializedFieldError``, and so does a
*missing* element: one that reads back as ``None``, either a ``None`` row of a
pending column or a masked row of an Awkward column. An *empty* result is
data, not a missing element. A selection with no rows, for example
``particles[assignments == k]`` for a class without particles, returns empty
columns and can be assigned to another struct like any other set. Static
columns are allocated with zeros, so a missing element of a static field reads
as zeros.

``is_initialized(key)`` (and ``key in storage``) is true if any column lies
at or below the field key. That way a nested struct counts as initialized when
one of its fields is. The engine caches the set of field keys that are a
column or a container of one, and rebuilds it when the set of columns
changes, so that repeated queries do not scan every column.

Writing
^^^^^^^

``RootEngine.write(key, entry, data)`` first checks the dtype: values must
cast to the field's dtype with ``same_kind`` casting (so writing floats to an
integer field raises ``TypeError``). A list of arrays is checked through its
first element, and Awkward Arrays and empty sequences are not checked. Then
one of three paths runs:

**Whole column** (no effective index). For a ``Set`` field, the length is
checked first: it may not exceed the capacity of the field, and without a
capacity it must equal the length of the sibling columns, pending ones
included. The sibling check is an assertion and is skipped under ``python -O``.
The buffer is then built from the data:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Data
     - Buffer
   * - Awkward Array
     - rejected for a static field; unwrapped to NumPy without copying if it
       only wraps a NumPy array; stored as is otherwise
   * - empty ``list`` / ``tuple``
     - empty NumPy column of shape ``(0, *element_shape)``, with length 0 for
       dynamic dimensions
   * - ``list`` / ``tuple`` of arrays of one shape
     - ``np.stack``
   * - ``list`` / ``tuple`` of arrays of different shapes
     - ragged Awkward Array (NumPy if the rows turn out to be regular)
   * - other ``list`` / ``tuple``
     - ``np.asarray`` with the field's dtype; an Awkward Array if the values
       are ragged
   * - scalar or array
     - ``np.asarray`` with the field's dtype. For a static field, the shape
       must end in the element shape

Columns built from a sequence are padded to the capacity of the field: with
zeros for NumPy columns and with missing rows for Awkward columns.

**Element of a static field.** The NumPy buffer is written in place, which is
why concurrent writes to different rows need no lock. If the column does not
exist yet, it is allocated with zeros. Its outer length is the largest of the
capacity of the field, the length of the sibling columns and the index being
written. A column that is too short for the index is copied into a larger
buffer. A read-only buffer (adopted from Arrow, see below) is copied once.
The data must broadcast to the selected part of the buffer, or a
``ValueError`` is raised.

**Element of a dynamic field.** The column becomes pending, as described
above, also when it is a NumPy array, and ``assign_rows`` writes the data into
the row list. Splitting a writable NumPy column gives row views without copying;
a read-only one (adopted from Arrow) is copied once:

* An integer index replaces one row. A list that is too short is extended with
  ``None`` rows first.
* A slice or an index array needs exactly one data element per selected row,
  and raises ``ValueError`` otherwise; the list never changes its length here.
* Nested indices (an element of a nested ``Set``) descend into the nested row
  lists.
* Any other index type raises ``TypeError``.

``clear(key)`` removes every column at or below the field key. ``Set``
assignment uses it to drop the old columns of the target field before writing
the new ones.

Length and capacity
^^^^^^^^^^^^^^^^^^^

``get_length(key)`` returns the length of the first column at or below the
field key, since all columns of a container share it. With an index, it is the
length of the selected element, and 0 if that element is missing. It returns
``None`` if no column exists yet.

A ``Set`` field with a capacity is padded to that capacity when it is written
as a whole from a list or tuple, and new columns written element by element
are allocated at the capacity. Its length is then the capacity, not the number
of written rows.

Concatenation
^^^^^^^^^^^^^

``_BaseStorage.concat`` (used by ``B.concat``) creates a new engine of the
root's type and fills it leaf by leaf. Each leaf must be initialized in all
inputs or in none of them. Static leaves are joined with ``np.concatenate``.
Dynamic leaves go through ``concat_columns``: if all inputs are NumPy arrays
with the same row shape, ``np.concatenate``; otherwise (rows of different
shapes, or Awkward inputs) ``ak.concatenate``, which accepts a mix of NumPy and
Awkward inputs. Empty inputs are skipped, so an empty selection never forces
the switch to Awkward. The inputs may be views; ``read`` returns only their
rows.

Serialization
^^^^^^^^^^^^^

Containers are not pickled through the engine. ``SchemaConvertible.__reduce_ex__``
exports the container with ``to_arrow``, which reads only the rows of the view
(stacking pending columns on the way), and the receiving side rebuilds it with
``from_arrow`` into a new ``RootEngine``. This is the path for Ray's object
store and for the ``PickleSpillStore`` that the Ray backend writes overflowing
items to.

``leaf_to_arrow`` and ``leaf_from_arrow`` (``core/struct/utils/arrow_utils.py``)
convert single columns:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Column
     - Arrow
     - Restored as
   * - static, NumPy
     - ``FixedShapeTensorArray`` (primitive array for 1-D columns)
     - NumPy
   * - dynamic, NumPy
     - nested ``FixedSizeListArray``, one level per row axis
     - NumPy (``is_regular_arrow`` / ``regular_arrow_to_numpy``)
   * - dynamic, Awkward
     - (nested) ``LargeListArray``, with validity bitmaps for missing rows
     - Awkward (``ak.from_arrow``)
   * - nested ``Set`` level
     - ``ListArray`` of the level below
     - NumPy if all nested Sets have one length, Awkward otherwise

On import, the Arrow type and null count decide, not the schema: null-free
nested fixed-size lists of numbers become NumPy, null-free nested-Set lists are
restored level by level, and everything else goes through ``ak.from_arrow``.
The fixed-size list type matches what ``ak.to_arrow`` produces for a regular
Awkward Array, so batches written by older versions restore as NumPy too, and
columns of different slots of a ``Collection`` still concatenate.

``from_arrow`` adopts Arrow buffers without copying where possible. Such
buffers are read-only, so the first in-place write copies them: a static column
in ``_writable_static_buffer``, a dynamic one in ``split_rows``.

``RootEngine`` also defines ``__getstate__`` and ``__setstate__`` for the rare
case where an engine is pickled directly. They drop and recreate its lock.

Thread safety
^^^^^^^^^^^^^

Element-wise access from several threads (for example ``Op.map_element``) is
safe as long as each thread reads and writes its own rows:

* Writes to different rows of an existing column touch different memory and
  take no lock.
* Transitions that replace a column object take the engine's lock and check
  their condition again inside it (double-checked locking): allocating a
  column, expanding or copying a static buffer, splitting a column into rows,
  and stacking pending rows.
* A transition publishes the new state before it removes the old one (rows
  before deleting the buffer, the buffer before deleting the rows), so a
  concurrent reader always finds the column.

Whole-column writes and ``clear`` are not thread-safe.

The struct tests (``tests/scipion_bridge/struct/``) cover the engine through
the public containers. ``test_root_engine.py`` tests ``RootEngine`` directly
with explicit schema entries, which is the easiest way to reproduce a storage
bug in isolation.
