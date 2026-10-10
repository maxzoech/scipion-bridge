Sets
====

.. currentmodule:: scipion_bridge

A ``Set[T]`` is a batch of ``T`` structs stored column by column. It is the
item type of almost every stream: inputs arrive as sets, operations such as
``chunk`` re-batch sets, and outputs are emitted as sets. This article
describes how sets are created, the two ways of accessing their data, how
selections work, and how sets are combined.

.. contents:: On this page
   :local:
   :depth: 2

Column-wise layout
------------------

Each leaf field of ``T`` becomes one contiguous column. The first axis of a
column is the element index:

.. code-block:: text

    root.pixels            [ img 0 ][ img 1 ][ img 2 ] ...
    root.ctf.defocus_u     [ 0 ][ 1 ][ 2 ] ...
    root.ctf.defocus_v     [ 0 ][ 1 ][ 2 ] ...
                           ─────────── physical memory ──────────►

Because consecutive values of one field are adjacent in memory, vectorized
operations on a field use CPU caches well, and handing a column to NumPy,
PyTorch or JAX does not require gathering values from objects.

Creating sets
-------------

.. code-block:: python

    # From elements
    particles = B.Set[Particle]([Particle(...), Particle(...)])

    # Empty; columns grow as they are written
    particles = B.Set[Particle]()

    # Preallocated for a known number of elements
    particles = B.Set[Particle](capacity=1_000)

The element type must be a ``Struct``. Building a set from a list of structs
copies each struct into the columns, which is convenient for small examples
and tests. Code that produces many elements, such as a resolver reading a
file, should write whole columns into a preallocated set instead:

.. code-block:: python

    particles = B.Set[Particle](capacity=len(images))
    particles["pixels"] = images                          # (N, H, W)
    particles["sampling_rate"] = np.full((len(images), 1), 1.35)

With a capacity, the length of the set is the capacity, and columns written
from lists or tuples are padded to it.

The object API
--------------

Indexing a set with an integer returns a struct that is a *view* of that row.
Reading and writing attributes of the view reads and writes the set's
columns; assigning a struct to an index copies its fields into that row:

.. code-block:: python

    particles[0] = Particle(...)        # writes row 0
    element = particles[0]              # Particle, a view of row 0
    element.ctf.defocus_u               # a float
    element.sampling_rate = 1.0         # writes through to the set

The object API is the natural choice for per-element logic, for example in a
function passed to ``map_element``, which receives one row view at a time.

Vectorized access
-----------------

Indexing with a **field name** returns the whole column. Array fields return
an array that NumPy, PyTorch, JAX and other libraries consume through the
Python array API; struct fields return a ``Set`` of the nested struct, which
can be indexed further:

.. code-block:: python

    images = particles["pixels"]               # (N, H, W)
    ctfs = particles["ctf"]                    # Set[CTF]
    defocus = ctfs["defocus_u"]                # np.ndarray, shape (N, 1)

    particles["sampling_rate"] = np.full((len(particles), 1), 1.0)

Assigning to a field name replaces the column. For nested struct fields,
assign a ``Set`` of the nested type.

.. note::

   Scalar fields have shape ``(1,)`` per element, so their columns have shape
   ``(N, 1)``. Use ``.ravel()`` to obtain a flat vector, for example for a
   boolean mask.

   A dynamic field (one with a ``None`` dimension) is returned as a NumPy
   array while all its elements share a shape, and as an Awkward Array once
   they differ or some are missing. See :doc:`dynamic_data`.

Selections and views
--------------------

Sets support slices (with step 1), integer index arrays and boolean masks.
Each returns a ``Set`` **view** of the selected rows, without copying data:

.. code-block:: python

    first = particles[:100]
    picked = particles[[0, 5, 9]]
    good = particles[particles["ctf"]["resolution"].ravel() < 4.0]

Slices with another step (``particles[::2]``) are not supported; use an index
array such as ``np.arange(0, len(particles), 2)`` instead.

Internally, every view holds a *path* into the storage of the set it was
taken from. Attribute access and selection narrow the path, and paths compose,
so selecting from a selection works as expected:

.. code-block:: text

    root[:].ctf[:]
        │  particles[2:10]
        ▼
    root[2:10].ctf[:]
        │  particles[2:10][1]
        ▼
    root[3].ctf[:]

Because views share storage with their parent, writes through a view change
the parent:

.. code-block:: python

    view = particles[1:4]
    view[0] = Particle(...)              # also changes particles[1]
    view["sampling_rate"] = np.ones((3, 1))

Assignment works with the same keys: ``particles[mask] = other`` writes the
rows of ``other`` into the selected rows.

Empty selections
^^^^^^^^^^^^^^^^

A selection that matches no row is a valid, empty set. Its columns are empty
arrays, and it can be stored in another struct like any other set:

.. code-block:: python

    members = particles[assignments == k]       # may be empty
    cls = Class2D(class_id=k, particles=members)
    len(cls.particles)                          # 0 for an empty class

An empty selection is different from a *missing* element; see
:doc:`dynamic_data`.

Concatenation
-------------

``B.concat`` joins sets with identical schemas along the first axis:

.. code-block:: python

    both = B.concat([particles[:100], particles[500:]])

The result is a new set with storage of its own. Static columns are joined
with ``np.concatenate``; dynamic columns stay NumPy arrays if all inputs have
elements of the same shape, and become Awkward Arrays otherwise. Empty inputs
are skipped. Each field must be initialized in all inputs or in none of them.

``chunk`` and ``collect`` in the streaming API use ``concat`` and slicing to
re-batch sets, so a stream of sets of arbitrary sizes can be turned into
batches of exactly the size a model expects.

Length and capacity
-------------------

``len(s)`` is the number of elements. For a set created with a capacity, it is
the capacity, whether or not every row has been written. The ``capacity``
property returns the capacity, or ``None`` for a set that grows with its
columns.
