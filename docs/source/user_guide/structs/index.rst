Array Types
===========

.. currentmodule:: scipion_bridge

The array type system describes the data that flows through a protocol. It
serves three purposes at once: it gives Python code a typed, object-like view
of cryo-EM data; it stores that data in columnar buffers that are cheap to
slice, combine and serialize; and it defines how data is split into batches
and how partial results are combined, which is what allows a backend to
distribute work automatically.

.. contents:: On this page
   :local:
   :depth: 2

Three containers
----------------

The system has three container types. Each one is a typed *view* over storage
that lives in NumPy arrays, Awkward Arrays and Apache Arrow buffers.

.. list-table::
   :header-rows: 1
   :widths: 18 32 20 30

   * -
     - Purpose
     - Storage layout
     - Reduction
   * - ``Struct``
     - Describes a single element
     - One value per field
     - Last write wins
   * - ``Set[T]``
     - A batch of ``T`` elements
     - Column-wise
     - Concatenation
   * - ``Collection[T]``
     - A fixed number of ``T`` items
     - Row-wise
     - Last write wins

A ``Struct`` subclass is a schema: a tree of named fields, much like a
``dataclass``. ``Set[T]`` stores many structs of type ``T``, with each leaf
field of ``T`` as one contiguous column. ``Collection[T]`` has a fixed number
of slots, each with columns of its own.

.. code-block:: python

    import numpy as np
    import scipion_bridge as B


    class Coordinate(B.Struct):
        x: float
        y: float


    class Particle(B.Struct):
        pixels: B.Array[np.float32] = B.Array(shape=(None, None))
        coordinate: Coordinate
        sampling_rate: float


    p = Particle(pixels=np.zeros((64, 64), np.float32), sampling_rate=1.35)
    batch = B.Set[Particle]([p, p, p])

    batch[0].sampling_rate          # 1.35       one element, as an object
    batch["sampling_rate"]          # (3, 1)     one field, as a column
    batch[1:3]                      # a view of two elements, no copy

Why columns
-----------

Cryo-EM code mostly operates on one field of many elements at a time: all
images of a batch go to a GPU, all defocus values into a histogram, all
coordinates into a filter. A column-wise layout keeps the values of one field
next to each other in memory, so such operations are a single array access
instead of a loop over objects. The same layout makes the operations that a
streaming system performs constantly cheap:

* **slicing** a batch selects a range of every column, which creates views
  rather than copies;
* **concatenating** batches concatenates columns;
* **serializing** a batch hands the column buffers to Apache Arrow, which Ray
  transfers through shared memory without copying.

The object API (``batch[0].sampling_rate``) remains available for code that
thinks in elements; it reads and writes through to the columns.

Schemas describe physical memory
--------------------------------

A struct describes the physical layout of its data, which ``print_schema``
shows:

.. code-block:: text

    >>> Particle.print_schema()
    /
    ├── pixels: Array[float32], shape: [None, None]
    ├── coordinate (struct)
    │   ├── x: Array[float64], shape: [1]
    │   └── y: Array[float64], shape: [1]
    └── sampling_rate: Array[float64], shape: [1]

Every scalar is an array of shape ``(1,)``, and every leaf of the tree becomes
one column of a set. Fields with a ``None`` dimension are *dynamic*: their
elements may differ in size. A schema without dynamic fields is marked
``(static size)``; every element of it has the same memory footprint.

Articles in this section
------------------------

.. toctree::
   :maxdepth: 1

   defining_structs
   sets
   collections
   dynamic_data
   reduction_and_serialization
   single_particle

* :doc:`defining_structs` covers field types, named dimensions, creating
  and reading structs, uninitialized fields, and inheritance.
* :doc:`sets` covers creating sets, the object and vectorized APIs, views,
  selections and concatenation.
* :doc:`collections` covers fixed-size, row-wise containers such as the
  classes of a classification.
* :doc:`dynamic_data` explains how fields with variable sizes are stored,
  when they become Awkward Arrays, and how missing elements differ from empty
  selections.
* :doc:`reduction_and_serialization` describes how batches are split and
  combined, and how containers are converted to Arrow and pickled.
* :doc:`single_particle` lists the ready-made types for single particle
  analysis.

The storage engine behind these containers is described for contributors in
:doc:`/developer/struct_storage`.
