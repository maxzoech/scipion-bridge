Array Types: Struct, Set and Collection
=======================================

.. currentmodule:: scipion_bridge

The array type system describes the data flowing through a protocol. It has
three container types:

.. list-table::
   :header-rows: 1
   :widths: 15 35 20 30

   * -
     - Purpose
     - Storage layout
     - Reduction
   * - ``Struct``
     - Describes a single element
     - n/a
     - Last write wins
   * - ``Set[T]``
     - A batch of ``T`` elements
     - Column-wise
     - Concatenation
   * - ``Collection[T]``
     - A fixed number of ``T`` items
     - Row-wise
     - Last write wins

The data itself lives in NumPy arrays, Awkward arrays (for ragged data), and
Apache Arrow buffers. The containers are thin, typed *views* over this storage.

.. contents:: On this page
   :local:
   :depth: 2

Structs
-------

A ``Struct`` subclass describes the layout of one element. Each annotated class
attribute becomes a field:

.. code-block:: python

    import numpy as np
    import scipion_bridge as B


    class CTF(B.Struct):
        defocus_u: float
        defocus_v: float
        defocus_angle: float
        phase_shift: float
        resolution: float
        fit_quality: float

A struct describes **physical memory**. Every scalar field is stored as an array
of shape ``(1,)``:

.. code-block:: text

    >>> CTF.print_schema()
    / (static size)
    ├── defocus_u: Array[float64], shape: [1]
    ├── defocus_v: Array[float64], shape: [1]
    ├── defocus_angle: Array[float64], shape: [1]
    ├── phase_shift: Array[float64], shape: [1]
    ├── resolution: Array[float64], shape: [1]
    └── fit_quality: Array[float64], shape: [1]

Field types
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Declaration
     - Meaning
   * - ``x: float`` (also ``int``, ``bool``, NumPy scalar types)
     - A scalar, stored as an array of shape ``(1,)``. Reading it returns a
       Python scalar.
   * - ``pixels: B.Array[np.float32] = B.Array(shape=(None, None))``
     - An n-dimensional array. The **rank is fixed**. Each dimension is either a
       fixed size or ``None`` (any size).
   * - ``ctf: CTF``
     - A nested struct.
   * - ``particles: B.Set[Particle]``
     - A nested set: a variable number of elements per struct.
   * - ``classes: B.Collection[Class2D] = B.Collection[Class2D](size=10)``
     - A nested collection of fixed size.

Nested structs compose into a tree of columns:

.. code-block:: python

    class Coordinate(B.Struct):
        x: float
        y: float


    class Particle(B.Struct):
        pixels: B.Array[np.float32] = B.Array(shape=(None, None))
        ctf: CTF
        coordinate: Coordinate
        sampling_rate: float

.. code-block:: text

    >>> Particle.print_schema()
    /
    ├── pixels: Array[float32], shape: [None, None]
    ├── ctf (struct)
    │   ├── defocus_u: Array[float64], shape: [1]
    │   └── ...
    ├── coordinate (struct)
    │   ├── x: Array[float64], shape: [1]
    │   └── y: Array[float64], shape: [1]
    └── sampling_rate: Array[float64], shape: [1]

A schema whose shapes are all fixed is marked ``(static size)``. Every element
of a static schema has the same memory footprint.

Named dimensions
^^^^^^^^^^^^^^^^

``B.Dim`` (an alias of ``B.Arg``) gives a dimension a name, so several arrays
can share it. ``B.Dim()`` without a value is a dynamic dimension:

.. code-block:: python

    class Box(B.Struct):
        N = B.Dim(3)
        corners = B.Array[np.float64](shape=(N, 2))

On an instance, the dimension reads as its integer value (``Box().N == 3``).
Dimensions are class-level schema parameters and cannot be assigned on
instances.

Creating and reading structs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass field values as keyword arguments. Attribute access reads and writes the
underlying storage:

.. code-block:: python

    p = Particle(
        pixels=np.zeros((64, 64), np.float32),
        coordinate=Coordinate(x=1.0, y=2.0),
        sampling_rate=1.35,
    )
    p.coordinate.x          # 1.0
    p.sampling_rate = 1.0

Fields may stay uninitialized. Reading one raises ``UninitializedFieldError``.
``is_initialized(name)`` and ``initialized_fields()`` let you check first:

.. code-block:: python

    >>> Coordinate(x=1.0).initialized_fields()
    ['x']

Inheritance
^^^^^^^^^^^

Structs support inheritance. A subclass adds fields to those of its base:

.. code-block:: python

    class FlexParticle(Particle):
        embeddings: B.Array[float] = B.Array(shape=(None,))

Every class-level attribute of a struct must be annotated or be a schema
object (``B.Array``, ``B.Set``, ``B.Dim``, ...). A plain ``x = 1.0`` raises a
``TypeError`` at class definition time.

Sets
----

A ``Set[T]`` stores a batch of ``T`` structs in **column-wise** format. Each
leaf field of ``T`` becomes one contiguous column:

.. code-block:: text

    root.pixels            [ img 0 ][ img 1 ][ img 2 ] ...
    root.ctf.defocus_u     [ 0 ][ 1 ][ 2 ] ...
    root.ctf.defocus_v     [ 0 ][ 1 ][ 2 ] ...
                           ─────────── physical memory ──────────►

This layout makes slicing, concatenation, and vectorized access cheap, and
it uses CPU caches well because consecutive values of a field sit next to each
other in memory.

Creating sets
^^^^^^^^^^^^^

.. code-block:: python

    # From elements
    particles = B.Set[Particle]([Particle(...), Particle(...)])

    # Empty, growing dynamically
    particles = B.Set[Particle]()

    # Preallocated
    particles = B.Set[Particle](capacity=1_000)

The element type must be a ``Struct``.

The object API
^^^^^^^^^^^^^^

Indexing with an integer returns a struct view of that row. Assigning a struct
writes a row:

.. code-block:: python

    particles[0] = Particle(...)
    element = particles[0]             # Particle (a view, not a copy)
    element.ctf.defocus_u              # float
    element.sampling_rate = 1.0        # writes through to the set

Vectorized access
^^^^^^^^^^^^^^^^^

Indexing with a **field name** returns the whole column. Array fields return an
array that NumPy, PyTorch, JAX, and others can consume through the Python array
API. Struct fields return a ``Set`` of the nested struct:

.. code-block:: python

    imgs = particles["pixels"]          # all images
    ctfs = particles["ctf"]             # Set[CTF]
    defocus = ctfs["defocus_u"]         # np.ndarray, shape (N, 1)

    particles["sampling_rate"] = np.full((len(particles), 1), 1.0)

.. note::

   Scalar fields have shape ``(1,)`` per element, so their columns have shape
   ``(N, 1)``. Use ``.ravel()`` to get a flat vector.

   Fields with dynamic (``None``) dimensions are returned as `Awkward Arrays
   <https://awkward-array.org>`_. Convert them with ``ak.to_numpy`` when all
   elements have the same shape.

Slicing and lensing
^^^^^^^^^^^^^^^^^^^

Sets support slices (step 1 only), integer index arrays, and boolean masks.
All of them return a ``Set`` **view** of the selected rows, without copying:

.. code-block:: python

    subset = particles[:100]
    picked = particles[[0, 5, 9]]
    good = particles[particles["ctf"]["resolution"].ravel() < 4.0]

Internally, every view holds a *path* into the root storage. Attribute access
and slicing **narrow** that path, and the storage uses the path to fetch the
right part of each column. Paths compose, so slicing a slice works as expected:

.. code-block:: text

    root[:].ctf[:]
        │  particles[2:10]
        ▼
    root[2:10].ctf[:]
        │  particles[1]
        ▼
    root[3].ctf[:]

Because views share storage with their parent, writes through a view change the
parent:

.. code-block:: python

    view = particles[1:4]
    view[0] = Particle(...)      # also changes particles[1]

Concatenation
^^^^^^^^^^^^^

``B.concat`` joins sets with identical schemas along the first axis:

.. code-block:: python

    both = B.concat([particles[:100], particles[500:]])

Ragged arrays
^^^^^^^^^^^^^

When an array field declares ``None`` dimensions, elements may have different
shapes. The storage starts as a regular dense array. When a sample with a
different shape is written, the storage is **promoted** to a ragged
representation: a flat buffer plus offsets and shapes, implemented with
Awkward Array.

.. code-block:: text

    root.pixels   [   pixel array   ]  ── add irregular sample ──►  [ flat pixel buffer ]
                                                                     [ offsets          ]
                                                                     [ shapes           ]

Collections
-----------

A ``Collection[T]`` holds a **fixed number** of ``T`` items, stored **row-wise**:
each slot has its own set of columns.

.. code-block:: text

    root.0.pixels   [   ][   ][   ]
    root.1.pixels   [   ][   ][   ]
    root.2.pixels   [   ][   ][   ]

Use a collection when items are addressed by index and each item contains
sets that grow independently. A typical example is a set of 2D classes:

.. code-block:: python

    class Class2D(B.Struct):
        particles: B.Set[Particle]
        class_id: int
        representative: Particle


    classes = B.Collection[Class2D](size=50)
    classes[3] = Class2D(particles=members, class_id=3, representative=average)

    classes.initialized_indices()   # [3]
    for cls in classes:             # iterates over initialized slots
        ...
    flat = classes.to_set()         # Set[Class2D] of the initialized slots

Slots start uninitialized. Reading an uninitialized slot raises ``KeyError``.
Collections iterate over their initialized items in index order, so
``flatten()`` in a stream emits one item per class.

Because classes are stored row-wise, each class can later be chunked or
checkpointed on its own.

.. _reduction-semantics:

Reduction semantics
-------------------

To distribute work, a scheduler must know how to split data (*tiling*) and how
to combine partial results again (*reduction*):

* **Tiling:** a ``Set[T]`` can be sliced and processed in independent batches.
* **Reduction:** a ``Set`` is reduced by **concatenation**. ``Struct`` and
  ``Collection`` values are reduced by **last write wins**.

.. code-block:: text

            ┌──► subset ──┐
            ├──► subset ──┤
    Set  ───┼──► subset ──┼───► Set        (concatenate)
            └──► subset ──┘

The rules nest. When a ``Collection[Class2D]`` is written in batches, new
particles are **appended** to each class's ``Set[Particle]``, while its
``representative`` (a struct) is **replaced**:

.. code-block:: text

    before                         after a new batch
    root.0.particles  [■][■]       root.0.particles  [■][■]
    root.0.rep        [■]          root.0.rep        [■]
    root.1.particles  [■][■][■]    root.1.particles  [■][■][■][■]   (appended)
    root.1.rep        [■]          root.1.rep        [■]            (replaced)

Sinks such as ``TensorStoreSinkWriter`` implement exactly these rules: set
columns are appended along the first axis, and plain array fields are
overwritten.

Serialization
-------------

Every container converts to and from an Arrow ``RecordBatch``:

.. code-block:: python

    batch = particles.to_arrow()                 # one row per element
    restored = B.Set[Particle].from_arrow(batch)

Sets, structs, and collections are pickled through this Arrow form. Only the
data of a view is serialized, never the parent storage it belongs to, and with
pickle protocol 5 the Arrow buffers are transferred out of band. This is how
Ray moves batches between workers without copies.

.. note::

   The parametrized class (for example ``Set[Particle]``) is pickled *by
   reference*. With the standard ``pickle`` module, the element type must be
   importable from a module. Classes defined in ``__main__`` or in a notebook
   need ``cloudpickle``, which Ray uses automatically.

Ready-made types
----------------

``scipion_bridge.single_particle`` defines the types used in single particle
analysis:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Struct
     - Fields
   * - ``Particle``
     - ``pixels`` (2D, dynamic), ``ctf``, ``coordinate``, ``sampling_rate``,
       ``acquisition``
   * - ``FlexParticle``
     - ``Particle`` plus ``embeddings`` (1D, dynamic)
   * - ``CTF``
     - ``defocus_u``, ``defocus_v``, ``defocus_angle``, ``phase_shift``,
       ``resolution``, ``fit_quality``
   * - ``Coordinate``
     - ``x``, ``y``
   * - ``Acquisition``
     - ``magnification``, ``voltage``, ``spherical_aberration``,
       ``amplitude_contrast``, ``dose_initial``, ``dose_per_frame``
   * - ``Class2D``
     - ``particles: Set[Particle]``, ``class_id``, ``representative: Particle``
