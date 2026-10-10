Defining Structs
================

.. currentmodule:: scipion_bridge

A ``Struct`` subclass describes the layout of one element. This article
covers the field types a struct can declare, how structs are created and read,
how uninitialized fields behave, and how structs are extended through
inheritance.

.. contents:: On this page
   :local:
   :depth: 2

Declaring fields
----------------

Each annotated class attribute of a ``Struct`` subclass becomes a field:

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

The order of the annotations is the order of the fields. The struct has no
``__init__`` to write; it accepts its fields as keyword arguments.

Field types
-----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Declaration
     - Meaning
   * - ``x: float`` (also ``int``, ``bool``, NumPy scalar types)
     - A scalar, stored as an array of shape ``(1,)``. Reading it from a
       struct returns a Python scalar.
   * - ``pixels: B.Array[np.float32] = B.Array(shape=(None, None))``
     - An n-dimensional array with a **fixed rank**. Each dimension is a fixed
       size or ``None`` (any size).
   * - ``ctf: CTF``
     - A nested struct.
   * - ``particles: B.Set[Particle]``
     - A nested set: a variable number of elements per struct.
   * - ``classes: B.Collection[Class2D] = B.Collection[Class2D](size=10)``
     - A nested collection with a fixed number of slots.

Scalars
^^^^^^^

Scalars are the most common field type: identifiers, scores, physical
parameters. Each is stored as an array of shape ``(1,)``, so that every field,
scalar or not, is an array, and a column of scalars in a set has shape
``(N, 1)``. The dtype follows the annotation: ``float`` is ``float64``,
``int`` is ``int64``, and NumPy scalar types such as ``np.float32`` are used as
given.

Arrays
^^^^^^

``B.Array[dtype]`` declares an array field. The ``shape`` fixes the rank, and
each dimension is either a number or ``None``:

.. code-block:: python

    class Pose(B.Struct):
        rotation: B.Array[np.float64] = B.Array(shape=(3, 3))     # static
        shifts: B.Array[np.float64] = B.Array(shape=(2,))         # static


    class Micrograph(B.Struct):
        pixels: B.Array[np.float32] = B.Array(shape=(None, None))  # dynamic

A field whose dimensions are all numbers is *static*: every element has the
same shape, and a column of it is always one NumPy array. A field with a
``None`` dimension is *dynamic*: elements may differ in that dimension. How
dynamic columns are stored is the subject of :doc:`dynamic_data`. Prefer
static shapes whenever the size is known.

Nested structs
^^^^^^^^^^^^^^

A field annotated with another struct nests it. Nesting groups related fields
without adding storage: each leaf is still one column, and the nested struct
only adds a level to the column's name:

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

Nested sets and collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A field annotated with ``B.Set[T]`` gives each element a variable number of
members, such as the particles of a 2D class. A ``B.Collection[T]`` field with
a ``size`` gives each element a fixed number of slots. Both are described in
:doc:`sets` and :doc:`collections`; as fields, they behave the same way as on
their own.

Named dimensions
----------------

``B.Dim`` (an alias of ``B.Arg``) gives a dimension a name, so that several
arrays can share it:

.. code-block:: python

    class Box(B.Struct):
        N = B.Dim(3)
        corners = B.Array[np.float64](shape=(N, 2))
        weights = B.Array[np.float64](shape=(N,))

``B.Dim()`` without a value is a dynamic dimension. On an instance, a
dimension reads as its integer value (``Box().N == 3``). Dimensions are
class-level schema parameters and cannot be assigned on instances.

Creating and reading structs
----------------------------

Pass field values as keyword arguments. Attribute access reads and writes the
underlying storage:

.. code-block:: python

    p = Particle(
        pixels=np.zeros((64, 64), np.float32),
        coordinate=Coordinate(x=1.0, y=2.0),
        sampling_rate=1.35,
    )

    p.coordinate.x            # 1.0
    p.sampling_rate = 1.0     # writes to storage
    p.pixels.shape            # (64, 64)

Values are checked against the field's dtype when they are written. A value
must cast to the dtype with ``same_kind`` casting, so writing ``1.5`` to an
``int`` field raises ``TypeError``, while writing ``1`` to a ``float`` field
works.

A nested struct is a view into the same storage as its parent: writing
``p.coordinate.x = 3.0`` changes ``p``.

Uninitialized fields
--------------------

Fields do not have to be set. A field that was never written is
*uninitialized*, and reading it raises ``UninitializedFieldError``.
``is_initialized(name)`` and ``initialized_fields()`` check first:

.. code-block:: python

    >>> c = Coordinate(x=1.0)
    >>> c.initialized_fields()
    ['x']
    >>> c.is_initialized("y")
    False

A nested struct counts as initialized as soon as one of its fields is. This
lets resolvers fill only the fields their source provides, for example only
the CTF parameters present in a STAR file, and lets consumers check what is
available instead of guessing.

Inheritance
-----------

A subclass of a struct adds fields to those of its base:

.. code-block:: python

    class FlexParticle(Particle):
        embeddings: B.Array[float] = B.Array(shape=(None,))

The subclass has all fields of ``Particle`` followed by ``embeddings``.
Inheritance is the way to extend an existing type with what a method adds.
Note that containers are specialized per element type: ``Set[FlexParticle]``
is not a subclass of ``Set[Particle]`` (see
:doc:`/integration/data_model`).

Rules for class attributes
--------------------------

Every class-level attribute of a struct must be annotated, or be a schema
object (``B.Array``, ``B.Set``, ``B.Collection``, ``B.Dim``). A plain
assignment such as ``x = 1.0`` raises ``TypeError`` when the class is defined,
because it would be ambiguous whether it is a field or a constant.

Define structs at module level in an importable module. Containers of a
struct are pickled by a reference to its module and name, which does not work
for classes defined in ``__main__``, in a notebook or inside a function (see
:doc:`reduction_and_serialization`).
