.. _ragged-arrays:

Dynamic Data and Ragged Arrays
==============================

.. currentmodule:: scipion_bridge

A field with a ``None`` dimension, such as
``pixels = B.Array(shape=(None, None))``, *allows* elements of different
shapes. This article explains how the columns of such fields are stored, when
they switch from NumPy to Awkward Array, how to write code that handles both,
and how missing elements differ from empty selections.

.. contents:: On this page
   :local:
   :depth: 2

The storage depends on the data
-------------------------------

The schema only states what is allowed. How a dynamic column is stored
depends on the data actually in it:

* While all elements have the same shape, the column is a NumPy array of shape
  ``(N, *element_shape)``, exactly like the column of a static field. Slicing,
  masking and serializing it cost what they cost in NumPy.
* Once elements differ in shape, or some are missing, the column becomes an
  `Awkward Array <https://awkward-array.org>`_. Awkward stores the elements
  in one flat buffer plus offsets, so elements of different shapes sit next to
  each other without padding.

.. code-block:: text

    all 64 × 64        NumPy    (3, 64, 64)              [ img 0 ][ img 1 ][ img 2 ]
    64 × 64, 32 × 32   Awkward  3 * var * var * float32  [ flat pixel buffer    ]
                                                         [ offsets per element  ]

The reason for this rule is speed. Selecting rows of an Awkward Array goes
through generic indexing that, for large image stacks, is much slower than
NumPy's, and converting Awkward Arrays to NumPy may copy them. In practice,
the images of one dataset share a box size, so particle columns stay NumPy
arrays, and the Awkward representation is only used when the data really is
ragged.

When a column becomes Awkward
-----------------------------

The switch happens only when the data requires it:

* an element of a different shape is written into the column;
* sets whose elements differ in shape are concatenated;
* elements are left unwritten (see `Missing elements`_ below).

The switch is one-way: a column does not turn back into NumPy when it is
read, because checking a large Awkward Array for regularity on every read
would cost what the rule is meant to save. A column that is *built* from
regular parts, by concatenation, by stacking rows, or when restored from
Arrow, is built as NumPy directly.

Writing code for both representations
-------------------------------------

Code that needs a NumPy array should convert explicitly. Both
``np.asarray(column)`` and ``ak.to_numpy(column)`` accept either type, and
the conversion raises if the elements actually differ in shape:

.. code-block:: python

    import awkward as ak

    pixels = ak.to_numpy(particles["pixels"])     # (N, H, W), or raises

When ragged data is expected, process it per element, for example in a
``map_element`` function, or pad it to a common shape in the stage that
needs one.

Element writes
--------------

Writing a single element of a dynamic column, as in
``particles[1].pixels = img``, cannot be done in place if the new element has
a different shape. The column is therefore split into one buffer per row, the
row is replaced, and the rows are joined into one column again the next time
the whole column is read (as NumPy, if all rows share a shape).

This intermediate state is invisible to user code, but it makes a loop of
element writes cheap: the column is rebuilt once, not once per write. That is
what ``map_element`` relies on when many threads write the rows of one batch.
Still, when all values are at hand, assigning the whole column at once is the
fastest option.

Missing elements
----------------

An element that was never written is *missing*. This happens when only some
rows of a field are written:

.. code-block:: python

    particles = B.Set[Particle]()
    particles["sampling_rate"] = np.ones((3, 1))   # the set has 3 rows
    particles[2].pixels = img                      # rows 0 and 1 stay missing

    particles["pixels"]       # an Awkward Array: [None, None, img]
    particles[0].pixels       # raises UninitializedFieldError

Only dynamic fields can tell a missing element apart. A static column is a
NumPy array that is allocated with zeros, so a missing element of a static
field reads as zeros.

Missing is not empty
^^^^^^^^^^^^^^^^^^^^

A missing element is the absence of data. An empty selection is data: a set
with zero rows. Reading an empty selection returns empty columns and never
raises; it can be assigned, concatenated and serialized like any other set.
This distinction matters for nested sets: a class whose particle set is empty
is a valid class with no members, while a class whose particle set was never
written is incomplete.

Nested sets of different lengths
--------------------------------

The same rule applies one level up. In a ``Set[Class2D]``, the column
``particles.pixels`` has one entry per class, each holding the images of that
class's particles. If every class has the same number of particles, the
column is a NumPy array with one axis for the classes and one for their
particles. If the classes have different numbers of particles, the nested
dimension is ragged, and the column is an Awkward Array. For this reason,
classes that accumulate particles independently are usually kept in a
``Collection`` (see :doc:`collections`).
