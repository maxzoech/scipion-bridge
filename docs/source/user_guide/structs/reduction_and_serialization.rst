.. _reduction-semantics:

Reduction and Serialization
===========================

.. currentmodule:: scipion_bridge

Two properties of the array types make them suitable for distributed
processing. They define how data is split into independent batches and how
partial results are combined (*tiling* and *reduction*), and they convert to
and from Apache Arrow, which lets batches move between processes without
copies. This article describes both.

.. contents:: On this page
   :local:
   :depth: 2

Tiling and reduction
--------------------

To distribute work, a scheduler must know how to split data and how to
combine partial results:

* **Tiling.** A ``Set[T]`` can be sliced into subsets that are processed
  independently. Slicing creates views, so tiling a batch costs nothing.
* **Reduction.** Partial results are combined according to their type. A
  ``Set`` is reduced by **concatenation**. A ``Struct`` and a ``Collection``
  are reduced by **last write wins**: a later value replaces an earlier one.

.. code-block:: text

            ┌──► subset ──┐
            ├──► subset ──┤
    Set  ───┼──► subset ──┼───► Set        (concatenate)
            └──► subset ──┘

Reduction rules nest
--------------------

The rules apply field by field, recursively. When a ``Collection[Class2D]``
is written repeatedly, each class's ``particles`` set is **appended** to,
while its ``representative`` (a struct) and ``class_id`` are **replaced**:

.. code-block:: text

    before                         after a new batch
    root.0.particles  [■][■]       root.0.particles  [■][■]
    root.0.rep        [■]          root.0.rep        [■]
    root.1.particles  [■][■][■]    root.1.particles  [■][■][■][■]   (appended)
    root.1.rep        [■]          root.1.rep        [■]            (replaced)

These rules are what makes streaming outputs well defined. A protocol emits
partial results, one per batch, and a sink combines them. Because the
combination follows from the types, a protocol does not have to describe how
its outputs are merged; it only has to choose output types whose rules match
its meaning (see :doc:`/integration/data_model`).

``TensorStoreSinkWriter`` implements these rules for Zarr: set columns are
appended along their first axis, and other arrays are overwritten. The
Scipion backend implements them for Scipion sets: new particles and classes
are appended to the protocol's output sets.

Arrow conversion
----------------

Every container converts to and from an Arrow ``RecordBatch``:

.. code-block:: python

    batch = particles.to_arrow()                    # one row per element
    restored = B.Set[Particle].from_arrow(batch)

Each leaf column becomes one Arrow column. Static columns become fixed-shape
tensor columns, dynamic columns with regular elements become nested
fixed-size lists, and ragged columns become variable-length lists with
validity bitmaps for missing elements. On import, the Arrow layout decides the
representation: regular data is restored as NumPy arrays, ragged data as
Awkward Arrays. ``from_arrow`` adopts the Arrow buffers without copying where
possible; such columns are read-only and are copied on their first in-place
write.

Pickling
--------

Sets, structs and collections are pickled through their Arrow form. Two
consequences are important in practice:

* **Only the data of a view is serialized.** Pickling ``particles[:10]``
  sends ten rows, not the set it was taken from.
* **Buffers travel out of band.** With pickle protocol 5, the Arrow buffers
  are transferred separately from the pickle stream. Ray uses this to place
  batches in its shared-memory object store, from which other processes on
  the same node map them without copying.

The container class, such as ``Set[Particle]``, is pickled *by reference*: the
receiving process imports the element type by module and name and rebuilds
the specialized class. With the standard ``pickle`` module, the element type
must therefore be importable. Classes defined in ``__main__``, in a notebook
or inside a function are serialized by value instead, which needs
``cloudpickle`` (Ray uses it automatically) and is slower.

Why this matters for streaming
------------------------------

Every item that moves between stages of a Ray pipeline is pickled. A list of
structs would be pickled object by object; a set is pickled as a handful of
contiguous buffers. This is why ``Pipeline.send`` rejects Python lists, why
``flatten`` refuses to split a ``Set`` into single elements, and why batches,
not elements, are the unit of streaming.
