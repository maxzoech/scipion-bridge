Collections
===========

.. currentmodule:: scipion_bridge

A ``Collection[T]`` holds a **fixed number** of ``T`` items. Unlike a set,
which stores all elements in shared columns, a collection stores each slot
**row-wise**, with columns of its own. This article explains when that layout
is the right one and how collections are used.

.. contents:: On this page
   :local:
   :depth: 2

Row-wise layout
---------------

Each slot of a collection has its own set of columns:

.. code-block:: text

    root.0.pixels   [   ][   ][   ]
    root.1.pixels   [   ][   ][   ]
    root.2.pixels   [   ][   ][   ]

Compare this with a ``Set[Class2D]``, where the particles of all classes would
share one column per field. In a collection, each slot's nested sets grow
independently: adding particles to class 3 does not touch the columns of
class 7.

When to use a collection
------------------------

Use a collection when items are **addressed by index** and each item
**contains data that grows independently**. The typical example is the set
of classes of a classification:

.. code-block:: python

    class Class2D(B.Struct):
        particles: B.Set[Particle]
        class_id: int
        representative: Particle


    classes = B.Collection[Class2D](size=50)
    classes[3] = Class2D(particles=members, class_id=3, representative=average)

The number of classes is fixed by a parameter, each class is identified by its
index, and each class accumulates its own particles. Because each slot has its
own columns, each class can later be processed, chunked or checkpointed on its
own.

Use a set instead when the items are interchangeable members of a batch that
is processed as a whole.

Working with slots
------------------

Slots start uninitialized. Assigning a struct to an index initializes the
slot; reading an uninitialized slot raises ``KeyError``.

.. code-block:: python

    classes.is_initialized(3)       # True
    classes.initialized_indices()   # [3]

    for cls in classes:             # iterates over initialized slots, in order
        ...

    for index, cls in classes.items():
        ...

    flat = classes.to_set()         # Set[Class2D] of the initialized slots

``len(classes)`` is the number of slots. Iteration yields only the
initialized items, in index order, which is also what ``flatten()`` emits in a
stream: one item per initialized class.

Collections in streams
----------------------

A collection travels through a stream as a single item. Two operations are
commonly applied to it:

* ``flatten()`` emits its initialized items one by one, for example to route
  each class into a ``group_by`` keyed by ``class_id``;
* a sink persists it following the :doc:`reduction rules
  <reduction_and_serialization>`: each slot's nested sets are appended to and
  its other fields replaced.

The Scipion backend converts a ``SetOfClasses2D`` into a
``Collection[Class2D]`` and back, with class ``k`` (1-based in Scipion) in slot
``k - 1``.
