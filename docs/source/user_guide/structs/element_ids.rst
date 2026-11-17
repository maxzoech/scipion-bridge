Element Ids
===========

.. currentmodule:: scipion_bridge

Every element of a ``Set``, and every ``Struct``, can carry an *element id*:
an identity that lineage uses to trace an output back to the input elements
it holds, and that a Scipion backend can store as the item id of an output.
Ids are not fields: they are not part of the schema, and code that does not
look for them never sees them.

.. contents:: On this page
   :local:
   :depth: 2

Where ids come from
-------------------

An id is a *namespace* and a *position* in it:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Elements
     - Namespace
     - Position
   * - Read from an input stack (MRC, SPIDER)
     - the file, as it is: its path, size and modification time
     - the row in the stack
   * - Read from a Scipion set
     - the table of the set (its database, and for the classes of a
       ``SetOfClasses2D`` the table of the class)
     - its item id
   * - Created by a stage of a pipeline
     - the item the stage emits
     - the row in that item

The resolvers set them: particles read from a ``.star`` or ``.xmd`` file are
the rows of their stack, whichever metadata file selected them, and disabled
rows of an Xmipp file leave gaps. Particles of temporary stacks, which an
external program wrote during the run, are rows a stage created.

Ids are stored compactly, like references to the rows of input files: a batch
of 256 particles read from one stack is a single range, not 256 values.

Ids follow the rows
-------------------

Ids belong to the rows of a Set, not to its values:

* **Views** keep them: ``particles[2:5]``, ``particles[mask]``,
  ``particles[[3, 0]]`` and ``particles[i]`` hold the ids of the rows they
  select.
* **Writing values** keeps them: ``particles["pixels"] = filtered`` changes
  the pixels of the same particles.
* **Copying rows** keeps them: ``concat``, ``astype``, ``Set([...])`` built
  from elements, writing an element into a row (``s[i] = p``), a Set into a
  Struct field (``Class2D(particles=particles[mask])``) and a Struct into a
  ``Collection`` slot.
* **New rows** have no id until the stage that created them emits them; they
  then get the id of the emitted item and their row in it. Every item a stage
  emits has an id of its own, derived from the items it was made from, so the
  ids of new rows are the same in every run of a pipeline on the same inputs
  (Ray backend).

Rows rebuilt from arrays are new rows: ``out["pixels"] =
particles["pixels"][order]`` into a new Set loses the ids of the particles.
Select with views (``particles[order]``) or change the type with ``astype`` to
keep them.

Nested Sets keep their ids in Struct fields and in ``Collection`` slots. A Set
whose elements hold nested Sets of elements with ids, such as a
``Set[Class2D]`` of particles a pipeline read, raises ``NotImplementedError``;
use a ``Collection[Class2D]`` (see :doc:`/user_guide/known_issues`).

New elements from old ones: ``tag``
-----------------------------------

A stage can make several elements out of one, e.g. random views of every
particle for self-supervised training. Views of the same particle would share
its id, but they are different elements. ``tag`` gives them ids of their own,
derived from the id of the element they come from, so lineage still links a
view to its particle:

.. code-block:: python

    n = len(particles)
    views = particles[np.repeat(np.arange(n), 6)]           # 6 views per particle
    views = views.tag("view", np.tile(np.arange(6), n))     # ids of their own

``tag(name, discriminator)`` takes one discriminator per element, or one for
all; ``Struct.tag`` takes one. The values are shared, as by ``astype``.
Integers (the index of a view) derive the ids from the value: running the
stage again gives the same ids. Other values (a seed, the parameters of an
augmentation, bytes) are hashed into the ids: other values give other
elements. Elements without ids keep none, since they are new elements
already.

Duplicate ids, e.g. views that were not tagged, will be reported where
outputs are written, with the stage that created them (planned, with
checkpointing).

Randomness and resume
---------------------

Derive randomness from ids, not ids from randomness. ``B.ids_rng(views,
salt)`` returns one NumPy random generator per element, seeded from its id and
``salt``:

.. code-block:: python

    def augmented(particles):
        n = len(particles)
        views = particles[np.repeat(np.arange(n), 6)]
        views = views.tag("view", np.tile(np.arange(6), n))
        for view, rng in zip(views, B.ids_rng(views, salt="augment")):
            view.pixels = random_crop(view.pixels, rng)
        return views

When the stage runs again, e.g. when a run resumes after a crash (planned,
with checkpointing), every view draws the same values: the index tag and the content of a view agree, and a
resumed run produces exactly the outputs an uninterrupted one would.

Tagging with the random values themselves (``views.tag("seed", seeds)``) also
gives every view an id, for code that cannot seed from ids. Views computed
again then draw new values and get new ids: after a resume, they are added to
the views written before the crash instead of replacing them.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tag by
     - The id means
     - After a resume
   * - The index of the view
     - view *k* of particle *p*
     - the same ids; with ``ids_rng`` the same values
   * - A random value
     - this particular sample
     - new values, new ids: views are added
