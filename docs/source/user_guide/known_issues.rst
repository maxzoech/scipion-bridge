Known Issues
============

This page lists bugs and gaps of the current version that you may run into,
with what to do instead. Behaviours that are intended but often surprise
users are collected in :doc:`sharp_bits`.

Every bug below has a test marked ``xfail(strict=True)`` that reproduces it.
When the bug is fixed, the test starts to pass, which fails the suite until
the marker and the entry here are removed.

.. contents::
   :local:
   :depth: 1

Sets of structs with nested Sets of different sizes cannot be pickled
---------------------------------------------------------------------

A ``Set`` whose elements hold nested Sets of different lengths cannot be
exported to Arrow, and therefore cannot be pickled or sent between the
stages of a Ray pipeline:

.. code-block:: python

    class Member(B.Struct):
        value: float

    class Group(B.Struct):
        members: B.Set[Member]
        group_id: int

    groups = B.Set[Group]([
        Group(members=B.Set[Member]([Member(value=k) for k in range(n)]), group_id=g)
        for g, n in enumerate([2, 3, 1])
    ])

    groups.to_arrow()      # TypeError: Cannot convert buffer of type 'Array'
    pickle.dumps(groups)   # for ArraySetEntry to Arrow.

**Why.** The leaf ``members.value`` is stored as one column with a row per
group. With nested Sets of different lengths, the column is ragged (an
Awkward array), but the schema entry of the leaf is static (``value: float``
has a fixed shape), and the Arrow export accepts only NumPy arrays for static
entries. When all nested Sets have the same length, the column is a regular
NumPy array and the export works.

**Affected.** For example a ``Set[Class2D]`` whose classes hold different
numbers of particles, which is the usual case.

**Instead.** Keep such elements in a ``Collection`` (``B.Collection[Class2D]``):
every slot of a Collection stores its own columns, so nested Sets of different
sizes pickle and travel between stages.

Test: ``test_pickle_roundtrip_nested_sets_of_different_sizes`` in
``tests/scipion_bridge/struct/test_set_arrow.py``.

The capacity of a nested Set is not checked on whole-column writes
------------------------------------------------------------------

A nested Set declared with a capacity accepts a column with more elements
per row when the whole column is written at once:

.. code-block:: python

    class Movie(B.Struct):
        frames: B.Set[Frame] = B.Set[Frame](capacity=10)

    movies = B.Set[Movie](capacity=2)
    movies["frames"]["pixels"] = np.ones((2, 15, 64, 64))   # accepted
    len(movies[0].frames)                                   # 15

Indexing beyond the capacity raises (``movies[0].frames[10]``: ``IndexError``),
and the capacity of the outer Set is checked. Check the length of the data
yourself before such a write.

Nested Sets with a capacity are also not preallocated: their leaves are
stored as ragged columns, one row per element of the outer Set, like nested
Sets without a capacity. A static allocation (one array of shape
``(outer, capacity, ...)``) is planned; the tests for it in
``tests/scipion_bridge/struct/test_set.py`` are skipped as "Pending storage
engine implementation". Some of them check internals that no longer exist;
the behaviour they describe, e.g. ``movies[0].frames[0].pixels = ...`` on an
empty Set, works.

Test: ``test_a_nested_set_column_longer_than_its_capacity_is_rejected`` in
``tests/scipion_bridge/struct/test_set.py``.

Sets assigned into Struct fields or Collection slots lose their references
--------------------------------------------------------------------------

Columns that are rows of input files (references, e.g. the ``pixels`` of
particles read from their stacks) stay references through views, ``concat``,
``chunk`` and ``astype``. Assigning a Set to a field of a Struct, or a Struct
to a slot of a Collection, copies its columns instead, which makes them
computed (owned) columns:

.. code-block:: python

    def classify(x):
        particles, classifier = x               # pixels: references
        labels = classifier.predict(particles["embeddings"])
        classes = {
            int(k): spa.Class2D(                # copies the selected rows: owned
                particles=particles[labels == k],
                class_id=int(k),
            )
            for k in np.unique(labels)
        }
        return B.Collection[spa.Class2D](size=n, items=classes)   # copies again

The selection ``particles[labels == k]`` keeps the reference (a view narrows
it to the selected rows); the copies happen in the descriptor of the Struct
field (``Set.__set__``) and in ``Collection.__setitem__``. Selecting by
positions instead of a mask does not change this.

**Effect.** Spills after such a stage write the pixels instead of
references (on the 2D pipeline: ~78 MB per item behind the classification, a
spill estimate saving of 0%). Values are correct.

**The report does not show it.** The *Key paths* panel shows ``·`` for stages
whose items are Collections, and no warning is raised: a reference counts as
lost when the same path turns from reference to owned, but here the path
``pixels`` of the input disappears and ``particles.pixels`` appears in every
slot of the Collection. The byte columns of the provenance layer
(``cached_bytes`` turning into ``owned_bytes``) show it.

**Instead.** None yet. Carrying references through these assignments is
planned for Collection slots and Structs; a ``Set[Class2D]``, whose particles
are below two axes, will keep copying.

Struct specialization is not implemented
----------------------------------------

Dimensions and arguments of Structs (``B.Dim``, ``B.Arg``) can be declared,
but not specialized: ``Struct.static(H=64)`` does not exist, and a nested
``Set`` whose capacity is a ``B.Arg`` keeps an unknown capacity. Declare fixed
shapes and capacities directly (``B.Array(shape=(64, 64))``,
``B.Set[Frame](capacity=10)``), or use ``None`` for dimensions that vary.

The tests are skipped as "Specialization feature WIP" and "Pending storage
engine implementation" in ``tests/scipion_bridge/struct/test_struct.py`` and
``test_set.py``.

Not implemented yet
-------------------

These raise ``NotImplementedError`` instead of guessing:

``astype`` with Collections
   ``Op.astype`` and the cast behind it change the type of Sets and Structs,
   not of a ``Collection`` (as the item or as the target). Cast the items
   of the Collection in a ``map`` instead. ``Collection`` *fields* of a Struct
   are cast with it, if both types declare the same size.

Upstream issues
---------------

Awkward Array ignores the offset of the validity bitmap of sliced Arrow
arrays (``ak.from_arrow(column.slice(i, 1))`` can read the validity of the
wrong rows). The library does not rely on it: it reads the validity of
nullable columns from Arrow itself. The test
``test_awkward_reads_validity_of_sliced_arrow_arrays`` in
``tests/scipion_bridge/struct/test_pickle.py`` starts to pass once Awkward
fixes it.
