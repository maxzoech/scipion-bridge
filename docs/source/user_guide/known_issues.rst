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
sizes pickle and travel between stages. With elements that carry ids (e.g.
particles a pipeline read), such a Set already raises when it is built (see
above).

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

Replacing the nested Set of one element clears it for every element
-------------------------------------------------------------------

Assigning a new nested Set to one element of a ``Set`` destroys the nested
Sets of all other elements:

.. code-block:: python

    groups = B.Set[Group]([...])            # three groups of 2, 3 and 1 members

    groups[1].members = B.Set[Member]([Member(value=99.0), Member(value=98.0)])

    groups[1].members["value"]              # [[99.], [98.]]
    groups[0].members["value"]              # UninitializedFieldError
    groups[2].members["value"]              # IndexError
    groups["members"]["value"].tolist()     # [None, [[99.0], [98.0]]]

**Why.** The members of all groups are one column with a row per group. Before
writing the new members, the assignment clears the old ones, and the clear
removes the whole column instead of the row of group 1. Only group 1 is then
written back.

**Instead.** Write values into the existing members
(``groups[1].members[0].value = 5.0`` or
``groups[1].members["value"] = ...`` with the same number of members), or
replace the whole element: ``groups[1] = Group(members=..., group_id=1)``.
Better still, keep such elements in a ``Collection``, whose slots store their
own columns. Assigning a nested Set to a standalone Struct or to a
``Collection`` slot is not affected.

Test: ``test_replacing_the_members_of_one_group_keeps_the_other_groups`` in
``tests/scipion_bridge/struct/test_set.py``.

Nested Sets of elements with ids inside a Set raise NotImplementedError
-----------------------------------------------------------------------

Building a ``Set`` whose elements hold nested Sets of elements that carry
element ids raises ``NotImplementedError``. Particles a pipeline reads carry
ids (their row in a stack, or their Scipion item id), so a ``Set[Class2D]``
built from them raises on every write path:

.. code-block:: python

    classes = B.Set[spa.Class2D]([
        spa.Class2D(particles=particles[labels == k], class_id=int(k))
        for k in np.unique(labels)
    ])                                  # NotImplementedError

    classes[i] = spa.Class2D(...)       # NotImplementedError
    classes[i].particles = particles    # NotImplementedError

**Why.** Element ids let lineage trace an output back to the input elements it
holds, and they are what ScipionWeb stores as ``scipionItemId``. In a
``Set[Class2D]``, the particles of all classes are one column with a row per
class: they sit below two axes (the class, then the particle within it), and
the number of particles differs per class, so the column is ragged. Ids, like
references to the input files, are stored per column as compact segments,
which cannot describe ragged columns yet. Copying the particles would silently
drop their ids: lineage could no longer tell which input particle a member
is. Raising makes this explicit. No values are lost either way.

**Instead.** Use a ``Collection``: every slot stores its own columns, so the
particles of a class sit below one axis and keep their ids and their
references to the input files (spills write references, not pixels). The
Scipion resolver (``SetOfClasses2D`` → ``Collection[Class2D]``) and the
protocols of ``scipion-em-foundation-models`` use it:

.. code-block:: python

    keys = np.unique(labels)
    classes = B.Collection[spa.Class2D](
        size=len(keys),
        items={
            i: spa.Class2D(particles=particles[labels == k], class_id=int(k))
            for i, k in enumerate(keys)
        },
    )

**What changed for existing code.** Two uses of ``Set[Class2D]`` with
particles from a pipeline worked before and raise now: a ``Set[Class2D]``
built and used within one stage, and one with classes of equal size sent
between stages (classes of different sizes already failed, see below).
Particles without ids, e.g. built by hand in tests, still form a
``Set[Class2D]``; their references (if any) are copied as bytes.

**Planned.** Storing the ids and references of nested Sets per outer element
(a list of segments per class), which removes the guard.

Tests: ``test_a_set_of_groups_from_members_with_ids_raises`` and
``test_a_set_of_groups_from_members_without_ids_copies`` in
``tests/scipion_bridge/struct/test_ids.py``.

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
