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

Lineage cannot tell where elements with Scipion item ids come from
------------------------------------------------------------------

The lineage of a run (``--run-dir``, see :doc:`/user_guide/backends/ray`)
traces a particle read from a stack back to the stack and its row. It cannot
do the same for a particle whose element id is a Scipion item id:

.. code-block:: python

    graph = LineageGraph.of(FileRunStore("runs/2d_classes"))

    origin = graph.trace_element(*stack_particle)     # read from a .star/.xmd
    origin.file, origin.row                           # the stack, row 17

    origin = graph.trace_element(*scipion_particle)   # read from a SetOfParticles
    origin.file                                       # None
    origin.created_by.kind                            # "source": reported as
                                                      # created by the source stage

**Why.** An element id is a *namespace* and a *position*. The namespace is a
16-byte hash, which cannot be turned back into what it names; lineage needs
a record of what every namespace means. Today it has such a record only for
files, and only indirectly:

* The resolvers of stacks set the namespace of a particle to the hash of its
  stack (path, size and modification time), and its ``pixels`` reference the
  same stack. The source stage records the files the references of an item
  point to, and ``LineageGraph`` maps the hash of every recorded file back to
  it. The reference explains the namespace, not the id itself.
* The Scipion 3 resolver (``SetOfParticles``, the classes of a
  ``SetOfClasses2D``) sets the namespace to the hash of the set's database and
  table, and the position to the Scipion item id. It reads the pixels into
  arrays, so the item references no file, and even a referenced file would
  not match a namespace that is not a file hash.

``trace_element`` then misses the table of files and falls back to the item
records: the first record that holds the element is the source item, so the
element is reported as *created* by the source stage. Its position, the
Scipion item id, is correct, but nothing says which set, database or table it
belongs to.

The same applies to any namespace that no reference explains: elements
whose rows were read into arrays before the source stage, ids set by
resolvers that build no references, and the namespaces ``tag`` derives (a
tagged view cannot be traced to its particle by its namespace, only through
the items that hold it).

**Affected.** Nothing in practice yet. The Ray runner reads particles from
files, whose elements are traced correctly, and the Scipion 3 backend records
no lineage. It becomes relevant with the interoperability with Scipion: a
Ray run reading Scipion sets, and a ScipionWeb resolver, whose elements are
``scipionItemId`` values of a ScipionWeb set, are exactly this case.

**Instead.** Read the item id from the element (its position) and identify
the set from the inputs in the manifest of the run.

**Planned**, with the interoperability with Scipion:

1. A description of every namespace, attached by the resolver that sets the
   ids and carried with them through views, ``concat``, casts and pickling, as
   references are: a file (path, size, modification time), a Scipion table
   (database, table prefix, set class), a ScipionWeb set (server, project,
   set), or a tag (namespace, name, discriminator).
2. The source stage records the descriptions of the namespaces of every item,
   replacing the files of its references, and ``tag`` records its
   derivations.
3. ``trace_element`` answers every input element alike: "row 17 of stack X",
   "item 4711 of table ``Class001_`` of classes.sqlite", "``scipionItemId``
   4711 of ScipionWeb set Y". ``created_by`` then only names rows a stage
   created.

Test: ``test_an_element_with_a_scipion_item_id_is_an_input_element`` in
``tests/scipion_bridge/streaming/test_run_store.py``.

.. _known-issue-protocol-inheritance:

Protocol subclasses do not inherit inputs, parameters and resources
-------------------------------------------------------------------

A protocol collects the inputs, parameters and resources it declares in its
own class body. A subclass does not see those of its base: it has only what
it declares itself.

.. code-block:: python

    class Accumulate(B.Protocol):
        classes: B.Input[B.Collection[spa.Class2D]] = B.Input()
        ctf_correct: B.Field[bool] = B.Field(default=False)
        ...

    class AccumulateScored(Accumulate):
        classes: B.Input[B.Collection[ScoredClass2D]] = B.Input()

    list(AccumulateScored().configuration.parameters)   # [], no ctf_correct

**Instead.** Declare every field again in the subclass. Reusing the
declaration of the base keeps both the same:

.. code-block:: python

    class AccumulateScored(Accumulate):
        classes: B.Input[B.Collection[ScoredClass2D]] = B.Input()
        ctf_correct: B.Field[bool] = Accumulate.ctf_correct

A subclass item type cannot feed an input of its base type
-----------------------------------------------------------

When protocols are chained, an output feeds an input only if the item types
are the same: ``Collection[Sub]`` does not feed an input declared as
``Collection[Base]``, even when ``Sub`` subclasses ``Base``. The same holds
for ``Set`` (see also :ref:`sharp-bits-set-sub` in :doc:`sharp_bits`).

.. code-block:: python

    class ScoredClass2D(spa.Class2D):
        relative_cost: float

    # AlignPCA outputs Collection[ScoredClass2D];
    # AccumulateClassAverages takes Collection[Class2D].
    AlignPCA() | AccumulateClassAverages()
    # ChainError: Output 'classes' of type Collection[ScoredClass2D]
    # cannot feed input 'classes' of type Collection[Class2D].

This is so even when the second protocol only reads fields of ``Base``, as
``AccumulateClassAverages`` only reads the fields of ``Class2D``.

**Instead.** Write a second protocol that declares the subclass as its input
type. Since protocol subclasses do not inherit fields
(:ref:`known-issue-protocol-inheritance`), it declares all of them again:

.. code-block:: python

    class AccumulateScoredClassAverages(AccumulateClassAverages):
        classes: B.Input[B.Collection[ScoredClass2D]] = B.Input()
        ctf_correct: B.Field[bool] = AccumulateClassAverages.ctf_correct
        ...  # every other field of AccumulateClassAverages

**Possible fix.** Covariant item types: an output whose item type subclasses
the item type of an input could feed it. This is not planned.

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
