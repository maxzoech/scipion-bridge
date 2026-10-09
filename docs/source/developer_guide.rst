Developer Guide
===============

This guide is for contributors who extend or maintain Scipion Bridge. It
covers the internal architecture, the streaming intermediate representation,
how to write a backend, and how to test the library.

.. contents:: On this page
   :local:
   :depth: 2

Repository layout
-----------------

.. code-block:: text

    src/scipion_bridge/
    ├── core/
    │   ├── struct/        Struct, Set, Collection, schemas, storage, Arrow conversion
    │   ├── streaming/     Ops (surface API), IR, lowering, Pipeline, sinks
    │   ├── typed/         Type resolution graph, Dijkstra search, proxies
    │   ├── protocol/      Protocol, Input / Field / Resource, chaining, @resources
    │   ├── environment/   Service interfaces: shell exec, temp files, storage,
    │   │                  configuration, resources, compute declarations
    │   └── utils/         ARC, shell_command, AST helpers, Marker
    ├── backend/
    │   ├── standalone/    Default DI container
    │   ├── ray/           Ray streaming backend, runner, resource provider, sinks
    │   └── pyworkflow/    Scipion 3 protocol generation and resolvers
    ├── single_particle/   Particle types, proxies and resolvers
    └── visualize/         Visualization helpers
    tests/                 pytest suite mirroring the package structure

Coding conventions are in ``AGENTS.md`` at the repository root. In short:
Python 3.11, ``black`` formatting with trailing commas, ``match`` statements
over ``if`` / ``elif`` chains, guard clauses, no defensive ``getattr``
fallbacks, no ``TYPE_CHECKING`` or inline imports to avoid import cycles, and
no special cases for tests.

Dependency injection
--------------------

All side effects go through services provided by a
``dependency_injector`` container. Library code declares what it needs with
``Provide["service_name"]`` and ``@inject``:

.. code-block:: python

    @inject
    def _get_value(
        self,
        config_provider: ProtocolConfigurationProvider = Provide["protocol_config_provider"],
    ):
        ...

A backend defines a ``DeclarativeContainer`` with the same service names and
wires it into the ``scipion_bridge`` package:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Service
     - Standalone
     - Ray
     - PyWorkflow
   * - ``shell_exec``
     - ``StandaloneExecProvider``
     - ``StandaloneExecProvider``
     - ``runJob`` of the protocol
   * - ``temp_file_provider``
     - system temp dir
     - system temp dir
     - protocol ``tmp`` dir
   * - ``storage_provider``
     - Arrow
     - Arrow
     - Zarr in the protocol directory
   * - ``protocol_config_provider``
     - abstract
     - ``StaticProtocolConfigurationProvider``
     - Scipion form values
   * - ``resource_provider``
     - ``DefaultResourceProvider``
     - ``RayResourceProvider``
     - ``DefaultResourceProvider``
   * - ``streaming_backend``
     - ``RayBackend`` if installed
     - ``RayBackend``
     - ``RayBackend`` if installed

Ray workers are separate processes. Every Ray actor and task wires a
``RayContainer`` with the pipeline's parameter values before it runs user code,
so ``Field.value`` and resources work inside stages.

.. note::

   The ``storage_provider`` service creates array groups (in memory, or Zarr in
   the protocol directory). The data of ``Struct``, ``Set`` and ``Collection``
   does not go through it: it always lives in an in-memory ``RootEngine``,
   described in :ref:`struct-storage`.

.. _struct-storage:

Struct storage
--------------

``Struct``, ``Set`` and ``Collection`` hold no arrays themselves. Each instance
holds a *schema* (the fields traced from the class annotations) and a *storage*
(``instance._storage``). Every attribute access, slice and column read is
translated into a ``read`` or ``write`` on that storage. The storage is
implemented in two modules:

* ``core/struct/storage.py``: the storage classes ``_BaseStorage``,
  ``StorageView`` and ``RootEngine``.
* ``core/struct/buffers.py``: stateless helpers that build, pad, split and
  stack column buffers. They do not depend on the storage classes and are
  tested directly in ``tests/scipion_bridge/struct/test_buffers.py``.

The storage tree
^^^^^^^^^^^^^^^^

A ``RootEngine`` owns the data. Everything derived from a container (a nested
field, a row, a slice, a mask selection) shares that data through a
``StorageView``. A view is only a ``KeyPath`` (its ``root``) and a parent. It
forwards every operation to ``root_storage``, the ``RootEngine`` at the top of
the chain:

.. code-block:: text

    particles = B.Set[Particle](...)    RootEngine          root
    particles[1:3]                      └─ StorageView      root[1:3]
    particles[1:3][0]                      └─ StorageView   root[1]
    particles[1:3][0].ctf                     └─ StorageView root[1].ctf

Views are created with ``append(name)`` for a field and with ``narrow_index``,
``narrow_slice``, ``narrow_indices`` and ``narrow_mask`` for a selection. The
narrowing composes with the index already on the path, so the view always holds
an absolute position in the root's columns: row 0 of ``particles[1:3]`` is
``root[1]``. A boolean mask is turned into an index array
(``particles[mask]`` has the root ``root[[0, 2]]``), so the engine never sees
masks.

``_BaseStorage`` defines the interface. ``read``, ``write``, ``clear``,
``get_length`` and ``is_initialized`` are abstract. ``__contains__``
(``key in storage``), the ``narrow_*`` methods and ``concat`` are implemented
on top of them. A new storage, including a test mock, has to implement all five
abstract methods; Python refuses to instantiate it otherwise.

Key paths and columns
^^^^^^^^^^^^^^^^^^^^^

A ``KeyPath`` is a sequence of ``(name, index)`` components that starts with
``root``. ``RootEngine`` splits it into two parts:

* the **field key**, the names without ``root``, which identifies the column;
* the **effective index**, the indices of the components with the no-op
  ``slice(None)`` entries removed, which selects the part of the column.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Access
     - Field key
     - Effective index
   * - ``particles["score"]``
     - ``("score",)``
     - ``()``
   * - ``particles[1].pixels``
     - ``("pixels",)``
     - ``(1,)``
   * - ``particles[1:3]["score"]``
     - ``("score",)``
     - ``(slice(1, 3),)``
   * - ``particles[mask]["score"]``
     - ``("score",)``
     - ``(array([0, 2]),)``
   * - ``classes[1].particles["pixels"]``
     - ``("particles", "pixels")``
     - ``(1,)``

Every leaf field of the schema is one column, stored under its field key.
Nested structs add their field names to the key. A nested ``Set`` adds a
dimension: in a ``Set[Class2D]``, the column ``("particles", "pixels")`` has
one entry per class, and each entry holds the pixels of all particles of that
class. A ``Collection`` stores each slot under its own key
(``("0", "pixels")``, ``("1", "pixels")``, ...), which makes it row-wise.

The type of a column follows from the **data**, not from the schema. This is
the central invariant of the engine:

    A column is an ``np.ndarray`` whenever all of its rows are present and have
    the same shape. It is an ``ak.Array`` only when rows differ in shape or
    some rows are missing.

The schema only decides what is *allowed*: a static field (shape fully known,
scalars) always has rows of one shape, while a dynamic field (a ``None``
dimension) may have rows of different shapes.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Data
     - Column
     - Example (3 elements)
   * - Static field
     - NumPy ``(*outer, *element_shape)``
     - ``score: float`` → ``(3, 1)``
   * - Dynamic field, all rows of one shape
     - NumPy ``(*outer, *row_shape)``
     - ``pixels``, all 64 × 64 → ``(3, 64, 64)``
   * - Dynamic field, rows of different shapes
     - Awkward Array
     - ``pixels``, 64 × 64 and 32 × 32 → ``3 * var * var * float32``
   * - Dynamic field, some rows missing
     - Awkward Array with option type
     - ``pixels``, row 1 unwritten → ``3 * option[var * 2 * float32]``
   * - Below nested ``Set`` s of one length
     - NumPy, with one axis for the nested ``Set``
     - ``Set[Class2D]`` with 2 particles per class → ``particles.score``:
       ``(2, 2, 1)``
   * - Below nested ``Set`` s of different lengths
     - Awkward Array (the nested dimension is ragged)
     - ``Set[Class2D]`` with 1 and 2 particles → ``particles.score``:
       ``2 * var * 1 * float64``

A column becomes Awkward only when the data requires it: a row of a different
shape is written, a ragged column is concatenated, or rows stay missing. The
switch is one-way: reads never convert an Awkward column back, because checking
the regularity of a large Awkward Array on every read would cost what the
invariant is meant to save. Where a column is *built* from parts, regular data
is built as NumPy directly: ``stack_rows`` and ``stack_sequence`` when rows are
joined, ``concat_columns`` in ``concat``, and ``leaf_from_arrow`` when a column
is restored from Arrow. A whole-column write of an Awkward Array that only wraps
a NumPy array (``RegularArray`` and ``NumpyArray`` layouts, checked by
``has_numpy_layout`` without touching the data) is unwrapped without copying.

The reason is speed. Selecting rows of an Awkward Array (``buffer[mask]``) goes
through Awkward's generic indexing, which for large image stacks is orders of
magnitude slower than NumPy's, and converting an Awkward column to NumPy can
copy it. Keeping regular columns in NumPy makes masks, ``concat`` and
serialization cost what they cost in NumPy.

All columns of one container share their outer length; the engine relies on
this to allocate new columns at full length and to answer ``get_length``.

The two states of a column
^^^^^^^^^^^^^^^^^^^^^^^^^^

A column is in one of two dictionaries of the engine:

* ``_data``: the column as one buffer (NumPy or Awkward, following the
  invariant above), its normal state;
* ``_chunks``: a *pending* column, a list with one buffer per row, where a
  row that was never written is ``None``.

Only dynamic columns become pending. A row written into a dynamic column may
have a different shape than the others, which neither a NumPy array nor an
(immutable) Awkward Array can take in place. Instead, the first element write
splits the column into rows, and later writes replace rows in the list. The
rows are joined into one buffer again when more than one element is read:

.. code-block:: text

                     element write to a dynamic field
                       (split_rows, assign_rows)
           ┌─────────────────────────────────────────────┐
           │                                             ▼
     _data: buffer                               _chunks: rows
           ▲                                             │
           └─────────────────────────────────────────────┘
                 read of a column, slice or index array
                         (stack_rows)

    whole-column write: replaces either state with a new buffer in _data

A loop of element writes therefore costs one split and one stack, not one
rebuild of the column per write. ``split_rows`` keeps regular columns as
writable NumPy row views (copying a read-only column once), and turns ragged
or masked columns into Awkward rows. ``stack_rows`` uses ``np.stack`` when all
rows are NumPy arrays of the same shape. Otherwise it joins them with
``ak.concatenate`` plus ``ak.unflatten``, masking the ``None`` rows; if no row
is missing and the result turns out regular (e.g. Awkward rows that all have
the same shape), it is converted to NumPy (``regularized``).

Static columns stay in ``_data`` and are written in place.

Reading
^^^^^^^

``RootEngine.read(key, entry)`` works in three steps:

1. If the column is pending and every index is an integer, the element is
   taken directly from the row list, without stacking the rows.
2. Otherwise a pending column is stacked into a buffer first
   (``_materialize_pending``).
3. The buffer is indexed with the effective index. Without an index, the
   whole buffer is returned.

A column that does not exist raises ``UninitializedFieldError``, and so does a
*missing* element: one that reads back as ``None``, either a ``None`` row of a
pending column or a masked row of an Awkward column. An *empty* result is
data, not a missing element. A selection with no rows, for example
``particles[assignments == k]`` for a class without particles, returns empty
columns and can be assigned to another struct like any other set. Static
columns are allocated with zeros, so a missing element of a static field reads
as zeros.

``is_initialized(key)`` (and ``key in storage``) is true if any column lies
at or below the field key. That way a nested struct counts as initialized when
one of its fields is.

Writing
^^^^^^^

``RootEngine.write(key, entry, data)`` first checks the dtype: values must
cast to the field's dtype with ``same_kind`` casting (so writing floats to an
integer field raises ``TypeError``). A list of arrays is checked through its
first element, and Awkward Arrays and empty sequences are not checked. Then
one of three paths runs:

**Whole column** (no effective index). For a ``Set`` field, the length is
checked first: it may not exceed the capacity of the field, and without a
capacity it must equal the length of the sibling columns, pending ones
included. The sibling check is an assertion and is skipped under ``python -O``.
The buffer is then built from the data:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Data
     - Buffer
   * - Awkward Array
     - rejected for a static field; unwrapped to NumPy without copying if it
       only wraps a NumPy array; stored as is otherwise
   * - empty ``list`` / ``tuple``
     - empty NumPy column of shape ``(0, *element_shape)``, with length 0 for
       dynamic dimensions
   * - ``list`` / ``tuple`` of arrays of one shape
     - ``np.stack``
   * - ``list`` / ``tuple`` of arrays of different shapes
     - ragged Awkward Array (NumPy if the rows turn out to be regular)
   * - other ``list`` / ``tuple``
     - ``np.asarray`` with the field's dtype; an Awkward Array if the values
       are ragged
   * - scalar or array
     - ``np.asarray`` with the field's dtype. For a static field, the shape
       must end in the element shape

Columns built from a sequence are padded to the capacity of the field: with
zeros for NumPy columns and with missing rows for Awkward columns.

**Element of a static field.** The NumPy buffer is written in place, which is
why concurrent writes to different rows need no lock. If the column does not
exist yet, it is allocated with zeros. Its outer length is the largest of the
capacity of the field, the length of the sibling columns and the index being
written. A column that is too short for the index is copied into a larger
buffer. A read-only buffer (adopted from Arrow, see below) is copied once.
The data must broadcast to the selected part of the buffer, or a
``ValueError`` is raised.

**Element of a dynamic field.** The column becomes pending, as described
above, also when it is a NumPy array, and ``assign_rows`` writes the data into
the row list. Splitting a writable NumPy column gives row views without copying;
a read-only one (adopted from Arrow) is copied once:

* An integer index replaces one row. A list that is too short is extended with
  ``None`` rows first.
* A slice or an index array needs exactly one data element per selected row,
  and raises ``ValueError`` otherwise; the list never changes its length here.
* Nested indices (an element of a nested ``Set``) descend into the nested row
  lists.
* Any other index type raises ``TypeError``.

``clear(key)`` removes every column at or below the field key. ``Set``
assignment uses it to drop the old columns of the target field before writing
the new ones.

Length and capacity
^^^^^^^^^^^^^^^^^^^

``get_length(key)`` returns the length of the first column at or below the
field key, since all columns of a container share it. With an index, it is the
length of the selected element, and 0 if that element is missing. It returns
``None`` if no column exists yet.

A ``Set`` field with a capacity is padded to that capacity when it is written
as a whole from a list or tuple, and new columns written element by element
are allocated at the capacity. Its length is then the capacity, not the number
of written rows.

Concatenation
^^^^^^^^^^^^^

``_BaseStorage.concat`` (used by ``B.concat``) creates a new engine of the
root's type and fills it leaf by leaf. Each leaf must be initialized in all
inputs or in none of them. Static leaves are joined with ``np.concatenate``.
Dynamic leaves go through ``concat_columns``: if all inputs are NumPy arrays
with the same row shape, ``np.concatenate``; otherwise (rows of different
shapes, or Awkward inputs) ``ak.concatenate``, which accepts a mix of NumPy and
Awkward inputs. Empty inputs are skipped, so an empty selection never forces
the switch to Awkward. The inputs may be views; ``read`` returns only their
rows.

Serialization
^^^^^^^^^^^^^

Containers are not pickled through the engine. ``SchemaConvertible.__reduce_ex__``
exports the container with ``to_arrow``, which reads only the rows of the view
(stacking pending columns on the way), and the receiving side rebuilds it with
``from_arrow`` into a new ``RootEngine``. This is the path for Ray's object
store and for the ``PickleSpillStore`` that the Ray backend writes overflowing
items to.

``leaf_to_arrow`` and ``leaf_from_arrow`` (``core/struct/utils/arrow_utils.py``)
convert single columns:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Column
     - Arrow
     - Restored as
   * - static, NumPy
     - ``FixedShapeTensorArray`` (primitive array for 1-D columns)
     - NumPy
   * - dynamic, NumPy
     - nested ``FixedSizeListArray``, one level per row axis
     - NumPy (``is_regular_arrow`` / ``regular_arrow_to_numpy``)
   * - dynamic, Awkward
     - (nested) ``LargeListArray``, with validity bitmaps for missing rows
     - Awkward (``ak.from_arrow``)
   * - nested ``Set`` level
     - ``ListArray`` of the level below
     - NumPy if all nested Sets have one length, Awkward otherwise

On import, the Arrow type and null count decide, not the schema: null-free
nested fixed-size lists of numbers become NumPy, null-free nested-Set lists are
restored level by level, and everything else goes through ``ak.from_arrow``.
The fixed-size list type matches what ``ak.to_arrow`` produces for a regular
Awkward Array, so batches written by older versions restore as NumPy too, and
columns of different slots of a ``Collection`` still concatenate.

``from_arrow`` adopts Arrow buffers without copying where possible. Such
buffers are read-only, so the first in-place write copies them: a static column
in ``_writable_static_buffer``, a dynamic one in ``split_rows``.

``RootEngine`` also defines ``__getstate__`` and ``__setstate__`` for the rare
case where an engine is pickled directly. They drop and recreate its lock.

Thread safety
^^^^^^^^^^^^^

Element-wise access from several threads (for example ``Op.map_element``) is
safe as long as each thread reads and writes its own rows:

* Writes to different rows of an existing column touch different memory and
  take no lock.
* Transitions that replace a column object take the engine's lock and check
  their condition again inside it (double-checked locking): allocating a
  column, expanding or copying a static buffer, splitting a column into rows,
  and stacking pending rows.
* A transition publishes the new state before it removes the old one (rows
  before deleting the buffer, the buffer before deleting the rows), so a
  concurrent reader always finds the column.

Whole-column writes and ``clear`` are not thread-safe.

The struct tests (``tests/scipion_bridge/struct/``) cover the engine through
the public containers. ``test_root_engine.py`` tests ``RootEngine`` directly
with explicit schema entries, which is the easiest way to reproduce a storage
bug in isolation.

The streaming layer
-------------------

The streaming module has two layers: the user-facing **ops** and a minimal
**IR**. Backends implement only the IR.

Surface ops and lowering
^^^^^^^^^^^^^^^^^^^^^^^^

Every op is a ``Node`` with ``upstream`` and ``downstream`` edges and a
``lower(ctx)`` method. ``lower(nodes)`` walks the graph upstream from the
given sinks with a ``LoweringContext``. The context memoizes lowered nodes, so
shared subgraphs are lowered once, and it deduplicates sources by name. Each
node lowers itself to one IR node, and the context wires the IR edges in the
same order as the surface edges.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface op
     - Lowered to
   * - ``Source(name)``
     - ``IRSource(name)``
   * - ``map``, ``map_batch``
     - ``IRMap(func, compute)``
   * - ``map_element``
     - ``IRMap(ElementMapper(func, config))``. Worker pools are cached per
       process.
   * - ``chunk``, ``collect``, ``flatten``
     - ``IRAccumulate`` with the op's accumulate, initial-state and flush
       functions
   * - ``combine_latest``
     - ``IRAccumulate(tag_inputs=True)``
   * - ``group_by``
     - ``IRDemux(key_fn, template, source_name, max_keys)``
   * - ``sink``, ``write_to``, ``checkpoint``
     - ``IRSink(writer)``

The IR primitives
^^^^^^^^^^^^^^^^^

``IRSource(name)``
   An ingestion point. Items arrive through ``CompiledPipeline.send(name, item)``.

``IRMap(func, name, compute)``
   A stateless 1:1 transformation. ``compute`` is the ``ComputeAssignment``
   (resources plus group) that every call requires, or ``None``.

``IRAccumulate(accumulate_fn, initial_state_fn, flush_fn, tag_inputs)``
   A stateful N:M transformation. The backend keeps one ``state`` per stage:

   .. code-block:: python

       state = initial_state_fn()
       state, outputs = accumulate_fn(state, item)    # for every item
       state, outputs = flush_fn(state)               # on FLUSH (optional)

   With ``tag_inputs=True``, items are wrapped as ``Tagged(port, item)``, where
   ``port`` is the position of the sending stage among the node's upstream
   nodes. A node therefore cannot consume the same stream on two inputs, and
   lowering rejects that case.

``IRSink(writer)``
   A terminal node. It awaits ``writer.write(item)`` for every item and
   ``writer.finalize()`` on FLUSH.

``IRDemux(key_fn, template, source_name, max_keys, workers, outer_keyed)``
   Per-key routing for ``group_by``. ``template`` is the exit node of a
   separately lowered child graph whose only source is ``source_name``. With
   ``workers=None``, the backend compiles a copy of the template
   (``clone_ir``) for every new key. With ``workers=n``, the keys share up to
   ``n`` copies made by ``share_keys``. Their items carry the key as
   ``Keyed(key, item)``, maps apply their function to the item, and
   accumulators keep a state per key, created with the key's first item. An
   accumulator's ``flush_fn`` must therefore emit nothing on its initial
   state. ``WorkersFrom.BACKEND`` (the default) uses the backend's setting.
   Either way, the results are emitted as ``Keyed(key, result)``. A
   ``group_by`` nested in a shared copy has ``outer_keyed`` set: it receives
   ``Keyed(outer, item)`` and emits ``Keyed(outer, Keyed(key, result))``.

Execution contract
^^^^^^^^^^^^^^^^^^

A backend must guarantee:

* **Per-stage ordering.** A stage processes items one at a time, in arrival
  order. Accumulator state and stage logic never run concurrently.
* **FLUSH barrier.** ``flush()`` returns only after all items sent before it
  have passed through every stage, every ``flush_fn`` has run, and every sink
  has been finalized. A stage with several upstream stages forwards FLUSH once
  it has received FLUSH from all of them.
* **Error propagation.** An exception in any stage fails the pipeline, and the
  driver's next ``send`` or ``flush`` raises it.
* **Backpressure.** Buffering between stages is bounded.

Writing a backend
^^^^^^^^^^^^^^^^^

Implement two interfaces from ``scipion_bridge.core.streaming.backend``:

.. code-block:: python

    class MyBackend(StreamingBackendProvider):
        def compile(self, ir_sinks: list[IROp]) -> CompiledPipeline:
            ...   # traverse upstream from ir_sinks, build stages, wire routes


    class MyCompiledPipeline(CompiledPipeline):
        def send(self, source_name: str, value) -> None: ...
        def flush(self) -> None: ...
        def stats(self) -> dict[str, StageStats]: ...
        def close(self) -> None: ...

Then provide it as the ``streaming_backend`` service of a container. A
single-threaded reference backend is a good way to test new ops: run every
``IRMap`` inline, keep accumulator state in a dict, and call writers with
``asyncio.run``.

The Ray backend
^^^^^^^^^^^^^^^

``backend/ray/backend.py`` compiles the IR as follows:

* every source, accumulator, demux and sink becomes a named actor, a
  ``_PipelinedStage`` with a bounded inbox, a compute task, and an emitter task;
* maps are not actors. The stage upstream of a map submits each call as a Ray
  task, with up to ``queue_size`` calls in flight, and forwards the results in
  order. Long-running compute groups run their calls on a shared executor
  actor, which holds the GPUs until FLUSH;
* resource claims (``gpus``, ``min_vram``, ``cpus``) are translated into Ray
  task options by ``gpu_claim`` and the backend's ``_claim``;
* a failing stage aborts the sources, so the driver fails fast.

Type resolution internals
-------------------------

``core/typed/resolve.py`` keeps a single ``Registry`` holding a
``networkx.DiGraph``. Nodes are types. Edges carry the resolver class, its
module (namespace), and whether it accepts ``metadata`` or ``slice``.

* ``@resolver`` reads the ``value`` and return annotations and adds an edge.
  Function resolvers are wrapped in a generated class with a ``forward``
  method.
* For every registered type, *downcast* edges to its bases in the MRO are
  added, so resolvers for a base class also apply to its subclasses.
* A lookup restricts the graph to the visible namespaces (see
  :doc:`user_guide/type_resolution`) and runs a custom Dijkstra search
  (``dijkstra.py``). Ties are broken in this order: path weight, then
  resolvers local to the caller, then slice-aware resolvers, then the more
  specific module path.
* The result is a ``ComposedResolver``, a list of steps that can be called
  repeatedly or iterated in chunks without searching again.

Proxies and ARC
---------------

``core/utils/arc.py`` implements a process-wide ``FileReferenceCounter``:

* ``register_temporary_file`` / ``new_managed_file`` start tracking a path with
  a count of 1;
* ``add_reference`` / ``remove_reference`` are called when managed proxies for
  the path are created and garbage collected;
* when a count reaches 0, the file is deleted through the
  ``temp_file_provider``.

Adding a reference to a path that is not tracked yet emits a
``DeprecationWarning``: such paths are usually persistent user files, which
reference counting must not delete.

``@shell_command`` checks the AST of the decorated function at definition
time: the body must consist of a single ``pass`` statement. Calling the
original function first lets Python validate the arguments before any command
is built.

Testing
-------

Tests live in ``tests/``, mirror the package layout, and run with ``pytest``.
Run only the affected test files during development. The full suite takes
several minutes.

.. code-block:: bash

    pytest tests/scipion_bridge/struct/test_set.py
    pytest -m "not ray"          # skip tests that need the Ray test cluster
    pytest -m "not slow"

``tests/conftest.py`` wires the standalone container for every test.
Tests that request the ``ray_cluster`` fixture run on a session-wide local
Ray cluster with 2 CPUs and 2 logical GPUs, and they are marked ``ray``
automatically.

To test shell commands without running them, override the ``shell_exec``
provider with a mock and assert on the generated arguments:

.. code-block:: python

    @B.shell_command
    def my_command(i: str, *, value: int):
        pass


    def test_command(mocker, setup_test_container):
        exec_mock = mocker.Mock()
        with setup_test_container.shell_exec.override(exec_mock):
            my_command("/in", value=42)

        _, _, args, _ = exec_mock.call_args.args
        assert args == ["my_command", "-i", "/in", "--value", "42"]

Struct and set tests use a mock storage. Do not add code paths that exist only
for tests: give the mock the methods the library calls, and rely on duck
typing.

Before submitting a change:

.. code-block:: bash

    black .
    pyright
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
