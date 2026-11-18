Sinks
=====

.. currentmodule:: scipion_bridge.core.streaming

A graph ends in one or more sinks, which consume the items that reach them:
printing them, collecting them, or persisting them. This article describes the
ways to attach a sink, the ``SinkWriter`` protocol that every sink implements,
and how to write a writer of your own.

.. contents:: On this page
   :local:
   :depth: 2

Attaching sinks
---------------

.. code-block:: python

    stream.sink(print)                       # a callback, handy for debugging
    stream.write_to(writer)                  # a SinkWriter
    stream.tap().map(...)                    # keep the items in the run store and continue

``sink`` and ``write_to`` return the sink node, which is what
``Pipeline.from_sink`` compiles. To keep intermediate items without ending
the stream, use ``tap()`` or ``checkpoint()``
(:doc:`checkpoints`): the backend writes them to the run store.

.. important::

   A pipeline consists of the nodes upstream of the sinks passed to
   ``Pipeline.from_sink``. A sink on a side branch (``stream.write_to(w)``
   while the graph continues from ``stream``) must be passed to
   ``from_sink`` as well to be compiled. Protocol runners compile only the
   protocol's final output, so a side branch inside ``steps()`` has no
   effect.

The SinkWriter protocol
-----------------------

A ``SinkWriter`` is any object with two asynchronous methods:

.. code-block:: python

    class SinkWriter(Protocol):
        async def write(self, item) -> None: ...     # commit one item durably
        async def finalize(self) -> None: ...        # flush and release, on FLUSH

The sink stage awaits ``write`` for every item, in order, and ``finalize``
once, when the stream is flushed. ``flush()`` on the pipeline returns only
after ``finalize`` has completed, so a writer can rely on all items having
been written by then.

A callback passed to ``sink`` is wrapped in a ``CallbackSinkWriter``, which
runs the callback in a thread, so a slow callback does not block the stage
from accepting further items.

Sinks keep no state across items
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sink writes every item on its own. Anything computed across items, such as
the running average of a class or a count, belongs in a stage of the
pipeline (``scan``, ``accumulate``, ``collect``), and the sink writes what
that stage emits:

.. code-block:: python

    # Not: a writer that adds every chunk of classes to averages it keeps.
    averages = classes.scan(add_to_averages, initial=empty_averages)
    averages.write_to(AveragesWriter("averages.npz"))   # writes the latest

There, lineage records which chunks every average was made from, and
checkpoints will save the state with the rest of the pipeline. A writer
that keeps state of its own loses it in a crash, and with a run store a
continued run skips the items that were already written (see below), so its
state would miss them.

Where a sink runs
^^^^^^^^^^^^^^^^^

On Ray, a sink runs in an actor of its own, not in the driver. A callback that
appends to a list in the driver's memory appends to a copy in the actor
instead. Collect results through a Ray actor, write them to storage, or use
``print`` while debugging (Ray forwards the output of actors to the driver).

``TensorStoreSinkWriter``
-------------------------

The Ray backend ships a writer that persists ``Struct``, ``Set`` and
``Collection`` items to Zarr through TensorStore:

.. code-block:: python

    from scipion_bridge.backend.ray.sinks import TensorStoreSinkWriter

    stream.write_to(TensorStoreSinkWriter("output.zarr"))

Every leaf field becomes a Zarr array under the given directory, at the path
of the field (``ctf/defocus_u``, ...). The writer follows the
:ref:`reduction semantics <reduction-semantics>` of the type system: set
columns are appended along their first axis, and other arrays are
overwritten. Columns must be dense; a field whose elements differ in shape
raises ``ValueError``.

With a run store, the writer owns its directory (see below): before its
first item, it cuts every array back to the rows the run store committed.

When a protocol is run with ``RayPipelineRunner``, the sink receives the
protocol's output dictionaries rather than the containers themselves; see
:doc:`/user_guide/backends/ray`.

.. _commits-and-reruns:

Commits and reruns
------------------

With a run store (``RayBackend(run_store=...)``, or ``--run-dir`` of the
runner), a sink records a commit after every write: the id of the item, the
number of its elements, and what the writer reported of the write. A run
continuing in the store of an earlier run, with the same protocols,
parameters and inputs, skips at its sinks the items committed before that
it receives again with the same ids. The stats count them as ``skipped``. A
run store of a run that computed something else is refused
(``RunStoreMismatchError``).

.. note::

   Item ids follow the order items arrive in. A sink skips only when the
   items reach it in the same order as before: with one loader, and no
   stage that depends on arrival order before it (``scan``, ``collect``, a
   ``group_by`` with several workers). With several loaders, every run
   chunks the inputs differently and the sinks skip nothing. Reruns that
   skip by input rows, whatever the order, are planned (see
   :doc:`checkpoints`, "Reruns").

Commits are written in batches, like lineage records, so a crash can leave
items written without a commit. A run continuing it writes them again. That
is harmless for writers that replace what they wrote, e.g. a file named by
the item; a writer that appends implements ``CommittingSinkWriter`` to undo
those writes:

.. code-block:: python

    class CommittingSinkWriter(SinkWriter, Protocol):
        async def write_identified(self, item, lid: bytes) -> Receipt | None: ...
        async def recover(self, receipt: Receipt | None) -> None: ...

The sink calls ``write_identified`` with the item id instead of ``write``,
and keeps the receipt it returns (JSON values) in the commit. Before its
first item, a continued run calls ``recover`` with the receipt of the last
commit, or ``None`` if there is none, and the writer removes everything
written after it. ``TensorStoreSinkWriter`` records the rows of its arrays.
``StoreSinkWriter(path)`` keeps every item in a file named by its id, which
can be read while the run goes (``writer.read(item_id)``).

Elements written twice
^^^^^^^^^^^^^^^^^^^^^^

Item ids tell items apart, not their elements: a particle can reach a sink
in two items, e.g. through two branches. The sink can check element ids.
Protocols do not attach sinks: the runner does, and sets the check
(``--unique-elements warn`` on the command line):

.. code-block:: python

    runner = RayPipelineRunner(
        protocol,
        sink=writer,
        unique_elements=B.UniqueElements.WARN,
    )

In a pipeline built by hand, ``write_to`` attaches the writer and sets it:

.. code-block:: python

    stream.write_to(writer, unique_elements=B.UniqueElements.RAISE)

``WARN`` logs the items with elements written before, ``RAISE`` fails the
pipeline, and ``OFF`` (the default) checks nothing. Elements are compared
per field of the Sets holding them, e.g. the members of all classes, and
elements without ids are not checked. The sink keeps the ids it wrote,
about 100 bytes per element.

Writing a custom writer
-----------------------

A custom writer implements the two methods. A writer that adds a line with
the number of particles of every item to a CSV file:

.. code-block:: python

    import asyncio
    from pathlib import Path


    class BatchSizeWriter:
        def __init__(self, path: Path):
            self.path = path

        async def write(self, item) -> None:
            await asyncio.to_thread(self._append, f"{len(item)}\n")

        def _append(self, line: str) -> None:
            with self.path.open("a") as file:
                file.write(line)

        async def finalize(self) -> None:
            pass

Guidelines for writers:

* ``write`` should return only once the item is durably stored, if the writer
  is meant to persist data; the pipeline treats a returned ``write`` as done.
* Blocking I/O should run in a thread (``asyncio.to_thread``), so that the
  sink keeps accepting items.
* The writer is pickled into the sink's process. Open files and connections
  lazily, in the first ``write``, not in ``__init__``.
* ``finalize`` may be called again after a later stream on the same pipeline.
* A writer keeps no state across items (see above). An appending writer, like
  this one, writes items twice after a crash; implement
  ``CommittingSinkWriter`` to undo that.
