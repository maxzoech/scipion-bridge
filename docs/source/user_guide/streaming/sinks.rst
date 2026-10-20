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
    stream.checkpoint(writer).map(...)       # attach a sink and continue

``sink`` and ``write_to`` return the sink node, which is what
``Pipeline.from_sink`` compiles. ``checkpoint`` attaches a sink and returns the
stream it was attached to, so the graph can continue after it.

.. important::

   A pipeline consists of the nodes upstream of the sinks passed to
   ``Pipeline.from_sink``. A sink attached with ``checkpoint`` is downstream
   of the stream, not upstream of the final sink, so it must be passed to
   ``from_sink`` as well to be compiled. Protocol runners compile only the
   protocol's final output, so a ``checkpoint`` inside ``steps()`` currently
   has no effect.

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

When a protocol is run with ``RayPipelineRunner``, the sink receives the
protocol's output dictionaries rather than the containers themselves; see
:doc:`/user_guide/backends/ray`.

Writing a custom writer
-----------------------

A custom writer implements the two methods. A writer that stores the number
of particles per batch in a CSV file:

.. code-block:: python

    import asyncio
    from pathlib import Path


    class BatchSizeWriter:
        def __init__(self, path: Path):
            self.path = path
            self._sizes: list[int] = []

        async def write(self, item) -> None:
            self._sizes.append(len(item))

        async def finalize(self) -> None:
            text = "\n".join(str(n) for n in self._sizes)
            await asyncio.to_thread(self.path.write_text, text)

Guidelines for writers:

* ``write`` should return only once the item is durably stored, if the writer
  is meant to persist data; the pipeline treats a returned ``write`` as done.
* Blocking I/O should run in a thread (``asyncio.to_thread``), so that the
  sink keeps accepting items.
* The writer is pickled into the sink's process. Open files and connections
  lazily, in the first ``write``, not in ``__init__``.
* ``finalize`` may be called again after a later stream on the same pipeline;
  reset per-stream state there.
