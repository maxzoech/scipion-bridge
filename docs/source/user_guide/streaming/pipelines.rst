Pipelines
=========

.. currentmodule:: scipion_bridge.core.streaming

A graph becomes executable when it is compiled into a ``Pipeline``. This
article describes compilation, the life cycle of a pipeline, the statistics it
reports, and how it bounds memory and reports errors. Protocols are usually
run through a runner that does all of this (see
:doc:`/user_guide/backends/ray`); the ``Pipeline`` API is what the runners
use, and it is available directly for graphs without a protocol.

.. contents:: On this page
   :local:
   :depth: 2

Compiling a graph
-----------------

.. code-block:: python

    pipeline = Pipeline.from_sink(*sinks, backend=None)

``from_sink`` lowers the graph that leads to the given sinks into IR and
compiles it with a backend. Only nodes upstream of the given sinks are part of
the pipeline; a branch that does not lead to one of them is not compiled. If
``backend`` is omitted, the backend provided by the dependency injection
container is used, which is the Ray backend when Ray is installed.

Compilation starts the backend's processes. On Ray, every stateful operation
and every sink becomes an actor, which takes on the order of a second per
actor; compile a pipeline once and send all data through it.

Life cycle
----------

.. list-table::
   :widths: 25 75

   * - ``send(**inputs)``
     - Push one item into each named source, e.g.
       ``pipeline.send(particles=batch)``. Blocks while the first stage has no
       room (see `Backpressure`_).
   * - ``flush()``
     - Send ``FLUSH`` into every source and wait until it has passed through
       every stage: stateful operations have emitted what they hold, all
       in-flight work has finished, and every sink has been finalized.
   * - ``stats()``
     - Per-stage ``StageStats``. After ``close()``, the stats the stages had
       when the pipeline was closed.
   * - ``close()``
     - Release the backend's resources (on Ray, kill the actors). Closing a
       closed pipeline does nothing; ``send()`` and ``flush()`` on it raise
       ``RuntimeError``.

Close every pipeline: on Ray, its actors stay alive until it is closed, also
after it is no longer referenced, since the stages hold references to each
other. A pipeline is a context manager that does it. Leaving the ``with``
block without an exception calls ``flush()`` and then ``close()``. On an
exception, the pipeline is only released: flushing a failed pipeline would
only raise again and hide the original error, and its stats are not
available.

.. code-block:: python

    with Pipeline.from_sink(out) as pipeline:
        for batch in batches:
            pipeline.send(particles=batch)
    print(pipeline.stats())

``RayPipelineRunner.run()`` closes its pipeline the same way, so a runner
runs once.

Statistics
----------

``stats()`` returns a ``StageStats`` per stage, keyed by a label such as
``2:chunk(256)``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``items_in``, ``items_out``
     - Items processed and forwarded (excluding FLUSH).
   * - ``idle_s``
     - Time spent waiting for input. High values mean the stage is *starved*
       by its upstream.
   * - ``process_s``
     - Time spent in the stage logic. For a map, the time from submitting a
       call until its result is ready; on an executor pool, without the wait
       for an executor.
   * - ``wait_s``, ``wait_max_s``
     - Time items waited before the stage took them, in total and at most:
       in the mailbox of the stage, or, for a map on an executor pool, for
       an executor (Ray backend). High values mean the stage is slower than
       its upstream.
   * - ``blocked_s``
     - Time spent waiting for room downstream. High values mean the stage is
       *throttled* by the maps after it.
   * - ``emit_s``
     - Time spent forwarding results, including serialization.
   * - ``buffered_peak``
     - Most items that waited for the stage at once. High values mean the stage
       is slower than its upstream.
   * - ``spilled``, ``fetch_s``, ``spill_s``
     - Items written to the spill store, and the time spent reading waiting
       items and spilling them (Ray backend).
   * - ``first_in_t``, ``last_out_t``
     - When the stage received its first item and forwarded its last, in
       seconds since the epoch (Ray backend); ``None`` before.

From these, ``throughput_of`` derives, for a stage that was active:

* ``throughput``: items forwarded per second, from its first item to its
  last;
* ``s_per_item``: the mean time to process an item;
* ``busy``: the items in processing at once, on average. A stage on an actor
  of its own is at most 1 (always working); a map on an executor pool can
  reach the calls its executors run at once;
* ``wait_mean_s``: the mean wait of an item before the stage took it.

To find the bottleneck, look for the stage with the highest ``busy`` whose
items wait (``wait_s``, ``buffered_peak``) while the stages after it are
idle. The report of a run lists the stages by ``busy`` and names it.

The measurements are a few clock reads per item and stage, about a
microsecond, against the hundreds of microseconds a stage spends on the
messages of every item; they are always on.

Backpressure
------------

Every stage has bounded buffers. When a stage falls behind, items wait in its
buffer; when the buffer is full, the stages before it wait, and eventually so
does ``send()`` on the driver. Memory use therefore stays bounded, no matter
how fast data arrives. The size of the buffers, and the option to spill
waiting items to disk instead of blocking, are backend settings; for Ray, see
:doc:`/user_guide/backends/ray`.

Two operations keep items in their own state, outside of the buffers:
``collect`` keeps its sample until it emits, and ``combine_latest`` keeps the
items that arrive before its other input produces a value. They never block
the pipeline, but their state can grow.

Errors
------

When a stage fails, its exception is stored and re-raised by the next
``send()`` or ``flush()`` on the driver. The failing stage also aborts the
sources of the pipeline, so the failure surfaces on the next ``send``
instead of only at the end of the stream, even if the failed stage would never
receive another item. After a failure, close the pipeline (or leave its
``with`` block) and compile a new one.

Requirements on backends
------------------------

Every backend provides the same guarantees, which the semantics of the
operations rely on:

* **Ordering.** A stage processes items one at a time, in arrival order.
* **Flush barrier.** ``flush()`` returns only after all items sent before it
  have passed through every stage and every sink has been finalized.
* **Error propagation.** An exception in any stage fails the pipeline and is
  raised on the driver.
* **Bounded buffering.** Buffers between stages are bounded.

How backends meet these guarantees is described in
:doc:`/developer/streaming_internals`.
