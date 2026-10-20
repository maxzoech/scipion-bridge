Streaming
=========

.. currentmodule:: scipion_bridge.core.streaming

The streaming module is a **declarative** API for data processing graphs.
Chaining operations describes a graph; compiling the graph with a backend
turns it into running processes; sending data into the compiled pipeline
executes it. This section explains the model behind the API and each of its
operations.

.. code-block:: python

    from scipion_bridge.core.streaming import Pipeline, Source

    particles = Source("particles")
    done = particles.chunk(256).map(embed).map(cluster).sink(save)

    with Pipeline.from_sink(done) as pipeline:
        for batch in batches:
            pipeline.send(particles=batch)

.. contents:: On this page
   :local:
   :depth: 2

The mental model
----------------

Four ideas explain the behavior of every operation.

**Building a graph runs nothing.** Every method of an ``Op`` returns a new
node connected to the previous one. Inside a protocol, ``steps()`` returns
the last node of the graph and is called once, when the pipeline is compiled.
The functions passed to the operations are called later, by the backend, for
each item.

**Items flow through stages.** An *item* is whatever travels along an edge of
the graph, usually a ``Set`` batch. Every operation is a *stage* that
receives items and emits items. Stages run concurrently: while one stage
processes batch *n*, the stage before it already processes batch *n + 1*.
Within a stage, items are processed one at a time, in arrival order.

**FLUSH ends the stream.** When the input is exhausted, a ``FLUSH`` signal
travels through the graph. Stateful operations (``chunk``, ``collect``,
``combine_latest``, ``group_by``) react to it by emitting what they still
hold, and sinks finalize their output. A flush is a barrier: it completes only
after all earlier items have passed through every downstream stage.

**Two layers.** The operations are the *surface API*. They are *lowered* into
five intermediate representation (IR) primitives (sources, maps,
accumulators, sinks and demultiplexers), and a backend only implements those.
The surface API can grow without changes to backends, and new backends do
not need to know about every operation. See
:doc:`/developer/streaming_internals`.

Operations at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Cardinality
     - Description
   * - ``map(func)`` / ``map_batch(func)``
     - 1 → 1
     - Apply ``func`` to every item.
   * - ``map_element(func, ...)``
     - 1 → 1
     - Apply ``func`` to every *element* of every item, in parallel.
   * - ``chunk(n, drop_last=False)``
     - N → M
     - Re-batch incoming ``Set`` items into sets of exactly ``n`` elements.
   * - ``collect(n)``
     - N → 1
     - Collect the first ``n`` elements of the stream into one ``Set``.
   * - ``flatten()``
     - 1 → N
     - Emit every element of an incoming iterable as its own item.
   * - ``scan(func, initial=None)``
     - 1 → 1
     - Fold the stream with ``func`` and emit every running state.
   * - ``combine_latest(other)``
     - (N, M) → N
     - Pair every item with the latest item of another stream.
   * - ``group_by(key, pipeline)``
     - N → N
     - Run a sub-pipeline separately for the items of every key.
   * - ``sink(callback)`` / ``write_to(writer)``
     - terminal
     - Consume the stream.
   * - ``checkpoint(writer)``
     - pass-through
     - Attach a sink without ending the stream.

Articles in this section
------------------------

.. toctree::
   :maxdepth: 1

   sources_and_maps
   stateful_operations
   group_by
   sinks
   pipelines

* :doc:`sources_and_maps` covers the entry points of a graph and the
  stateless operations ``map`` and ``map_element``.
* :doc:`stateful_operations` covers ``chunk``, ``collect``, ``flatten``,
  ``scan`` and ``combine_latest``, with their exact behavior at the end of a stream.
* :doc:`group_by` explains per-key sub-pipelines and how keys share worker
  processes.
* :doc:`sinks` covers terminal nodes, the ``SinkWriter`` protocol and writing
  a custom writer.
* :doc:`pipelines` covers compilation, sending data, flushing, statistics,
  backpressure and error handling.
