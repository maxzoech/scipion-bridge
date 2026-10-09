Streaming
=========

.. currentmodule:: scipion_bridge.core.streaming

The streaming module is a **declarative** API for building data processing
graphs. You describe the graph by chaining operations, and a backend decides
how to execute it.

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

Mental model
------------

**Building a graph runs nothing.** Every method on an ``Op`` returns a new node
connected to the previous one. Inside a protocol, ``steps()`` returns the last
node of the graph.

**Items flow through stages.** An *item* is whatever travels between nodes,
usually a ``Set`` batch. Every operation is a *stage*. Stages run
concurrently, so while one stage processes batch *n*, the stage before it can
already process batch *n + 1*.

**FLUSH ends the stream.** When the input is exhausted, a ``FLUSH`` signal
travels through the graph. Stateful operations (``chunk``, ``collect``, ...)
react to it by emitting what they still hold. Sinks finalize their output. A
flush acts as a barrier: it completes only after all earlier items have passed
through every downstream stage.

**Two layers.** The operations below are the *surface API*. They are
*lowered* into five intermediate representation (IR) primitives, and a backend
only needs to implement those primitives. See :doc:`../developer_guide`.

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
   * - ``combine_latest(other)``
     - (N, M) → N
     - Pair every item with the latest item of another stream.
   * - ``group_by(key, pipeline)``
     - N → N
     - Run a separate sub-pipeline for every key.
   * - ``sink(callback)`` / ``write_to(writer)``
     - terminal
     - Consume the stream.
   * - ``checkpoint(writer)``
     - pass-through
     - Persist items without ending the stream.

Sources
-------

A ``Source`` is a named entry point. Data enters the stream through
``Pipeline.send(name=value)``:

.. code-block:: python

    particles = Source("particles")
    ...
    pipeline.send(particles=batch)

Inside a :doc:`protocol <protocols>`, every ``B.Input`` field *is* a source
named after the field, so you write ``self.particles.chunk(...)``. Sources
with the same name are the same input. Using an input in several branches
**fans out** its items to all of them.

.. warning::

   ``send`` rejects plain Python lists. Wrap the elements in a ``B.Set``, which
   serializes efficiently.

map
---

.. code-block:: python

    stream.map(func, *, cpu_only=False)

Applies ``func`` to every item and forwards the result (``map_batch`` is the
same operation). Every ``map`` runs as its own stage, so splitting work into
several maps lets the parts overlap on consecutive batches:

.. code-block:: python

    # GPU forward pass of batch n+1 overlaps with CPU post-processing of batch n.
    particles.chunk(256).map(forward).map(build_metadata)

``cpu_only=True`` runs the map without the GPUs that the protocol declared with
``@B.resources``. Use it for cheap bookkeeping maps, so they do not wait for a
GPU.

map_element
-----------

.. code-block:: python

    stream.map_element(
        func, *, workers="auto", executor="thread",
        start_method=None, chunksize=None, cpu_only=False,
    )

For every incoming collection ``col`` (anything with ``__len__``,
``__getitem__`` and ``__setitem__``, such as a ``Set``), runs
``col[i] = func(col[i])`` for all ``i`` on a worker pool and forwards the same,
modified collection. ``func`` may modify the element in place (for example a
``Set`` row view) or return a new value.

.. code-block:: python

    def preprocess(p: Particle) -> Particle:
        p.pixels = bandpass(p.pixels)
        return p

    particles.chunk(256).map_element(preprocess).map(forward)

* ``workers``: the number of workers, or ``"auto"`` for one per element, capped
  at the available CPUs.
* ``executor``: ``"thread"`` (default) or ``"process"``. Threads are safe next to
  CUDA and JAX, and they scale when ``func`` releases the GIL (NumPy, PyTorch,
  OpenCV). Processes help with pure-Python work.
* ``start_method``: the process start method (``"spawn"`` by default). Avoid
  ``"fork"`` in processes that have initialized CUDA or JAX.

chunk
-----

.. code-block:: python

    stream.chunk(n, drop_last=False)

Accumulates incoming ``Set`` items and emits sets of exactly ``n`` elements,
regardless of how the input was batched. On ``FLUSH`` the remainder is
emitted as a final, smaller set, unless ``drop_last=True``. Dropping it is
useful for JIT-compiled models that would otherwise recompile for an odd batch
size.

.. code-block:: text

    in:   [7]  [3]  [5]          chunk(4)
    out:  [4]  [4]  [4]  …flush→ [3]

Empty sets are ignored. Items that are not a ``Set`` raise ``TypeError``.

collect
-------

.. code-block:: python

    stream.collect(n)

Collects the **first** ``n`` elements of a stream of sets into one ``Set`` and
emits it once. All later items are ignored. If the stream ends before ``n``
elements arrive, the elements collected so far are emitted on ``FLUSH``.

Use it to train a model on an initial sample of the stream:

.. code-block:: python

    model = particles.collect(5_000).map(train)

flatten
-------

.. code-block:: python

    stream.flatten()

Emits every element of an incoming iterable as a separate item. Accepts
lists, tuples, generators, and a ``Collection`` (which yields its initialized
items in index order).

It rejects:

* strings, bytes and mappings, because iterating them yields characters or
  keys;
* a ``Set``, because sending its rows one by one would be far more expensive
  than sending the batch. Use ``chunk`` to change batch sizes instead.

combine_latest
--------------

.. code-block:: python

    stream.combine_latest(other)

Emits ``(item, latest)`` for every item of ``stream``, where ``latest`` is the
most recent item produced by ``other``:

* Items that arrive before ``other`` has produced anything are buffered, and
  released once it has.
* Items of ``other`` only update ``latest``. They emit nothing themselves.
* Items still buffered at ``FLUSH`` are **dropped**, because ``other`` never
  produced a value to pair them with.

Together with ``collect``, this is how a model trained on a sample is applied
to the whole stream:

.. code-block:: python

    model = particles.collect(5_000).map(train)
    labels = particles.chunk(256).combine_latest(model).map(predict)

    def predict(pair):
        batch, model = pair
        ...

``other`` must be a different stream than ``stream``.

group_by
--------

.. code-block:: python

    stream.group_by(key, pipeline, *, max_keys=None, workers=WorkersFrom.BACKEND)

Routes every item by its key into the sub-pipeline. Stateful operations inside
the sub-pipeline (``chunk``, ``collect``, ``combine_latest``) only see the
items of one key.

The keys share a few copies of the sub-pipeline, the **workers** (4 by
default). A key is assigned to the next worker when its first item arrives.
Within a worker, every stateful operation keeps a separate state per key, and
the stateless ``map`` and ``map_element`` carry the key along. Each worker
starts the stages (processes) of the sub-pipeline once, so a fan-out to many
keys doesn't start processes for each key. A long-running protocol inside the
sub-pipeline loads its process-scope resources (such as a model) once per
worker, and the keys of the worker share them. Each map call still runs as a
task of its own, so external programs and temporary files stay separate for
each call. Keys of one worker share its queues, so a slow key delays the
other keys of its worker.

* ``key``: an index or field name (``item[key]``), or a function that computes
  the key. Keys must be hashable.
* ``pipeline``: a function that receives the input stream of one key and
  returns the output stream. It may only consume that input, and every branch
  must lead to the stream it returns.
* ``max_keys``: an upper bound on the number of keys; a further key fails the
  pipeline.
* ``workers``: the number of workers the keys share. ``None`` gives every key a
  sub-pipeline of its own. By default, the backend decides
  (``RayBackend(group_by_workers=...)``, 4 unless set).

The output carries ``Keyed(key, value)`` tuples. ``unkey()`` marks the end of
the keyed region:

.. code-block:: python

    def refine(cls_stream):
        return cls_stream.map(align).collect(1_000).map(average)

    (
        classes.flatten()
        .group_by(lambda cls: cls.class_id, pipeline=refine)
        .unkey()
        .map(write_class)      # receives Keyed(key, value)
    )

Sinks and checkpoints
---------------------

A graph ends in one or more sinks:

.. code-block:: python

    stream.sink(print)                       # a callback, handy for debugging
    stream.write_to(writer)                  # a SinkWriter
    stream.checkpoint(writer).map(...)       # persist and continue

A ``SinkWriter`` is any object with two async methods:

.. code-block:: python

    class SinkWriter(Protocol):
        async def write(self, item) -> None: ...     # commit one item durably
        async def finalize(self) -> None: ...        # flush and release, on FLUSH

The Ray backend ships ``TensorStoreSinkWriter``. It persists ``Struct``,
``Set``, and ``Collection`` items to Zarr through TensorStore and follows the
:ref:`reduction semantics <reduction-semantics>` of the type system: set
columns are appended, and plain array fields are overwritten.

.. code-block:: python

    from scipion_bridge.backend.ray.sinks import TensorStoreSinkWriter

    stream.write_to(TensorStoreSinkWriter("output.zarr"))

Pipelines
---------

``Pipeline.from_sink(*sinks, backend=...)`` lowers the graph that leads to the
given sinks and compiles it with a backend. If ``backend`` is omitted, the
injected default backend is used (Ray, if it is installed).

.. list-table::
   :widths: 30 70

   * - ``send(**inputs)``
     - Push one item into each named source.
   * - ``flush()``
     - Send ``FLUSH``, drain in-flight work, and finalize the sinks. Called
       automatically when a ``with`` block exits without an exception.
   * - ``stats()``
     - Per-stage ``StageStats``.
   * - ``close()``
     - Release backend resources (Ray actors).

``StageStats`` records, per stage:

* ``items_in`` / ``items_out``: the item counts.
* ``idle_s``: time spent waiting for input. A high value means the stage is
  *starved* by its upstream.
* ``process_s``: time spent in the stage logic.
* ``blocked_s``: time spent waiting for room downstream. A high value means
  the stage is *throttled* by its downstream.
* ``emit_s``: time spent forwarding results, including serialization.

Use the stats to find the bottleneck. It is usually the stage with high
``process_s`` that has a starved stage after it and a blocked stage before it.

Backpressure and errors
^^^^^^^^^^^^^^^^^^^^^^^

Every stage has a bounded input queue (``queue_size``, default 2). When a
stage falls behind, the stages before it block, and eventually so does
``send()`` on the driver. Memory use therefore stays bounded, no matter how
fast data arrives.

When a stage fails, its exception is stored and re-raised by the next
``send()`` or ``flush()`` on the driver. The pipeline also aborts its sources,
so the failure surfaces immediately instead of only at the end of the stream.
