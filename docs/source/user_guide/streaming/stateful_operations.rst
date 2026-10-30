Stateful Operations
===================

.. currentmodule:: scipion_bridge.core.streaming

Maps transform one item at a time and remember nothing. Logic that depends on
earlier items, such as re-batching, sampling or pairing two streams, is
expressed with the stateful operations. The backend runs each of them in a
single process that keeps its state, processes items strictly in arrival
order, and calls its flush logic when the stream ends. This article describes
``chunk``, ``collect``, ``flatten``, ``scan``, ``combine_latest`` and ``merge``;
``group_by`` has an article of its own (:doc:`group_by`).

.. contents:: On this page
   :local:
   :depth: 2

``chunk()``
-----------

.. code-block:: python

    stream.chunk(n, drop_last=False)

``chunk`` accumulates incoming ``Set`` items and emits sets of exactly ``n``
elements, regardless of how the input was batched. When the stream ends, the
remainder is emitted as a final, smaller set, unless ``drop_last=True``.

.. code-block:: text

    in:   [7]  [3]  [5]          chunk(4)
    out:  [4]  [4]  [4]  …flush→ [3]

Incoming sets are concatenated and sliced, so the emitted sets are views or
concatenations of the input, not element-by-element copies. Empty sets are
ignored, and items that are not a ``Set`` raise ``TypeError``. ``n`` must be
positive.

``chunk`` belongs directly before the stage whose cost or correctness depends
on the batch size: a GPU model, an external program, a JIT-compiled function.
Pass ``drop_last=True`` when a smaller final batch would be harmful, for
example because a model compiled for a fixed batch size would recompile for
it. The dropped elements are then not processed at all; padding the final
batch in the map is the alternative.

``collect()``
-------------

.. code-block:: python

    stream.collect(n)

``collect`` gathers the **first** ``n`` elements of a stream of sets into one
``Set`` and emits it once, as soon as ``n`` elements have arrived. All later
items are ignored. If the stream ends before ``n`` elements arrive, the
elements collected so far are emitted when the stream is flushed.

.. code-block:: text

    in:   [3]  [3]  [3]  [3]     collect(8)
    out:            [8]          (the remaining items are ignored)

Use it to take a sample of the stream, typically to train or calibrate a
model before applying it to all of the data:

.. code-block:: python

    model = particles.collect(5_000).map(train)

``collect`` keeps the collected sets in its state until it emits, outside of
the pipeline's buffer limits; choose ``n`` with memory in mind.

``flatten()``
-------------

.. code-block:: python

    stream.flatten()

``flatten`` emits every element of an incoming iterable as a separate item.
It accepts lists, tuples, generators and a ``Collection``, which yields its
initialized items in index order:

.. code-block:: python

    def split_by_class(pair) -> list[spa.Class2D]:
        batch, model = pair
        labels = model.predict(...)
        return [spa.Class2D(particles=batch[labels == k], class_id=int(k)) for k in set(labels)]

    classes = stream.map(split_by_class).flatten()    # one Class2D per item

It rejects:

* strings, bytes and mappings, because iterating them yields characters or
  keys rather than elements;
* a ``Set``, because sending its rows one by one would be far more expensive
  than sending the batch. Use ``chunk`` to change batch sizes instead.

``flatten`` is the usual step before a ``group_by``: a map produces one item
per group, ``flatten`` emits them separately, and ``group_by`` routes them.

``scan()``
----------

.. code-block:: python

    stream.scan(func, initial=None, on_flush=None)

``scan`` keeps a running state and emits ``state = func(state, item)`` for
every item, like ``itertools.accumulate``. Without ``initial``, the first item
becomes the state and is emitted unchanged:

.. code-block:: text

    in:   c1   c2    c3              scan(merge)
    out:  c1   r12   r123            (r12 = merge(c1, c2), …)

``initial`` is a *function* that creates the initial state, not the state
itself. Inside a ``group_by``, it is called once for every key, so that the
keys never share a mutable state such as a NumPy array that ``func`` updates
in place.

With ``on_flush``, the end of the stream emits one more item,
``on_flush(state)``, computed from the last state. Use it to hand a *final*
state to stages that must not act on a partial one. It is not called for a
stream (or a ``group_by`` key) that had no items.

Use it to keep a running representative per 2D class. A ``map`` computes the
average of every chunk, and ``scan`` merges the averages, weighted by their
particle counts:

.. code-block:: python

    def merge_averages(running: ClassAverage, chunk: ClassAverage) -> ClassAverage:
        count = running.count + chunk.count
        image = (running.image * running.count + chunk.image * chunk.count) / count
        return ClassAverage(image=image, count=count)

    class_stream.chunk(2_000).map(align_and_average).scan(merge_averages)

.. warning::

   ``func`` runs in the single process of the stage, without the GPUs of the
   protocol (``@resources``), and the stage processes one item at a time. Do
   the expensive work in the ``map`` before it and keep ``func`` a cheap
   merge. The state is held by the stage, so a state that grows with every
   item, such as all particles seen so far, grows without limit.

``combine_latest()``
--------------------

.. code-block:: python

    stream.combine_latest(other)

``combine_latest`` emits ``(item, latest)`` for every item of ``stream``,
where ``latest`` is the most recent item produced by ``other``:

* items of ``stream`` that arrive before ``other`` has produced anything are
  buffered, and released once it has;
* items of ``other`` only update ``latest``; they emit nothing themselves;
* items still buffered when the stream is flushed are **dropped**, because
  ``other`` never produced a value to pair them with.

.. code-block:: text

    stream:  a     b           c     d
    other:            M1                  M2
    out:            (a,M1)(b,M1) (c,M1)(d,M1)

Together with ``collect``, it applies a model trained on a sample to the whole
stream:

.. code-block:: python

    model = particles.collect(5_000).map(train)
    labels = particles.chunk(256).combine_latest(model).map(predict)

    def predict(pair):
        batch, model = pair
        ...

``other`` must be a different stream than ``stream``. Make sure that ``other``
always produces at least one item; otherwise every item of ``stream`` is
buffered until the end and then dropped. Like ``collect``, the buffer of
``combine_latest`` is part of its state and grows with the number of items
that arrive before ``other`` has produced its first item.

``merge()``
-----------

.. code-block:: python

    stream.merge(other)

``merge`` joins two branches of a pipeline again (fan-in). It emits the items
of both streams as they arrive, unchanged. Use it for items that pass through
untouched next to items that took a detour through a batching stage:

.. code-block:: python

    kept = classes.map(keep)
    fixed = classes.map(take_rest).chunk(4_096).map(fix)
    kept.merge(fixed).map(write)

The items of one input keep their order; the items of the two inputs
interleave in arrival order, so do not rely on a particular interleaving. The
flush is forwarded once, after **both** inputs ended. ``other`` must be a
different stream than ``stream``.

Behavior at the end of the stream
---------------------------------

The flush behavior is part of each operation's contract:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - On FLUSH
   * - ``chunk``
     - Emits the remainder as a smaller set, unless ``drop_last=True``.
   * - ``collect``
     - Emits the elements collected so far, if it has not emitted yet.
   * - ``flatten``
     - Nothing; it holds no state.
   * - ``scan``
     - Nothing; every state was already emitted. With ``on_flush``, emits
       ``on_flush(state)`` first, unless no item arrived.
   * - ``combine_latest``
     - Drops items still waiting for the other stream.
   * - ``merge``
     - Nothing; forwards the flush once both inputs ended.
   * - ``group_by``
     - Flushes every key's sub-pipeline, then forwards the flush.

Each stateful operation lowers to one IR accumulator: three plain functions
for the initial state, the accumulation of one item, and the flush. They can
be called directly in tests (see :doc:`/integration/testing`).
