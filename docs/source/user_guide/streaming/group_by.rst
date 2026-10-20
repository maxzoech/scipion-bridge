``group_by()``
==============

.. currentmodule:: scipion_bridge.core.streaming

``group_by`` splits a stream by a key and runs a sub-pipeline separately for
the items of each key. It is the operation for any logic that is stateful
*per group*: refining each 2D class with its own particles, processing each
micrograph's particles together, or keeping a running average per tilt
series. This article explains its semantics, the way keys share processes,
and its constraints.

.. contents:: On this page
   :local:
   :depth: 2

Semantics
---------

.. code-block:: python

    stream.group_by(key, pipeline, *, max_keys=None, workers=WorkersFrom.BACKEND)

Every item is routed by its key into ``pipeline``. Stateful operations inside
the sub-pipeline (``chunk``, ``collect``, ``scan``, ``combine_latest``) see only the
items of one key, exactly as if each key had a pipeline of its own:

.. code-block:: python

    def refine(cls_stream):
        return (
            cls_stream
            .map(lambda cls: cls.particles)
            .chunk(2_000)                 # 2,000 particles of one class
            .map(align_and_average)
        )

    (
        classes.flatten()
        .group_by(lambda cls: cls.class_id, pipeline=refine)
        .unkey()
        .map(write_class)                 # receives Keyed(class_id, result)
    )

The arguments are:

``key``
   An index or field name, used as ``item[key]``, or a function that computes
   the key from an item. Keys must be hashable.

``pipeline``
   A function that receives the input stream of one key and returns the output
   stream. It is called once, when the graph is built, not once per key.

``max_keys``
   An upper bound on the number of keys. A further key fails the pipeline,
   which protects against a key function that produces more keys than
   expected.

``workers``
   The number of processes the keys share, described below.

Results
-------

The results of the sub-pipelines are emitted as ``Keyed(key, value)`` named
tuples, merged into one stream. ``unkey()`` marks the end of the keyed region
and returns the merged stream:

.. code-block:: python

    def write_class(keyed):
        class_id, cls = keyed               # or keyed.key, keyed.value
        ...

Results of different keys are interleaved in the order they are produced;
results of the same key keep their order. When the stream is flushed, every
sub-pipeline is flushed first, so all results precede the flush downstream.

How keys share processes
------------------------

Starting a pipeline has a cost: on Ray, every stage of it is a process. A
``group_by`` over 100 classes that started a pipeline per class would start
hundreds of processes. Instead, the keys share a few copies of the
sub-pipeline, the **workers**:

* A key is assigned to the next worker, in turn, when its first item arrives.
  A worker is started with its first key.
* Within a worker, every stateful operation keeps a separate state per key,
  created with the key's first item, and the stateless ``map`` and
  ``map_element`` carry the key along with each item.
* A long-running protocol inside the sub-pipeline loads its process-scope
  resources (such as a model) once per worker, and the keys of the worker
  share them.
* Each map call still runs as a call of its own, so external programs and
  their temporary files stay separate for each call.

The number of workers is set by the ``workers`` argument. By default, the
backend decides: ``RayBackend(group_by_workers=4)`` unless configured
otherwise. ``workers=None`` gives every key a sub-pipeline of its own.

The trade-off is isolation against startup cost. Keys of one worker share its
queues, so a slow key delays the other keys of its worker. On an 8-CPU node,
a 3-stage sub-pipeline with 20 keys started 15 processes instead of 63 with 4
workers, and reached its first flush in 3.9 s instead of 9.9 s; throughput
once running was the same.

Constraints
-----------

The sub-pipeline is compiled separately from the enclosing graph, which
imposes two rules:

* It may only consume its input stream. Referring to another stream of the
  enclosing graph (for example a protocol input) inside ``pipeline`` raises
  ``ValueError``.
* Every branch inside it must lead to the stream it returns. A branch that
  ends elsewhere, such as a separate sink, raises ``ValueError``.

Because stateful operations in a shared worker create their state with a
key's first item and flush only the keys they have seen, an accumulator's
flush must emit nothing for a state that never received an item. All built-in
operations satisfy this.

``group_by`` operations can be nested. An inner ``group_by`` inside a shared
worker routes by the outer key and its own, and emits
``Keyed(outer, Keyed(inner, result))``.
