Sources and Maps
================

.. currentmodule:: scipion_bridge.core.streaming

Every graph starts at one or more sources, and most of its work happens in
maps. This article covers both: how data enters a graph, and the two
stateless operations that transform it, ``map`` and ``map_element``.

.. contents:: On this page
   :local:
   :depth: 2

Sources
-------

A ``Source`` is a named entry point of a graph. Data enters it through
``Pipeline.send(name=value)``:

.. code-block:: python

    particles = Source("particles")
    out = particles.map(process).sink(save)

    with Pipeline.from_sink(out) as pipeline:
        pipeline.send(particles=batch)

Inside a :doc:`protocol </user_guide/protocols/index>`, every ``B.Input``
field *is* a source named after the field, so graphs start with
``self.particles``.

Sources with the same name are the same input, even if they are different
objects. Using an input in several branches therefore **fans out** its items:
every branch receives every item.

.. code-block:: python

    sample = self.particles.collect(5_000).map(train)       # branch 1
    batches = self.particles.chunk(256)                     # branch 2
    # both branches receive every batch sent to "particles"

``send`` rejects plain Python lists. Wrap elements in a ``B.Set``, which
serializes as a few contiguous buffers instead of object by object.

map
---

.. code-block:: python

    stream.map(func, *, cpu_only=False)

``map`` applies ``func`` to every item and forwards the result. ``map_batch``
is the same operation under a more explicit name. The function may return any
picklable value; the next stage receives it as its item.

Every ``map`` runs its calls independently of the other maps of the graph, so
splitting work into several maps lets the parts overlap on consecutive items:

.. code-block:: python

    # The GPU forward pass of batch n+1 overlaps with the CPU post-processing of batch n.
    particles.chunk(256).map(forward).map(build_metadata)

On Ray, each call of a map runs as a task, scheduled by the compute resources
of the protocol that created it (see :doc:`/user_guide/protocols/compute`).
``cpu_only=True`` runs the map without the protocol's GPUs and outside its
long-running executor; use it for cheap bookkeeping maps that should not wait
for a GPU.

What a map function may do
^^^^^^^^^^^^^^^^^^^^^^^^^^

Map functions are called in worker processes, possibly on other machines,
and possibly in a different process for every call. They and everything they
capture, including ``self`` for bound methods, are pickled with
``cloudpickle``. Write them as functions of their input:

* do not rely on state set by earlier calls; use the
  :doc:`stateful operations <stateful_operations>` for that;
* do not mutate captured objects in the expectation that other stages or
  the driver see the change;
* read parameters through ``Field.value`` and expensive objects through
  ``B.Resource``, which work inside workers.

map_element
-----------

.. code-block:: python

    stream.map_element(
        func, *, workers="auto", executor="thread",
        start_method=None, chunksize=None, cpu_only=False,
    )

For every incoming collection ``col`` (anything with ``__len__``,
``__getitem__`` and ``__setitem__``, such as a ``Set``, a list or a NumPy
array), ``map_element`` runs ``col[i] = func(col[i])`` for every ``i`` on a
worker pool and forwards the same, modified collection. Each index is
processed by exactly one worker. ``func`` may modify the element in place, for
example a row view of a ``Set``, or return a new value.

.. code-block:: python

    def preprocess(p: Particle) -> Particle:
        p.pixels = bandpass(p.pixels)
        return p

    particles.chunk(256).map_element(preprocess).map(forward)

Use it for per-element CPU work that does not vectorize, placed after a
``chunk`` so that each call processes a batch of known size.

Options
^^^^^^^

``workers``
   The number of workers, or ``"auto"`` for one per element, capped at the
   CPUs available to the process (the reserved cores, if the protocol
   reserved some with ``@B.resources(cpus=...)``).

``executor``
   ``"thread"`` (default) or ``"process"``. Threads are safe next to CUDA and
   JAX, and they scale when ``func`` releases the GIL, which NumPy, PyTorch
   and OpenCV do. Processes help for pure-Python work that holds the GIL.

``start_method``
   The start method of the process pool, ``"spawn"`` by default. Avoid
   ``"fork"`` in processes that have initialized CUDA or JAX: forking such a
   process corrupts the child.

``chunksize``
   The number of consecutive indices per pool task. Defaults to 1 with
   ``workers="auto"``, and otherwise to ``ceil(len(col) / (4 * workers))``,
   which gives each worker about four tasks.

The worker pools are created on the first call in each process and reused for
later calls, so their startup cost is paid once per process, not per batch.
