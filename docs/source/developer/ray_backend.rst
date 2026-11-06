Ray Backend Internals
=====================

``backend/ray`` compiles the streaming IR into Ray actors and tasks and runs
it with the guarantees of the :doc:`execution contract
<streaming_internals>`. This article follows a pipeline from compilation to
shutdown: how nodes become actors and routes, how a stage is structured, how
map calls are scheduled, how flushes and errors propagate, how ``group_by``
runs its children, and how buffering, spilling and profiling are
implemented. The user-facing options are described in
:doc:`/user_guide/backends/ray`.

.. contents:: On this page
   :local:
   :depth: 2

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Contents
   * - ``backend.py``
     - ``RayBackend``, ``RayCompiledPipeline``, the stage actors, map
       execution, compute groups, ``group_by`` routing
   * - ``mailbox.py``
     - ``Mailbox``: the bounded, spilling buffer in front of every stage
   * - ``profiling.py``
     - ``Recorder``, ``ProfileCollector``: the live log and trace file
   * - ``resource_provider.py``
     - ``RayResourceProvider`` and the cluster-wide coordinator of shared
       resources
   * - ``container.py``
     - ``RayContainer`` and ``configure_ray_env``
   * - ``ray_protocol_runner.py``
     - ``RayPipelineRunner``: input resolution, CLI, statistics output
   * - ``sinks/``
     - ``TensorStoreSinkWriter``

Compilation
-----------

``RayBackend.compile`` (``_instantiate``) proceeds in three steps.

1. **Discovery.** The IR graph is traversed upstream from the sinks, which
   yields the nodes in topological order. Each node gets a label
   ``<prefix><index>:<description>``, such as ``2:chunk(256)``; the prefix
   identifies nested pipelines (``group_by`` children).
2. **Actors.** Every node that is not *hosted* becomes a named actor
   (``<pipeline id>:<label>``): ``RaySourceActor``, ``RayAccumulatorActor``,
   ``RaySinkActor``, ``RayDemuxActor``, or ``RayMergeActor`` for a map with
   several inputs, all defined in ``scipion_bridge.backend.ray.backend``. A
   map with a single input is hosted: it has no actor of its
   own, and its calls are submitted by the actor of the nearest upstream stage
   (``_host_of``).
3. **Routes.** Every actor receives its *routes* with ``connect``: a
   ``_Push`` to a downstream actor on a given port, or an ``_Apply`` that runs
   a map and continues with that map's routes. Chains and branches of maps
   therefore nest inside the routes of the stage that hosts them. ``connect``
   also passes the abort targets and the number of inputs, and starts the
   stage's loops.

Compute claims are resolved during compilation: each distinct
``ComputeAssignment`` is translated into Ray task options (``_claim``,
``_ray_options``), and each long-running group gets a ``RayComputeGroup``
coordinator. A claim that exceeds the cluster fails compilation.

Anatomy of a stage
------------------

Every actor is a ``_PipelinedStage``, which decouples receiving, computing and
forwarding items so that consecutive stages run concurrently. It runs four
asyncio tasks:

.. code-block:: text

    push() ──► Mailbox ──fetch──► inbox ──compute──► outbox ──emit──► routes ──forward──► push() downstream
               (buffer_size,      (queue_size)       (queue_size)    (max_in_flight)
                spill)

``push``
   Called by upstream stages. Buffers the item in the mailbox and returns, so
   a busy stage does not stall its upstream while it has room. Once
   ``buffer_size`` items wait, ``push`` waits for room, which propagates
   backpressure upstream to the driver.

fetch task
   Reads waiting items in order into the bounded inbox, up to ``queue_size``
   items ahead of the compute task, fetching object store references or
   reading spilled items as needed.

compute task
   Processes items one at a time in arrival order (``process``, and
   ``on_flush`` at FLUSH), so stage logic and accumulator state never run
   concurrently.

emit and forward tasks
   Route results downstream without stalling the compute task. Each item takes
   one of ``max_in_flight`` slots while it passes the maps on its routes; the
   forward task pushes results on in order, so maps may finish out of order
   without reordering the stream.

Transport
^^^^^^^^^

Items enter the transport between stages in one place, ``_deliver``. Results
of maps are already in the object store; they travel as a ``Ref`` wrapper,
which Ray does not resolve, so the receiving stage fetches the data only when
it processes the item. Other items travel inline in the call, which is
cheapest for the many small items of, for example, a ``flatten``; Ray moves
large inline arguments through the object store by itself.

Map execution
-------------

Each ``_Apply`` route has a ``_MapRunner``, which runs the calls of one map
with the map's compute resources.

**Ephemeral calls** (and maps without resources) run as Ray tasks of
``_run_map_call``, which serves the pipeline's parameters in the worker
process, records the call for the profiler, and calls the function. Ray
queues the calls of all maps on the cluster's resources. Calls that claim
GPUs use a task definition with ``max_calls=1``: CUDA and JAX bind a process
to the GPU it first used, and Ray sets ``CUDA_VISIBLE_DEVICES`` per task, so a
reused worker would keep running on the GPU of its first call. Ray only
disables reuse automatically for GPUs requested in the decorator, not through
``.options()``, hence the separate definition. CPU calls reuse worker
processes.

**Long-running calls** run on a ``RayComputeExecutor``, a threaded actor that
holds the group's resources. The ``RayComputeGroup`` coordinator creates it on
the first ``acquire`` and kills it once every map of the group has released
it at FLUSH; items arriving after a flush create a new executor. Every map has
two calls in flight on the executor, but a per-map lock runs its function one
call at a time; Ray deserializes the argument of the next call and serializes
the result of the previous one meanwhile.

Task options
^^^^^^^^^^^^

``_ray_options`` builds the options of a call:

* ``num_gpus`` from ``gpu_claim``: the larger of ``gpus`` and the share of
  a GPU holding ``min_vram`` on the smallest GPU of the cluster, probed once
  per process with ``nvidia-smi`` on every GPU node;
* ``num_cpus`` from ``cpus``, or 0 with ``OMP_NUM_THREADS`` set to the cluster's
  CPU count, so that unreserved calls are not limited to one thread;
* environment variables limiting framework memory for fractional GPUs
  (``gpu_memory_env``) and switching off NumPy's hugepage hint
  (``numpy_hugepage_env``).

Flush
-----

FLUSH travels through the same path as items, as a barrier. A stage that
receives FLUSH waits for its preceding items to be processed, runs
``on_flush`` (for an accumulator, ``flush_fn``; for a sink, ``finalize``;
for a ``group_by``, the flush of every child), forwards the emissions, and
then forwards the FLUSH. A stage with several inputs counts FLUSH signals and
forwards one once all inputs have flushed. Each ``push(FLUSH)`` returns only
after the flush has propagated through the stage and everything downstream,
so ``RayCompiledPipeline.flush`` on the driver returns when every sink is
finalized. Long-running map runners release their compute group at FLUSH.

Errors
------

An exception in a stage, or in a map on its routes, is stored, and every
subsequent ``push`` to that stage raises it, so the error propagates upstream
to the driver. The failing stage also aborts its abort targets: the sources
of the pipeline and, for a ``group_by`` child, the sources of its parent.
Without this, the driver would only see the failure at the final flush, since a
failed stage may never receive another item (for example, after a
``collect``). Errors are wrapped in ``_Failure`` when passed to other actors,
because Ray treats a ``RayTaskError`` argument as a failed input.

``group_by()``
--------------

``RayDemuxActor`` routes items into child pipelines, compiled with a nested
``RayBackend`` that inherits the buffering, profiling and parameter settings:

* With ``workers=None``, every key gets a child compiled from
  ``clone_ir(template)``. With ``workers=n``, keys are assigned round-robin to
  up to ``n`` children compiled from ``share_keys(template)``, and items are
  sent as ``Keyed(key, item)``.
* Children are compiled asynchronously (``compile_async``), so the router keeps
  routing items of other keys while a child starts. Items for a starting child
  wait in its ``_Child`` queue, bounded by ``buffer_size``.
* Each child ends in a sink whose writer hands results back to the router's
  ``emit``, which puts them into the router's outbox as ``Keyed(key, result)``.
* On FLUSH, the router flushes every child after the items queued for it; a
  child returns from its flush only after its sink has handed back all
  results, so they precede the FLUSH downstream.
* Children abort the router's abort targets on failure, and are terminated
  together with the router.

Mailboxes and spilling
----------------------

The ``Mailbox`` holds the items waiting for a stage. Up to the spill threshold,
it keeps them in memory, as received or as object store references. Beyond
it, items are handed to the stage's ``SpillStore`` and no reference is kept,
which frees their memory in the object store. When the fetch task reaches a
spilled item, it reads it back and discards it from the store. The mailbox
makes no Ray calls itself; awaiting a reference in an actor fetches it.

The default store, ``PickleSpillStore``, pickles each item into a file in a
directory per stage under a temporary root that the compiled pipeline removes
on ``close``. ``SpillStore`` is an abstract interface (``put``, ``get``,
``discard``) so that a store aware of the data's provenance can persist only
what changed.

Shared resources
----------------

``RayResourceProvider`` caches process-scope resources per worker process.
Shared resources are built once per cluster by a detached, named
``RayResourceCoordinator`` actor (``backend/ray/resource_provider.py``), which runs the builder as a task holding 35%
of the cluster's CPU cores and keeps the resulting object reference. Every
worker fetches the same object.

Profiling
---------

Every process of a profiled pipeline records events through a ``Recorder``:
spans (with duration) and instants, each on a *lane*. Events are logged
immediately on ``scipion_bridge.profile`` and sent in batches, at least once
per second, to a ``ProfileCollector`` actor, which appends them to the trace
file in the Trace Event Format. The format tolerates a missing closing
bracket, which is why the file can be opened while the pipeline runs. Stages
flush their recorders at FLUSH, and task workers before their task returns,
since a killed or reused worker would otherwise lose its events. Without
profiling, a ``NullRecorder`` makes the instrumentation free.

Process environment
-------------------

``RayBackend`` switches off NumPy's transparent hugepage hint in the driver
and passes ``NUMPY_MADVISE_HUGEPAGE`` to every actor and task (see
:doc:`/user_guide/sharp_bits`). When it starts a local Ray instance, it sets
the workers' ``PYTHONPATH`` to the working directory, ``src``, the
directories under ``tests`` and the driver's ``sys.path``, so that modules
defining structs and protocols can be imported by the workers.

Testing the backend
-------------------

Code that runs inside actors and tasks is not measured by line coverage,
because Ray stops actors with ``ray.kill``, which discards their coverage
data. Logic of the backend should therefore be factored into functions that
can be tested in-process, with Ray tests reserved for behavior across actors:
flush ordering, backpressure, resource scheduling. Every actor of a test
pipeline starts a worker process that imports the library, so the number of
stages dominates a Ray test's runtime. See :doc:`testing`.
