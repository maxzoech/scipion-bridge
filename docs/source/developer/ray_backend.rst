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

**Tasks.** Calls without GPUs of ephemeral protocols, and of maps without
resources, run as Ray tasks of ``_run_map_call``, which serves the pipeline's
parameters in the worker process, records the call for the profiler, and
calls the function. Ray queues them on the cluster's CPUs and reuses the
worker processes.

**Pools.** Calls of long-running protocols, and calls claiming GPUs, run on a
pool of ``RayComputeExecutor``\ s, threaded actors that hold the claim of the
protocol for their lifetime; Ray pins each to its GPU. Calls claiming GPUs
cannot run as tasks: CUDA and JAX bind a process to the GPU it first used, so
a worker reused by tasks would keep running on the GPU of its first call.
There is one pool per protocol (compute group), created by ``_compute_groups``
of the top-level pipeline over the maps of the pipeline and of the children of
its ``group_by``\ s (``_all_maps``). The router of a ``group_by`` passes the
pools to its children's backend (``RayBackend(pools=...)``), so the children
run on them instead of starting executors of their own, and
``group_by(workers=)`` only partitions the keys.

An executor runs one call of a map *of the protocol* at a time: calls are
locked by the map's ``stage_fp``, which the copies of a map in the children of
a ``group_by`` share. So an executor runs at most as many calls at once as the
protocol has maps, however many children share it, and every map has at most
two calls in flight on an executor; Ray deserializes the argument of the next
call and serializes the result of the previous one meanwhile. Results return
in order through the forward tasks of the stage, as for tasks.

**Leases.** The ``RayComputeGroup`` coordinator keeps up to ``replicas``
executors and hands them out as leases. A map leases an executor it does not
hold yet: one nobody leases, else, while the pool is not full, a new one, for
which the lease waits, else the one with the fewest leases. Waiting for a new
executor rather than sharing a leased one matters because of the lock per
map: children sharing one executor while another starts would queue their
calls behind each other. On the map's side, ``_LeasedExecutors`` sends a call
to an idle leased executor, else to the least busy one with room, and leases
another one in the background while it has fewer than ``replicas``.

No lock is held while an executor starts: it may wait for GPUs another pool
holds, which that pool releases only if its maps can return their leases
meanwhile. Holding the lock while waiting deadlocked two pools that each held
one GPU and waited for the other's. For the same reason, a map leases in the
background (``_LeasedExecutors._grow``) and keeps calling the executors it
has while the lease waits.

**Warm-up.** An executor is handed out once it holds its resources and has
built the protocol's process-scope resources (``prepare``): the protocol's
``_prepare_process`` travels in the ``ComputeAssignment`` to the function
table, by compute group, and not with the routes, which every stage receives.
When the pipeline is compiled, the coordinators of long-running protocols
start their whole pools (``start``), in parallel to creating the stages; a
failed warm-up kills the executor and surfaces on the first lease.

**Release.** The top-level pipeline owns the pools. After its FLUSH has passed
every stage, the children of ``group_by`` included, ``flush`` releases them
(``release_all``): their executors are killed, and items arriving after a
flush start new ones. Maps return their leases at FLUSH. A lease that
outlives a FLUSH returns ``None``, and an executor that finishes starting
after its pool was released is killed (a generation counter). Ephemeral pools
also kill executors that have had no lease for ``ephemeral_idle_s``; their
maps return their leases after ``_LEASE_LINGER_S`` without calls, also when
a lease arrives late. ``describe`` reports the state of a pool without taking
its lock.

**Fit.** ``_check_pools_fit`` rejects, at compile time, long-running pools
that do not fit the cluster at once: their executors hold their resources
until FLUSH, so an executor that fits only after another pool released its
GPUs would wait forever. Ephemeral pools share, by default, the GPUs the
long-running pools leave in equal parts (``_pool_replicas``).

Function tables
^^^^^^^^^^^^^^^

A call carries neither its function nor the pipeline's parameters. Ray
pickles the arguments of every call, and a protocol's map function is a bound
method: as an argument it would carry the whole protocol on every call, and
for a class defined in ``__main__`` the class itself.

Instead, compiling a pipeline puts one ``FunctionTable``
(``backend/ray/function_table.py``) in the object store: the functions of all
its maps by label, its parameters and its profiling setup. A call carries a
``Ref`` to the table, its label and its item; wrapped in ``Ref``, Ray does not
resolve the table for every call. ``_run_map_call`` and
``RayComputeExecutor.call`` look the table up in the process's
``TableCache``, which unpacks it once and keeps the tables of the last few
pipelines, since task workers are pooled and outlive pipelines. Pickle stores
an object once per payload, so the methods of a protocol in one table share
one instance of it. ``RayCompiledPipeline`` holds the table while it runs and
drops it on ``close``.

A copy of the functions per process instead of per call is correct only
because protocols are sealed once their pipeline is built (see
:doc:`/user_guide/protocols/declarations`). A GPU call runs in a process of
its own and therefore unpacks the table every time.

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
