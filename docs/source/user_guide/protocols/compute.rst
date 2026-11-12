Compute Resources
=================

.. currentmodule:: scipion_bridge

``@B.resources`` declares what every call of a protocol's ``map`` and
``map_element`` functions needs: GPUs, GPU memory, CPU cores, and whether the
resources are taken per call or held for the whole stream. The Ray backend
schedules calls by these declarations; the standalone and pyworkflow backends
ignore them. This article describes each option and how the backend
implements it.

.. contents:: On this page
   :local:
   :depth: 2

Declaring resources
-------------------

.. code-block:: python

    @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
    class Inference(B.Protocol): ...

    @B.resources(gpus=0.5, cpus=4)
    class Denoise(B.Protocol): ...

    @B.resources(min_vram=10)          # a share of a GPU with at least 10 GiB
    class SmallModel(B.Protocol): ...

The declaration applies to the maps created in the protocol's ``steps()``.
Built-in operations such as ``chunk`` need no resources and are not affected.
Maps created with ``cpu_only=True`` keep the protocol's CPU reservation but
run without its GPUs and outside its long-running executor, so bookkeeping
does not wait for a GPU.

Options
-------

``gpus``
   The number of GPUs per call. Fractions let several calls share a GPU:
   with ``gpus=0.5``, two calls run on one GPU at the same time.

``min_vram``
   The GPU memory in GiB that a call needs on each of its GPUs. The backend
   claims the smallest share of a GPU (1/8, 1/4, 1/2 or 1) that provides this
   much memory on the *smallest* GPU of the cluster, or ``gpus`` if that is
   more. Ray treats the GPUs of a node as interchangeable, so claims are
   sized for the smallest one; on larger GPUs, part of the memory stays
   unused, which the backend logs. The GPU memory is probed with
   ``nvidia-smi``; pass ``RayBackend(gpu_memory=[...])`` to set it explicitly.

``cpus``
   The number of CPU cores reserved per call. ``None`` (the default)
   reserves none: calls may use all cores of the machine, and the operating
   system shares them. The reserved count is exported to user code as
   ``SCIPION_BRIDGE_CPUS``, for example to size thread pools; ``map_element``
   with ``workers="auto"`` uses it.

``task``
   ``TaskType.EPHEMERAL`` (the default) or ``TaskType.LONG_RUNNING``,
   described below.

``replicas``
   The number of executor processes of the protocol, each holding the
   resources. The calls of the protocol's maps are spread over them, in order:
   ``@B.resources(gpus=1, task=B.TaskType.LONG_RUNNING, replicas=2)`` runs a
   model on two GPUs, without partitioning the stream in the protocol.
   ``replicas="auto"`` gives a long-running protocol one executor, plus as
   many as the GPUs that the other protocols leave hold. The count is decided
   when the pipeline is compiled. The default is
   1 for long-running protocols; ephemeral protocols with GPUs share the GPUs
   that long-running protocols leave in equal parts.

Placement
---------

When a pipeline is compiled, the Ray backend plans where the executors of
every protocol run:

* Long-running protocols get GPUs of their own where the cluster has enough:
  processes of different protocols on one GPU take turns on it, which can
  make each of them several times slower. If they do not fit otherwise, they
  share GPUs, with a warning.
* The executors of one protocol fill a GPU before the next (two executors of
  ``gpus=0.5`` share one GPU).
* Protocols get their minimum in the order of the graph: their declared
  ``replicas`` (one by default), or one executor for ephemeral protocols and
  ``replicas="auto"``. A protocol never takes the minimum of one that feeds
  it.
* Then ephemeral protocols and ``replicas="auto"`` split the GPUs that are
  left, in equal parts. A long-running protocol with ``replicas="auto"``
  grows by whole executors and leaves room for one executor of every
  ephemeral protocol beside it.

The backend enforces the plan: every long-running protocol reserves its
planned GPUs (a Ray placement group, on the planned node) from its first
executor until it releases them, so that no other process lands on them. A
GPU a protocol has to itself is reserved whole, even if its executors claim
half of it. Ephemeral protocols reserve nothing and run on the GPUs that
nobody reserved. On a cluster shared with other jobs, reserving GPUs may
wait for them (the pool logs a warning after 30 seconds); there,
``RayBackend(placement="ray")`` (``--placement ray``,
``SCIPION_STREAM_PLACEMENT=ray``) leaves placement to Ray, as before:
executors may then share a GPU although another one is free.

Protocols that do not fit the cluster at once run in **phases**, one after
another. For example, on one GPU, a long-running motion correction runs in
phase 0 and a long-running 2D classification after it in phase 1:

* A long-running protocol holds its GPUs until the stream is flushed. Its
  pool then releases them as soon as FLUSH has passed its own stages, and
  the pools of the next phase on the same GPUs start (with
  ``placement="ray"``, once every pool of the earlier phases has released
  its GPUs).
* The items for a later phase wait in front of its first stage, on disk
  (in the temporary directory of that stage's node). Within a phase,
  backpressure works as usual. The stage before the boundary therefore keeps
  up with its upstream, and the disk holds everything that arrives during the
  earlier phase: whole items, until provenance lets it keep only what
  changed.
* A later phase starts only after FLUSH, at the end of the stream. In a live
  session, the stages of a later phase wait until the session is flushed.
* A crash while items wait loses them; checkpointing will keep them.

``--explain`` and the report show the phase of every stage, and which stages
wait. Phases follow the graph: a protocol never waits for one it feeds.

A declaration no cluster state satisfies fails when the pipeline is
compiled: more GPUs or CPUs per executor than a node has, a claim above one
GPU that is not a whole number, or more replicas of a long-running protocol
than the cluster holds. The protocols of one ``group_by`` must fit together.

``--explain`` shows the plan without running the pipeline, for this machine or
another cluster (``--explain-gpus 4,4`` for two nodes of four GPUs); see
:doc:`/user_guide/streaming/reports`.

Ephemeral and long-running tasks
--------------------------------

The task type decides how resources are held, and with it in which process a
call runs.

**Ephemeral.** Resources are held only while calls need them. Calls without
GPUs run as Ray tasks, which queue for the CPUs and reuse worker processes.
Calls that claim GPUs run on a pool of executor processes, each keeping the
GPU it was started on (CUDA and JAX bind a process to the GPU it first used):
executors start when calls need them, build the protocol's process-scope
resources once, and are stopped after ``RayBackend(ephemeral_idle_s=...)``
seconds without calls (10 by default), so that other protocols can use the
GPUs while the stream pauses. This fits external programs and models that
are only needed now and then.

**Long-running.** A pool of ``replicas`` dedicated executor processes holds
the resources from the moment the pipeline is compiled until the stream is
flushed, then releases them; items that arrive after a flush start new
executors. Every executor builds the protocol's ``ResourceScope.PROCESS``
resources (for example, loads its model) before it takes its first call, so
the first items do not wait for it. All maps of the
protocol (except ``cpu_only`` ones) run on the pool, so process-scope
resources such as a model are built once per executor. A call goes to an idle
executor if there is one; each map runs one call at a time on an executor,
while the input of its next call is already being transferred, which hides the
transfer behind the computation. The results keep the order of the items.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Workload
     - Task type
     - Typical declaration
   * - In-process model on a GPU
     - Long-running
     - ``gpus=1``
   * - External GPU program
     - Ephemeral
     - ``gpus=1``
   * - Several small models per GPU
     - Either
     - ``min_vram=...``
   * - CPU library with its own threads
     - Ephemeral
     - ``cpus=N``

GPU memory limits
-----------------

A call that claims a fraction of a GPU (through ``gpus`` or ``min_vram``)
runs with environment variables that keep frameworks within their share:

* JAX/XLA preallocates 90% of the claimed share
  (``XLA_PYTHON_CLIENT_MEM_FRACTION``) instead of most of the GPU;
* TensorFlow allocates on demand (``TF_FORCE_GPU_ALLOW_GROWTH``);
* PyTorch allocates on demand already and is not limited, but the share is
  exported as ``SCIPION_BRIDGE_GPU_FRACTION``, for use with
  ``torch.cuda.set_per_process_memory_fraction``.

Other memory use is not enforced. Calls that share a GPU must stay within
their claim.

CPU threads
-----------

Without reserved cores, the backend sets ``OMP_NUM_THREADS`` to the number of
cores of the cluster in every call, so that NumPy, BLAS and PyTorch use all
cores (Ray would otherwise limit them to one thread). With reserved cores,
``SCIPION_BRIDGE_CPUS`` holds the reservation; libraries that size their
thread pools themselves should be configured from it.
