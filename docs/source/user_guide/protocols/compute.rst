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
   The number of executor processes of a long-running protocol, each holding
   the resources (default 1). The calls of the protocol's maps are spread over
   them, in order: ``@B.resources(gpus=1, task=B.TaskType.LONG_RUNNING,
   replicas=2)`` runs a model on two GPUs, without partitioning the stream in
   the protocol.

A protocol whose declaration exceeds the cluster (more GPUs or CPUs per call
than the cluster has) fails when the pipeline is compiled. So do long-running
protocols whose executors, all replicas together, do not fit the cluster at
once: an executor holds its resources until the stream is flushed, so one
that fits only after another one released its GPUs would wait forever.

Ephemeral and long-running tasks
--------------------------------

The task type decides how resources are held, and with it in which process a
call runs.

**Ephemeral.** Every call acquires the resources for its own duration, as a
Ray task. Many calls can interleave on few resources, and calls of different
maps queue for the same GPUs. A call that claims a GPU runs in a **fresh
worker process**: CUDA and JAX bind a process to the GPU it first used, so a
reused process would keep running on the GPU of its first call. As a
consequence, process-scope resources are built again for every GPU call. This
fits external programs, which hold the GPU only while they run; it does not
fit in-process models, which would be loaded for every batch. CPU calls run
in reused worker processes, which keep their resources.

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
