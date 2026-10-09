scipion_bridge.core.environment.compute
=======================================

.. py:module:: scipion_bridge.core.environment.compute

.. autoapi-nested-parse::

   Compute resources (GPUs, CPUs) that the stages of a protocol require.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.environment.compute.logger
   scipion_bridge.core.environment.compute.GPU_FRACTIONS
   scipion_bridge.core.environment.compute.CPUS_ENV_VAR
   scipion_bridge.core.environment.compute.GPU_FRACTION_ENV_VAR
   scipion_bridge.core.environment.compute.NUMPY_HUGEPAGE_ENV_VAR


Classes
-------

.. autoapisummary::

   scipion_bridge.core.environment.compute.TaskType
   scipion_bridge.core.environment.compute.ComputeResources
   scipion_bridge.core.environment.compute.ComputeAssignment


Functions
---------

.. autoapisummary::

   scipion_bridge.core.environment.compute.gpu_memory_env
   scipion_bridge.core.environment.compute.numpy_hugepage_env
   scipion_bridge.core.environment.compute.configure_numpy_hugepages
   scipion_bridge.core.environment.compute.gpu_claim


Module Contents
---------------

.. py:data:: logger

.. py:data:: GPU_FRACTIONS

.. py:data:: CPUS_ENV_VAR
   :value: 'SCIPION_BRIDGE_CPUS'


.. py:data:: GPU_FRACTION_ENV_VAR
   :value: 'SCIPION_BRIDGE_GPU_FRACTION'


.. py:data:: NUMPY_HUGEPAGE_ENV_VAR
   :value: 'NUMPY_MADVISE_HUGEPAGE'


.. py:class:: TaskType

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   How the resources of a protocol are held while its stages run.


   .. py:attribute:: EPHEMERAL
      :value: 'ephemeral'



   .. py:attribute:: LONG_RUNNING
      :value: 'long_running'



.. py:class:: ComputeResources

   Resources required by every call of the stages of a protocol.

   Only the Ray backend schedules by these resources; the standalone and
   pyworkflow backends ignore them.

   .. attribute:: gpus

      Number of GPUs; fractions let calls share a GPU.

   .. attribute:: min_vram

      GPU memory in GiB a call needs on each of its GPUs. The
      backend claims the share of a GPU that holds it (see
      :func:`gpu_claim`), or ``gpus`` if that is more. GPU memory is
      not enforced: calls sharing a GPU must stay within their claim.

   .. attribute:: cpus

      Number of CPU cores reserved for a call. ``None`` reserves
      none: the calls may use all cores of the machine and the OS
      scheduler shares them.

   .. attribute:: task

      Whether the resources are taken per call or held across calls.


   .. py:attribute:: gpus
      :type:  float
      :value: 0



   .. py:attribute:: min_vram
      :type:  Optional[float]
      :value: None



   .. py:attribute:: cpus
      :type:  Optional[float]
      :value: None



   .. py:attribute:: task
      :type:  TaskType


.. py:function:: gpu_memory_env(num_gpus: float) -> Dict[str, str]

   Environment variables keeping a process within its share of a GPU.

   JAX (XLA) preallocates 75% of the memory of a GPU and TensorFlow all of
   it, so processes sharing a GPU would run out of memory. With a fraction
   of a GPU, XLA preallocates only (most of) that fraction, and TensorFlow
   allocates on demand. PyTorch allocates on demand already; user code can
   cap it with the fraction in :data:`GPU_FRACTION_ENV_VAR`.

   :param num_gpus: The GPU claim of the process.


.. py:function:: numpy_hugepage_env() -> Dict[str, str]

   Environment variables turning off NumPy's transparent hugepage hint.

   NumPy advises the kernel to back every array of 4 MiB or more with
   transparent hugepages. With the kernel setting ``defrag=madvise``, a page
   fault in such an array compacts physical memory synchronously to build a
   2 MiB page. On a long-running host whose memory is fragmented (by the Ray
   object store in ``/dev/shm``, the page cache, pinned CUDA memory), the
   compaction mostly fails and stalls the process each time: allocating a
   stack of particles goes from a fraction of a second to a minute.

   The hint is therefore off in every process of the Ray backend, unless the
   user set ``NUMPY_MADVISE_HUGEPAGE`` in the environment of the driver, whose
   value is passed through.


.. py:function:: configure_numpy_hugepages() -> None

   Apply :func:`numpy_hugepage_env` to NumPy in this process.

   NumPy reads ``NUMPY_MADVISE_HUGEPAGE`` only on import, so a process that
   has imported it already (the driver) switches the hint at runtime. The
   value is parsed as NumPy parses it.


.. py:function:: gpu_claim(resources: ComputeResources, gpu_memory: Optional[float]) -> float

   Number of GPUs a call requests: the larger of ``gpus`` and its memory claim.

   The memory claim is the share of a GPU holding ``min_vram``, rounded up to
   one of :data:`GPU_FRACTIONS`. ``min_vram`` applies to every GPU, so with
   ``gpus >= 1`` the count decides.

   :param resources: The declared resources.
   :param gpu_memory: Memory in GiB of the smallest GPU, or ``None`` if unknown.


.. py:class:: ComputeAssignment

   Compute resources of a stage, with the group of stages sharing them.

   Stages of the same protocol form a group; a long-running group holds its
   resources once for all of its stages.


   .. py:attribute:: resources
      :type:  ComputeResources


   .. py:attribute:: group
      :type:  str


