Protocols
=========

.. currentmodule:: scipion_bridge

A **protocol** is the unit of work that users run. It bundles a
:doc:`streaming graph <streaming>` with declarations of its inputs, its
parameters, the resources it needs, and its outputs. Because these
declarations are explicit, the same protocol class can run as a Ray job, a
command line tool, or a native Scipion 3 protocol with a generated GUI form.

.. contents:: On this page
   :local:
   :depth: 2

Anatomy of a protocol
---------------------

.. code-block:: python

    from enum import Enum

    import torch
    import scipion_bridge as B
    from scipion_bridge import single_particle as spa


    class FoundationModelInference(B.Protocol):
        """Embed particles with a foundation model."""

        class ModelType(Enum):
            CRYO_IEF_SMALL = "Cryo-IEF (Base)"

        # Inputs: the streams the protocol consumes.
        particles: B.Input[B.Set[spa.Particle]] = B.Input(label="Particles")

        # Fields: user-facing parameters.
        model_type: B.Field[ModelType] = B.Field(default=ModelType.CRYO_IEF_SMALL)
        chunk_size: B.Field[int] = B.Field(default=10_000, label="Chunk size")

        # Resources: read-only state, built lazily once per process.
        model: B.Resource[torch.nn.Module] = B.Resource(builder=load_model)

        def outputs(self):
            return {"particles": B.Set[spa.FlexParticle]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._compute_latents)
                .map(self._to_output, cpu_only=True)
            )

        def _compute_latents(self, batch):
            with torch.no_grad():
                return batch, self.model(torch.as_tensor(batch["pixels"]))

        def _to_output(self, batch_and_latents):
            ...
            return {"particles": flex_particles}

A protocol subclasses ``B.Protocol`` and implements two methods:

``steps(self) -> B.Op``
   Builds and returns the streaming graph. It is called once, when the
   pipeline is compiled. Treat it like a model's ``forward`` *definition*, not a
   loop over the data.

``outputs(self) -> dict[str, type]``
   Declares the outputs by name and type. The last stage of ``steps()`` must
   return a ``dict`` with these keys. Every result is validated against the
   declaration, and a mismatch raises ``ValidationError``.

Inputs
------

.. code-block:: python

    particles: B.Input[B.Set[spa.Particle]]
    particles: B.Input[B.Set[spa.Particle]] = B.Input(label="Particles", help="...")

An ``Input[T]`` declares a stream of items of type ``T``. On an instance,
``self.particles`` is a ``Source`` named after the attribute, so you can chain
operations onto it directly. When you access the same input several times, all
accesses refer to the same source, which fans out to every branch.

``B.Input(...)`` accepts ``default``, ``optional``, ``label``, and ``help``.
``optional`` defaults to ``True`` exactly when a default is given.

Fields (parameters)
-------------------

.. code-block:: python

    chunk_size: B.Field[int]
    chunk_size: B.Field[int] = B.Field(default=256, label="Chunk size", help="...")

A ``Field[T]`` is a user-facing parameter. Read its value with ``.value``:

.. code-block:: python

    self.chunk_size.value

Values are supplied by the backend: by ``RayPipelineRunner(parameters=...)``,
by command line arguments, or by the Scipion form. Parameters without a value
fall back to their default. ``.value`` works both on the driver (inside
``steps()``) and inside every pipeline stage, because each backend serves the
parameter values to its workers.

``B.Field(...)`` accepts ``default``, ``optional``, ``label``, ``group``, and
``help``. Supported types map to form widgets and CLI arguments:

.. list-table::
   :header-rows: 1

   * - Type
     - Scipion form
     - Command line
   * - ``int``, ``float``, ``str``
     - Int / Float / String parameter
     - ``--name VALUE``
   * - ``bool``
     - Boolean parameter
     - ``--name`` / ``--no-name``
   * - ``Enum`` subclass
     - Drop-down of the enum *values*
     - ``--name MEMBER`` (choices are the member *names*)

Resources
---------

.. code-block:: python

    model: B.Resource[torch.nn.Module] = B.Resource(
        builder=load_model,
        scope=B.ResourceScope.PROCESS,
    )

A ``Resource[T]`` is **read-only state** that is expensive to create: network
weights, large lookup tables, or thread pools. ``builder`` is called with the
protocol instance the first time the attribute is read, and the result is
cached:

``ResourceScope.PROCESS`` (default)
   Built once per worker process, the first time a stage on that worker reads
   it.

``ResourceScope.SHARED``
   Built once for the **whole cluster** and placed in Ray's object store.
   Every worker fetches the same object. Use it for large, immutable data such
   as a reference volume.

Assigning to a resource raises ``AttributeError``. Resources need a backend
that provides a resource provider; the standalone and Ray backends do.

State
-----

Any other annotated class attribute is plain instance state:

.. code-block:: python

    tracker_handle: Any = None

Every class-level attribute of a protocol must be annotated. ``state = 42``
raises ``TypeError`` when the class is defined.

The instance is pickled into the workers together with the bound methods
used in ``map``, so state must be picklable and must not be mutated in
stages with the expectation that the change is visible elsewhere.

Compute resources
-----------------

``@B.resources`` declares what every call of the protocol's ``map`` and
``map_element`` functions needs. Only the Ray backend schedules by these
declarations; the standalone and pyworkflow backends ignore them.

.. code-block:: python

    @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
    class Inference(B.Protocol): ...

    @B.resources(gpus=0.5, cpus=4)
    class Denoise(B.Protocol): ...

    @B.resources(min_vram=10)          # a share of a GPU with at least 10 GiB
    class SmallModel(B.Protocol): ...

``gpus``
   The number of GPUs per call. Fractions let several calls share a GPU.

``min_vram``
   The GPU memory (GiB) a call needs on each of its GPUs. The backend claims
   the smallest share of a GPU (1/8, 1/4, 1/2 or 1) that provides this much
   memory on the *smallest* GPU of the cluster, or ``gpus`` if that is more.

A call that claims a fraction of a GPU (through ``gpus`` or ``min_vram``) runs
with its memory limited for frameworks that preallocate it: JAX/XLA may
preallocate 90% of the claimed share (``XLA_PYTHON_CLIENT_MEM_FRACTION``), and
TensorFlow allocates on demand (``TF_FORCE_GPU_ALLOW_GROWTH``). PyTorch
allocates on demand and is not limited; the share is exported as
``SCIPION_BRIDGE_GPU_FRACTION`` for
``torch.cuda.set_per_process_memory_fraction``. Other memory use is not
enforced: calls that share a GPU must stay within their claim.

``cpus``
   The number of CPU cores reserved per call. ``None`` reserves none, and the
   OS shares the cores. The reserved count is exported to user code as
   ``SCIPION_BRIDGE_CPUS``.

``task``
   * ``TaskType.EPHEMERAL`` (default): each call acquires the resources for its
     own duration, so many calls can interleave on few resources.
   * ``TaskType.LONG_RUNNING``: the resources are held across calls by a
     dedicated executor, which pays off for expensive setup such as loading a
     model onto the GPU. They are released when the stream is flushed. All
     stages of the protocol share one executor.

Maps created with ``cpu_only=True`` run without the protocol's GPUs and
outside its long-running executor, so they do not block the GPU stage.

Chaining protocols
------------------

Two protocols compose into one with ``|`` or ``pipe``:

.. code-block:: python

    pipeline = FoundationModelInference() | ClusterLatents()
    pipeline = FoundationModelInference().pipe(
        ClusterLatents(),
        mapping={"particles": "latents"},      # output of first -> input of second
    )

The result is a ``ChainedProtocol``. Its steps are merged into a single
streaming graph, and every batch emitted by the first protocol becomes one item
of the second protocol's input. Outputs are wired to inputs in this order:

1. by the explicit ``mapping``, if one is given;
2. otherwise by matching names;
3. otherwise, if the first protocol has a single output, to the single
   type-compatible input of the second protocol.

The chain's inputs are the inputs of the first protocol plus the unwired inputs
of the second. Parameters and resources of both are merged, and the outputs are
those of the second protocol.

Introspection
-------------

``protocol.configuration`` returns the bound declarations in declaration
order:

.. code-block:: python

    config = FoundationModelInference().configuration
    config.inputs["particles"]      # Input[Set[Particle]]
    config.parameters["chunk_size"] # Field[int](default=10000)
    config.resources["model"]

Backends use this to build forms and CLI parsers.

Running a protocol
------------------

A protocol does nothing on its own. A backend runs it:

.. code-block:: python

    from pathlib import Path
    from scipion_bridge.backend.ray import RayPipelineRunner

    with RayPipelineRunner(FoundationModelInference(), parameters={"chunk_size": 512}) as runner:
        runner.run(particles=Path("particles.star"))

See :doc:`backends` for the Ray runner, its command line mode, and the
conversion to Scipion 3 protocols.
