Protocols
=========

.. currentmodule:: scipion_bridge

A **protocol** is the unit of work that users run. It bundles a
:doc:`streaming graph </user_guide/streaming/index>` with explicit
declarations of its inputs, parameters, resources, compute requirements and
outputs. The declarations are what allows the same protocol class to run as a
Ray job, as a command line program, and as a native Scipion 3 protocol with a
generated form: each backend reads them to build its interface, to convert
data, and to validate results.

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


    @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
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
   pipeline is compiled. Treat it like the *definition* of a model's forward
   pass, not like a loop over the data.

``outputs(self) -> dict[str, type]``
   Declares the outputs by name and type. The last stage of ``steps()`` must
   return a ``dict`` with these keys, and every result is validated against
   the declaration.

Everything else is declared with annotated class attributes, and an optional
``@B.resources`` decorator declares the compute requirements.

The life of a protocol
----------------------

A protocol instance passes through the same phases on every backend:

1. **Declaration.** When the class is defined, the library reads its
   annotations and sorts them into inputs, fields, resources and state, in
   declaration order. Unannotated attributes raise ``TypeError`` here.
2. **Configuration.** The backend provides parameter values: from
   ``RayPipelineRunner(parameters=...)``, from command line arguments, or
   from the Scipion form.
3. **Compilation.** The backend calls ``get_pipeline()``, which calls
   ``steps()``, assigns the protocol's compute resources to its maps, and
   appends the validation of outputs. The graph is compiled into the
   backend's processes.
4. **Execution.** The instance is pickled into the worker processes together
   with the bound methods used in ``map``. Stage functions read parameters
   through ``.value`` and resources through their attributes, both served by
   the backend in every worker.
5. **Completion.** The stream is flushed, the outputs finalized, and the
   backend's processes released.

Articles in this section
------------------------

.. toctree::
   :maxdepth: 1

   declarations
   compute
   composition

* :doc:`declarations` covers inputs, fields, resources, state, outputs and
  their validation, and the optional hooks.
* :doc:`compute` covers ``@B.resources``: GPUs, GPU memory, CPU cores, and
  the difference between ephemeral and long-running tasks.
* :doc:`composition` covers chaining protocols into one pipeline and
  introspecting their declarations.

Running protocols is the subject of :doc:`/user_guide/backends/index`, and
the complete workflow of writing one is described in
:doc:`/integration/index`.
