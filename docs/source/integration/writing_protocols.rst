Step 3: Writing the Protocol
============================

.. currentmodule:: scipion_bridge

With the data types defined and the program wrapped, the protocol class
connects them: it declares what the protocol consumes, which parameters and
resources it needs, and what it produces, and it builds the streaming graph
that does the work. This page develops the ``Denoise`` protocol of the running
example in both styles, in-process and with the external program, and
explains the design decisions along the way.

The reference for every declaration is :doc:`/user_guide/protocols/index`; the
operations available in ``steps()`` are described in
:doc:`/user_guide/streaming/index`.

.. contents:: On this page
   :local:
   :depth: 2

The in-process protocol
-----------------------

The complete protocol for the PyTorch version of the denoiser:

.. code-block:: python

    # scipion_denoise/protocols.py
    import numpy as np
    import torch

    import cryodenoise
    import scipion_bridge as B
    from scipion_bridge import single_particle as spa

    from .types import DenoisedParticle


    def load_denoiser(protocol: "Denoise") -> torch.nn.Module:
        model = cryodenoise.load_model(protocol.weights.value)
        return model.eval().cuda()


    @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
    class Denoise(B.Protocol):
        """Denoise particle images and estimate their signal-to-noise ratio."""

        # Inputs: the streams the protocol consumes.
        particles: B.Input[B.Set[spa.Particle]] = B.Input(
            label="Particles",
            help="Particles to denoise.",
        )

        # Fields: user-facing parameters.
        weights: B.Field[str] = B.Field(
            default="/opt/models/cryodenoise.pt",
            label="Model weights",
        )
        chunk_size: B.Field[int] = B.Field(default=256, label="Batch size")

        # Resources: expensive, read-only state, built once per process.
        model: B.Resource[torch.nn.Module] = B.Resource(builder=load_denoiser)

        def outputs(self):
            return {"particles": B.Set[DenoisedParticle]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._denoise)
                .map(self._to_output, cpu_only=True)
            )

        def _denoise(self, batch: B.Set[spa.Particle]):
            images = torch.as_tensor(np.asarray(batch["pixels"])).cuda()
            with torch.no_grad():
                denoised, snr = self.model(images)
            return batch, denoised.cpu().numpy(), snr.cpu().numpy()

        def _to_output(self, result):
            batch, pixels, snr = result
            out = B.Set[DenoisedParticle](capacity=len(batch))
            out["pixels"] = pixels
            out["sampling_rate"] = batch["sampling_rate"]
            out["snr"] = snr.reshape(-1, 1)
            return {"particles": out}

The rest of this page explains each part.

Declarations
------------

Everything a backend needs to know about a protocol, without running it, is
declared as annotated class attributes. Backends read these declarations to
build command line parsers and Scipion forms, to resolve inputs, and to serve
parameter values.

Inputs
^^^^^^

An ``Input[T]`` declares a stream of items of type ``T``. On an instance,
``self.particles`` is a ``Source`` named after the attribute, which is where
the graph starts. Using the same input several times refers to the same
source; its items fan out to every branch.

The input type decides where data can come from: the Ray runner needs a
resolution path from ``Path`` (see :doc:`external_programs`), and the Scipion
backend needs a resolver from a Scipion object (see :doc:`deployment`).

Fields
^^^^^^

A ``Field[T]`` declares a parameter. Its value is read with ``.value``, on the
driver inside ``steps()`` as well as inside every stage, because each backend
serves the values to its worker processes. Supported types are ``int``,
``float``, ``str``, ``bool`` and ``Enum`` subclasses; they map to Scipion form
widgets and command line arguments.

A field declared by its annotation alone (``threshold: B.Field[float]``) is
required. With ``B.Field(...)``, ``optional`` defaults to ``True`` when no
default is given and to ``False`` otherwise; a parameter without a value
falls back to its default.

Read field values only inside ``steps()`` and inside stage functions, never at
import time or in ``__init__``: the values are provided by the backend when
the pipeline is compiled.

Resources
^^^^^^^^^

A ``Resource[T]`` is read-only state that is expensive to create, such as
model weights. Its ``builder`` receives the protocol instance, so it can read
fields, and it runs lazily, the first time a stage reads the attribute. The
result is cached per worker process (``ResourceScope.PROCESS``, the default)
or once for the whole Ray cluster (``ResourceScope.SHARED``, for large
immutable data such as a reference volume).

Loading a model in ``__init__`` or at module level instead would load it on the
driver, and then pickle it into every worker. A resource is built where it is
used, and only there.

State
^^^^^

Any other annotated attribute is plain instance state. The protocol instance
is pickled into the worker processes together with the bound methods passed to
``map``, so state must be picklable, and changes made to it in one stage are
not visible in other stages or on the driver.

Outputs
^^^^^^^

``outputs()`` declares each output by name and type. The last stage of
``steps()`` must return a dictionary with these keys, and every returned
dictionary is validated: a value whose type is not exactly the declared type
raises ``ValidationError``.

Designing the graph
-------------------

``steps()`` is called once, when the pipeline is compiled, and returns the
last node of a graph of operations. It does not process data. Per-batch logic
goes into the functions passed to ``map``, which the backend calls for every
item. A few principles lead to graphs that run efficiently.

Control the batch size where it matters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inputs arrive in batches of whatever size the reader chose (about 32 MiB per
chunk for the Ray runner, or the size of new data in a Scipion stream).
``chunk(n)`` re-batches them into sets of exactly ``n`` elements, followed by
a smaller final set when the stream ends. Place it directly before the stage
whose performance depends on the batch size, and make ``n`` a field so that it
can be chosen per environment:

* for in-process GPU models, choose the largest batch that fits into GPU
  memory;
* for external programs, choose large batches (thousands of particles), so
  that the cost of writing files and starting the program is amortized.

Pass ``drop_last=True`` if the model is compiled for a fixed batch size and
the final, smaller batch would trigger a recompilation.

Split stages along resource boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``map`` runs its calls independently of the other maps, and consecutive
maps work on different batches at the same time. Splitting a computation into
several maps therefore lets its parts overlap. The protocol above separates
the GPU forward pass (``_denoise``) from building the output set
(``_to_output``): while the GPU processes batch *n + 1*, the CPU assembles the
output of batch *n*.

``cpu_only=True`` lets a map run without the protocol's GPUs and outside its
long-running executor, so cheap bookkeeping does not wait for the GPU.

Per-element CPU work, such as filtering every image of a batch, belongs in
``map_element``, which runs a function on every element of a batch on a thread
pool:

.. code-block:: python

    self.particles.chunk(256).map_element(self._bandpass).map(self._denoise)

Use the stateful operations for stateful logic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage functions are stateless: they may run in any worker process, and
instance state does not carry over between calls. Logic that depends on
earlier items is expressed with the stateful operations, which the backend
runs in a single process per stage and flushes at the end of the stream. Three
patterns cover most needs:

**Sample, then apply.** Train or calibrate on the first ``n`` elements, then
apply the result to the whole stream:

.. code-block:: python

    calibration = self.particles.collect(5_000).map(self._calibrate)
    return (
        self.particles.chunk(256)
        .combine_latest(calibration)
        .map(self._apply)               # receives (batch, calibration)
    )

**Per group.** Process the elements of each class, micrograph or tilt series
separately, with state per group:

.. code-block:: python

    return (
        self.classes.flatten()
        .group_by(lambda cls: cls.class_id, pipeline=self._refine_class)
        .unkey()
        .map(self._to_output)           # receives Keyed(class_id, result)
    )

**Several inputs.** Pair every item of one input with the latest item of
another, for example particles with the current reference volume, using
``combine_latest``.

The :doc:`/tutorials/2d_classification` combines all three patterns.

Everything must lead to the returned node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Backends compile only the nodes upstream of the node that ``steps()``
returns. A branch that does not lead to it, such as a ``map`` whose result is
never used or a sink attached with ``checkpoint``, is not compiled and does
not run. Persist intermediate results by returning them as additional outputs
instead.

Validate parameters in ``steps()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since ``steps()`` runs on the driver before any data flows, on every backend,
it is the place to reject invalid parameter combinations early:

.. code-block:: python

    def steps(self) -> B.Op:
        if self.chunk_size.value <= 0:
            raise ValueError("The batch size must be positive.")
        ...

The ``setup()`` and ``validate_protocol_configuration()`` hooks of
``Protocol`` run on every backend when the pipeline is built, before
``steps()``. Afterwards the protocol is sealed: see
:ref:`where-state-belongs` for where state goes instead.

Instructions for coding agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coding agents that run into ``B.SealedProtocolError`` tend to look for a way
around it. Copy this block into the ``AGENTS.md`` or ``CLAUDE.md`` of a
repository with protocols:

.. code-block:: markdown

    ## scipion-bridge protocols are sealed
    A protocol is sealed once `get_pipeline()` has run `setup()` and
    `steps()`; afterwards its attributes cannot change
    (`B.SealedProtocolError`). The functions of its stages run as copies of
    the protocol in other processes, so state kept in a stage would differ
    between the copies.
    - Never write to `self` in a function passed to `map`, `map_element` or
      `scan`, and never bypass the seal (`object.__setattr__`, `__dict__`,
      class attributes, module globals, mutable objects held by the protocol).
    - State across items (counts, running means, the best item so far):
      a `.scan(func, initial=..., on_flush=...)` stage.
    - Objects that are expensive to build (models, compiled programs):
      `B.Resource(builder=...)`.
    - Configuration: set it in `__init__()`, `setup()` or `steps()`, as a
      tuple, frozenset or frozen dataclass rather than a list, dict or set.

Declaring compute requirements
------------------------------

``@B.resources`` declares what every call of the protocol's ``map`` and
``map_element`` functions needs. Only the Ray backend schedules by it, but the
declaration belongs to the protocol, so it is written once. The right
declaration depends on how the method uses the hardware:

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Workload
     - Declaration
     - Why
   * - In-process GPU model held in a ``Resource``
     - ``gpus=1, task=TaskType.LONG_RUNNING``
     - The model is loaded once and stays on the GPU.
   * - External program using a GPU
     - ``gpus=1``
     - Each call holds the GPU only while the program runs.
   * - Small model that can share a GPU
     - ``min_vram=8`` (GiB), or ``gpus=0.5``
     - Several calls run on one GPU.
   * - CPU-bound library with its own threads
     - ``cpus=8``
     - Reserves cores; the count is in ``SCIPION_BRIDGE_CPUS``.
   * - Bookkeeping next to a GPU stage
     - ``cpu_only=True`` on that map
     - Runs without waiting for the GPU.

The distinction between the two task types matters most for GPU work:

* With ``TaskType.EPHEMERAL`` (the default), every call acquires the
  resources for its own duration. A call that claims a GPU runs in a fresh
  worker process, because CUDA binds a process to the GPU it first used.
  Process-scope resources are therefore built again on every such call, which
  is right for external programs but wasteful for an in-process model.
* With ``TaskType.LONG_RUNNING``, a dedicated executor process holds the
  resources from the first call until the stream is flushed. All maps of the
  protocol run on it, and resources such as the model are built once. Each map
  runs one call at a time on the executor, while the input of its next call is
  already being transferred.

The external program variant
----------------------------

With the command line version of the denoiser, the protocol keeps the same
declarations and graph shape. The GPU stage calls the wrapped program from
:doc:`external_programs`, the model resource disappears, and the compute
declaration becomes ephemeral:

.. code-block:: python

    from .programs import denoise_batch


    @B.resources(gpus=1)
    class DenoiseExternal(B.Protocol):
        """Denoise particle images with the cryodenoise program."""

        particles: B.Input[B.Set[spa.Particle]] = B.Input(label="Particles")

        weights: B.Field[str] = B.Field(default="/opt/models/cryodenoise.pt")
        chunk_size: B.Field[int] = B.Field(default=5_000, label="Batch size")

        def outputs(self):
            return {"particles": B.Set[DenoisedParticle]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._denoise)
            )

        def _denoise(self, batch: B.Set[spa.Particle]):
            return {"particles": denoise_batch(batch, self.weights.value)}

The larger default chunk size reflects the cost of writing every batch to
files and starting the program for every call.

Composing protocols
-------------------

Protocols compose with ``|``. The outputs of the first protocol feed the
inputs of the second, matched by name, by an explicit ``mapping``, or by type
if the first protocol has a single output:

.. code-block:: python

    pipeline = Denoise() | FoundationModelInference()

The result is again a protocol and runs on every backend. Integrate methods
as small protocols with one responsibility each, and compose them, rather than
writing one protocol that does everything.

Parameters of chained protocols are merged by name. ``Denoise`` and
``FoundationModelInference`` both declare a ``chunk_size`` field with
different defaults, which the chain rejects with ``ChainError``; protocols
meant to be combined should give such fields specific names, such as
``denoise_chunk_size``. See :ref:`chaining-protocols`.

Checklist
---------

* Inputs, fields and resources are declared with annotations; plain state is
  picklable.
* Expensive objects are ``Resource`` instances, not attributes set in
  ``__init__``.
* ``chunk`` sits before the stage that depends on the batch size, and its
  size is a field.
* GPU and CPU work are in separate maps, with ``cpu_only=True`` on the CPU
  ones.
* Stateful logic uses ``collect``, ``combine_latest`` and ``group_by``, not
  instance state.
* Every branch leads to the returned node, and the last stage returns a
  dictionary matching ``outputs()``.
* ``@B.resources`` matches the workload, and in-process GPU models use
  ``TaskType.LONG_RUNNING``.

Next, :doc:`testing` shows how to test the protocol before deploying it.
