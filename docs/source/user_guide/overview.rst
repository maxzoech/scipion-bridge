Overview
========

This page explains the problem Scipion Bridge addresses, the design decisions
that follow from it, and how the parts of the library fit together. It closes
by following a single batch of particles through every layer, from a file on
disk to a persisted result. The concept pages describe each layer in detail;
this page provides the context in which they make sense.

.. contents:: On this page
   :local:
   :depth: 2

The problem
-----------

Cryo-EM processing pipelines handle very large numbers of independent items.
A single 2D classification can involve tens of millions of particles, and many
steps (extraction, CTF correction, embedding with a neural network,
classification against a fixed set of references) treat each particle, or each
batch of particles, independently of the others. Such work is *embarrassingly
parallel*: it can be split across many workers without coordination, as long
as the data can be split and the results combined.

Two properties of the established software make this hard to exploit:

* **Programs exchange files.** Most cryo-EM tools are command line programs
  that read inputs from files and write outputs to files. Chaining them means
  writing intermediate files, keeping track of their paths, and deleting them
  afterwards.
* **Protocols process data in large steps.** Workflow systems such as Scipion
  run each protocol as a sequence of steps over whole datasets, or over
  large file-based batches in streaming mode. Scheduling, batching and
  resource allocation are written by hand in each protocol.

Meanwhile, the field of large-scale data processing has produced frameworks
(Ray, Apache Spark, Dask) that schedule work across clusters, move data
between workers through shared memory, and recover from failures. Scipion
Bridge treats cryo-EM processing as a data processing problem and uses these
frameworks as execution engines, while keeping existing programs and Scipion
workflows usable.

The MapReduce model
^^^^^^^^^^^^^^^^^^^

Distributed data processing frameworks are built around the *MapReduce*
model:

* **Map** applies a transformation to one batch of data on a single worker.
* **Reduce** (or *shuffle*) moves batches between workers and combines them.

.. code-block:: text

                    ┌──► Map() ──┐
    Big data ───────┼──► Map() ──┼──► Reduce() ──► Output
                    └──► Map() ──┘

For a scheduler to apply this model, it must know how to *split* the data into
batches and how to *combine* partial results. Scipion Bridge builds both into
its :doc:`type system </user_guide/structs/index>`: a ``Set`` can be sliced into independent
batches, and every container type defines how partial results are reduced
(sets are concatenated, single values are replaced). The type system is
therefore not only a convenient container, but the contract that makes
automatic batching and persistence possible.

As a proof of concept, extracting 1.25 million particles from CryoPPP for
neural network pre-training with PySpark on AWS took about 15 minutes of wall
time.

Design goals
------------

Four goals shape the API. They explain most of the decisions that a protocol
author encounters.

**Declarative.**
   A protocol describes *what* it computes as a graph of operations. When,
   where and in which batches the operations run is decided by a backend.
   This is why ``steps()`` builds a graph instead of looping over data, and
   why the same protocol can run on a laptop and on a cluster.

**Efficient data movement.**
   Batches move between stages as in-memory, columnar Arrow buffers, which Ray
   transfers through shared memory without copying. Files are written only
   when an external program needs them, and they are deleted automatically
   once no Python object refers to them.

**Reuse of existing frameworks.**
   The streaming layer compiles to a minimal intermediate representation (IR)
   of five primitives. A backend only implements those primitives; Ray is the
   first, and others can follow without changes to protocols.

**Pythonic.**
   Protocols are plain Python classes, external programs are called like
   Python functions, and data is exposed through the Python array API, so
   NumPy, PyTorch and JAX operate on it directly.

Architecture
------------

The library is organized in layers. Each layer only depends on the layers
below it, and each can be used on its own.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────┐
    │                           Protocol                           │
    ├──────────────────────────────────────────────────────────────┤
    │   Array types:   Particle   Micrograph   Atomic model   ...  │
    ├──────────────────────────────────────────────────────────────┤
    │                        Streaming API                         │
    └──────────────┬───────────────────────────────┬───────────────┘
                   │ type resolution               │ type resolution
                   ▼                               ▼ + temporary files (ARC)
    ┌──────────────────────┐   ┌───────────────────────────────────┐
    │   Scipion 3 types    │   │ Proxy types: MRC  STAR  PDB  ...  │
    │   (SetOfParticles)   │   ├───────────────────────────────────┤
    │                      │   │     Shell execution provider      │
    ├──────────────────────┤   ├───────────────────────────────────┤
    │  PyWorkflow engine   │   │ Ray scheduler / ScipionWeb server │
    └──────────────────────┘   └───────────────────────────────────┘

From top to bottom:

1. **Protocols** (:doc:`/user_guide/protocols/index`) declare inputs, parameters, resources and
   outputs, and build a streaming graph in ``steps()``. The declarations are
   what allows backends to generate command line interfaces and Scipion
   forms, and to validate results.
2. **Array types** (:doc:`/user_guide/structs/index`) describe the data flowing through the
   graph: ``Struct`` for one element, ``Set`` for a column-wise batch of
   elements, ``Collection`` for a fixed number of row-wise items.
3. The **streaming API** (:doc:`/user_guide/streaming/index`) provides operations such as
   ``map``, ``chunk``, ``collect`` and ``group_by``. They are *lowered* to IR
   primitives, which a backend compiles and runs.
4. **Type resolution** (:doc:`/user_guide/type_resolution/index`) converts data between
   representations, for example from a STAR file to a ``Set[Particle]``, or
   from a ``Set[Particle]`` to a Scipion ``SetOfParticles``.
5. **Proxies** (:doc:`proxies`) are typed handles for the files that external
   programs read and write. Automatic reference counting (ARC) deletes
   temporary files when no proxy refers to them anymore.
6. **Shell commands** (:doc:`shell_commands`) declare command line programs
   as Python functions and run them through an injected execution provider.
7. **Backends** (:doc:`/user_guide/backends/index`) supply the services that the layers above
   need (command execution, temporary files, parameter values, resources,
   pipeline compilation) through dependency injection. The standalone, Ray
   and pyworkflow (Scipion 3) backends ship with the library.

The life of a batch
-------------------

The layers are easiest to understand by following data through them. Take a
protocol that embeds particles with a foundation model, run on Ray from a STAR
file:

.. code-block:: python

    with RayPipelineRunner(FoundationModelInference(), sink=writer) as runner:
        runner.run(particles=Path("particles.star"))

1. **Compilation.** The runner calls the protocol's ``steps()`` once, on the
   driver, to obtain the graph. It lowers the graph to IR and compiles it with
   the Ray backend: stateful operations such as ``chunk`` become Ray actors,
   and every call of a ``map`` runs as a Ray task, or on a dedicated executor
   for long-running protocols, scheduled with the compute resources declared by
   ``@B.resources``.
2. **Input resolution.** Before any data flows, the runner searched the type
   resolution graph for a path from ``Path`` to the declared input type
   ``Set[Particle]``. The path goes through a ``ParticleStackProxy`` (a STAR
   file plus an MRC stack) and ends in a *slice-aware* resolver that can read
   a range of particles from the memory-mapped stack. The runner uses it to
   read the input in chunks of about 32 MiB each.
3. **Transport.** Each chunk is a ``Set[Particle]``: a set of NumPy columns,
   one per field. When it is sent to the first stage, it is serialized as an
   Arrow record batch and placed in Ray's shared-memory object store. The
   receiving process maps the buffers without copying them.
4. **Re-batching.** The ``chunk(256)`` stage concatenates and slices incoming
   sets until it has exactly 256 particles, the batch size the GPU expects.
   Slicing a set creates a view, not a copy.
5. **Computation.** The ``map(self._compute_latents)`` call runs on a worker
   holding a GPU. The first call builds the model through the protocol's
   ``Resource``; later calls reuse it. If the model is a command line program
   instead, the call passes the batch to a function declared with
   ``@B.shell_command``: type resolution writes the batch to a temporary STAR
   file and MRC stack, the program runs, and the temporary files are deleted
   as soon as the proxies referring to them are garbage collected.
6. **Validation.** The last stage returns a dictionary such as
   ``{"particles": flex_particles}``. The protocol checks it against the
   types declared in ``outputs()``.
7. **Persistence.** The sink receives the dictionary and writes each output,
   for example with a ``TensorStoreSinkWriter`` that appends the set's
   columns to Zarr arrays, which is the reduction rule for sets.
8. **Flush.** When the input is exhausted, a ``FLUSH`` signal travels through
   the graph. ``chunk`` emits its remaining particles as a final, smaller
   batch, every stage drains its in-flight work, and the sink finalizes its
   output. Only then does ``run`` return and print per-stage statistics.

At no point does the protocol code deal with paths, process management,
batching across files, or cleanup. Each of these concerns is handled by one
layer.

Execution environments
----------------------

Because every side effect goes through an injected service, a protocol does
not know where it runs. The library ships three backends:

**Standalone.**
   For scripts, notebooks and tests. Commands run with ``subprocess``,
   temporary files go to the system temporary directory, and resources are
   cached per process.

**Ray.**
   For parallel and distributed execution. Every stateful operation runs in a
   Ray actor, map calls are Ray tasks scheduled by GPU and CPU requirements,
   and batches move through the shared-memory object store with bounded
   buffers between stages.

**PyWorkflow (Scipion 3).**
   A protocol class is converted into a native Scipion 3 streaming protocol
   with a generated form. Inputs and outputs are converted from and to
   Scipion objects, commands run through the protocol's ``runJob``, and
   temporary files are created in the protocol's directory.

Interoperability with Scipion
-----------------------------

Scipion Bridge protocols and existing Scipion protocols can be combined in two
ways.

**Converting protocols to Scipion.**
   A Scipion Bridge protocol becomes a native Scipion 3 protocol with a
   generated form (see :ref:`scipion-backend`). It reads its inputs from the
   outputs of other Scipion protocols, and its outputs can be used by them in
   turn. The same class still runs on Ray outside of Scipion. Parameter
   values such as the chunk size should be chosen for the environment the
   protocol runs in.

**Running Scipion protocols in containers.**
   Existing Scipion protocols run inside containers, orchestrated by a Ray
   pipeline. The array type system was designed with this use in mind, so
   that data can be passed into and out of containerized protocols without
   loss.

Performance
-----------

With the Ray backend, a protocol can call a PyTorch module directly on
in-memory batches. In the prototype, embedding 100,000 particles with a
foundation model ran at 76 particles/s in FP32 and 300 particles/s in FP16,
matching the throughput of the PyTorch model on its own. The overhead of the
streaming layer is therefore small compared with the computation, as long as
batches are large enough to amortize the per-batch transport.

Where to go next
----------------

* To bring a method into the library, continue with
  :doc:`/integration/index`.
* To learn the building blocks one by one, start with :doc:`/user_guide/structs/index`.
* To work on the library itself, read :doc:`/developer/index`.
