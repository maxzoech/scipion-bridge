Overview
========

This page explains why Scipion Bridge exists, which design goals shape it, and
how its layers fit together. The rest of the user guide covers each layer in
detail.

Motivation
----------

Many cryo-EM workflows are *embarrassingly parallel*. Extracting, embedding,
or classifying a particle does not depend on the other particles in the
dataset. Datasets keep growing: a single 2D classification can involve tens of
millions of particles. Scipion Bridge treats cryo-EM processing as a big data
problem and builds on the tools of that field.

The library aims to make three things easy:

**Streaming.**
   A protocol states the operations it applies to the data. Tracking which
   items have been processed, scheduling work as new data arrives, and
   finalizing outputs are handled by the library.

**Moving data efficiently.**
   Batches travel between stages as in-memory, columnar buffers. Files are
   only created when an external program needs them.

**Reusing data processing frameworks.**
   Frameworks such as Ray, Apache Spark, and Dask provide scheduling, data
   movement, and fault tolerance. Scipion Bridge uses them as execution
   backends.

The MapReduce model
^^^^^^^^^^^^^^^^^^^

Distributed data processing frameworks are built on the *MapReduce* model:

* **Map** applies a transformation to a batch of data on a single worker.
* **Reduce / shuffle** moves and combines batches between workers.

.. code-block:: text

                    ┌──► Map() ──┐
    Big data ───────┼──► Map() ──┼──► Reduce() ──► Output
                    └──► Map() ──┘

For this model to work, the scheduler must know how to *split* data into
batches and how to *combine* results again. Scipion Bridge builds these
semantics into its :doc:`type system <structs>`. A ``Set`` can be sliced into
subsets, and every container type has a defined reduction.

As a proof of concept, extracting 1.25 million particles from CryoPPP for
neural network pre-training with PySpark on AWS took about 15 minutes of wall
time.

Design goals
------------

**Declarative.**
   A protocol describes *what* it computes as a graph of operations. Scheduling,
   batching, and resource allocation are left to the backend.

**Efficient data movement.**
   Data moves between stages as in-memory, columnar Arrow buffers. Files are
   only written when an external program needs them, and they are deleted
   automatically afterwards.

**Leverage existing frameworks.**
   The streaming layer compiles to a minimal intermediate representation (IR).
   A backend only implements that IR: Ray today, other engines later.

**Pythonic.**
   Protocols are plain Python classes, shell commands look like Python
   functions, and data is accessible through the Python array API, so you can
   use NumPy, PyTorch, or JAX directly.

Architecture
------------

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

1. **Protocols** (:doc:`protocols`) declare inputs, parameters, outputs, and a
   ``steps()`` graph.
2. **Array types** (:doc:`structs`) describe the data flowing through the
   graph: ``Struct`` for one element, ``Set`` for a column-wise batch,
   ``Collection`` for a fixed number of row-wise items.
3. The **streaming API** (:doc:`streaming`) offers high-level operations. They
   are *lowered* to five IR primitives, which a backend compiles and runs.
4. **Type resolution** (:doc:`type_resolution`) converts data between
   representations, for example from a ``Set[Particle]`` to a Scipion
   ``SetOfParticles``, or to a STAR file on disk.
5. **Proxies** (:doc:`proxies`) wrap the files that external programs read
   and write. A reference counter deletes them when they are no longer used.
6. **Shell commands** (:doc:`shell_commands`) run external programs through
   an injected execution provider.
7. **Backends** (:doc:`backends`) are selected through dependency injection:
   standalone, Ray, or pyworkflow (Scipion 3).

Interoperability with Scipion
-----------------------------

Scipion Bridge protocols can be used together with existing Scipion
protocols in two ways:

**PyWorkflow backend.**
   A Scipion Bridge protocol is converted into a native Scipion 3 protocol
   with a generated GUI form (see :ref:`scipion-backend`). The same class can
   then run both on Ray and in Scipion. Choose parameter values such as the
   chunk size for the environment the protocol runs in.

**Scipion in a container.**
   Scipion protocols run inside containers, orchestrated by the Ray
   pipeline. Streaming protocols use the full Ray pipeline, and the array type
   system was designed with containerized protocols in mind. Existing Scipion
   containers can be reused.

Performance
-----------

With the Ray backend, a protocol can call a PyTorch module directly on
in-memory batches. In the prototype, embedding 100,000 particles with a
foundation model ran at 76 particles/s in FP32 and 300 particles/s in FP16,
matching the throughput of the PyTorch model on its own.
