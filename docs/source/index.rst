Scipion Bridge
==============

**Scipion Bridge** is a Python library for writing streaming cryo-EM
protocols. A protocol written with it runs in a notebook, as a command line
program, distributed across a `Ray <https://www.ray.io>`_ cluster, or as a
native protocol of `Scipion 3 <https://scipion.i2pc.es>`_, without changes to
its code.

A protocol states *what* it computes as a small declarative graph of
operations. The library decides *how* to run that graph: how to batch the
data, how to move batches between processes and machines, how to convert data
into the files that external programs expect, and when to delete those files
again.

.. code-block:: python

    import scipion_bridge as B
    from scipion_bridge import single_particle as spa


    @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
    class FoundationModelInference(B.Protocol):
        """Embed particles with a foundation model."""

        particles: B.Input[B.Set[spa.Particle]]
        chunk_size: B.Field[int] = B.Field(default=256)
        model: B.Resource[Model] = B.Resource(builder=load_model)

        def outputs(self):
            return {"particles": B.Set[spa.FlexParticle]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._compute_latents)
                .map(self._to_output, cpu_only=True)
            )

Why the library exists
----------------------

Cryo-EM datasets have grown to tens of millions of particles, and most of the
processing applied to them is independent per particle or per batch of
particles. Distributed data processing frameworks such as Ray, Spark and Dask
solve exactly this class of problem: they schedule work, move data between
workers and recover from failures. Existing cryo-EM software, however, is
largely made of command line programs that exchange files, and protocols in
workflow systems such as Scipion process data in large, file-based steps.

Scipion Bridge connects the two worlds. It provides:

* a **columnar type system** (``Struct``, ``Set``, ``Collection``) that stores
  cryo-EM data in NumPy, Awkward Array and Apache Arrow buffers, and that knows
  how to split and combine that data;
* a **declarative streaming API** that lowers protocol graphs to a minimal
  intermediate representation, executed by an interchangeable backend;
* a **type resolution engine** that converts data between in-memory types,
  files on disk and Scipion objects by searching a graph of converter
  functions;
* **proxies** for files with automatic reference counting, and
  **shell command** declarations that make command line programs callable as
  typed Python functions;
* **backends** for standalone use, for Ray, and for Scipion 3, selected through
  dependency injection.

Each of these parts can be used without the others, but they are designed to
be combined in a protocol.

How this documentation is organized
-----------------------------------

The documentation follows the order in which most readers need it.

**Getting started**
   :doc:`/getting_started/installation` sets up an environment,
   :doc:`/getting_started/quickstart` builds and runs a first protocol in five
   steps, and :doc:`/user_guide/overview` explains the architecture and
   follows a single batch of data through it.

**Integrating a protocol**
   A step-by-step guide for scientists and developers who want to bring a
   method or an external program into the library: modelling the data,
   wrapping programs, writing the protocol, testing it, and deploying it on
   Ray or in Scipion. Start at :doc:`/integration/index`.

**Concepts**
   One page per building block, from the :doc:`array types
   </user_guide/structs/index>` to the :doc:`backends </user_guide/backends/index>`. The
   integration guide links to these pages where it needs them. The
   :doc:`/user_guide/sharp_bits` page collects behaviors that commonly
   surprise new users.

**Tutorials**
   :doc:`/tutorials/2d_classification` builds a complete, stateful streaming
   2D classification workflow. The notebook :doc:`examples </examples>` work
   with real data.

**Developer guide**
   For contributors to the library itself: the internal architecture, the
   storage engine of the array types, the streaming intermediate
   representation, the Ray and Scipion backends, and the testing setup. Start
   at :doc:`/developer/index`.

**Reference**
   The :doc:`/api` lists the public API by topic, and the generated
   :doc:`module reference </autoapi/index>` documents every module.

Which path to take
------------------

* **You want to run an existing protocol.** Read the installation page and
  the section on :doc:`running protocols on Ray </user_guide/backends/index>`.
* **You want to make a method or program available as a protocol.** Read the
  quickstart and the overview, then follow the :doc:`integration guide
  </integration/index>` from start to end. The tutorial shows the result for a
  larger example.
* **You want to change or extend the library.** Read the overview and the
  concept pages for the parts you will touch, then the :doc:`developer guide
  </developer/index>`.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   user_guide/overview

.. toctree::
   :maxdepth: 2
   :caption: Integrating a Protocol
   :hidden:

   integration/index
   integration/data_model
   integration/external_programs
   integration/writing_protocols
   integration/testing
   integration/deployment

.. toctree::
   :maxdepth: 2
   :caption: Concepts
   :hidden:

   user_guide/structs/index
   user_guide/streaming/index
   user_guide/protocols/index
   user_guide/type_resolution/index
   user_guide/proxies
   user_guide/shell_commands
   user_guide/backends/index
   user_guide/sharp_bits

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/2d_classification
   examples

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/index
   developer/dependency_injection
   developer/struct_storage
   developer/streaming_internals
   developer/ray_backend
   developer/typed_internals
   developer/pyworkflow_backend
   developer/testing
   developer/contributing

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   api
   autoapi/index
