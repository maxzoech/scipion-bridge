.. _scipion-backend:

PyWorkflow Backend (Scipion 3)
==============================

.. currentmodule:: scipion_bridge

The pyworkflow backend turns a Scipion Bridge protocol into a native Scipion 3
protocol. The generated protocol has a form built from the declarations,
reads its inputs from other Scipion protocols, streams them through the
protocol's pipeline as they arrive, and writes Scipion output sets that
downstream protocols can consume while it is still running. This article
describes how to generate such a protocol, what the generated class does, and
what it requires of a protocol's types.

.. contents:: On this page
   :local:
   :depth: 2

Generating a Scipion protocol
-----------------------------

``convert_protocol_to_scipion3_protocol`` generates the class. Call it in a
Scipion plugin's ``protocols`` module:

.. code-block:: python

    from scipion_bridge.backend.pyworkflow import convert_protocol_to_scipion3_protocol
    from .protocol_model_inference import FoundationModelInference

    FoundationModelInference = convert_protocol_to_scipion3_protocol(
        FoundationModelInference(),
        label="embed parts",
        conda_env="foundation-models",
    )

``label`` is the name shown in Scipion, and ``conda_env`` the environment in
which commands of isolated domains run. The generated class keeps the
original ``__name__`` and ``__module__``, so Scipion's plugin discovery lists
it. Install the library with the ``pyworkflow`` extra into the Scipion
environment (see :doc:`/getting_started/installation`).

The form
--------

The form is built from the protocol's declarations:

* an **Input** section with a pointer parameter per input, accepting the
  Scipion set whose items resolve into the input type (for example
  ``SetOfParticles`` for ``Set[Particle]``);
* a **Parameters** section with one widget per field: ``IntParam``,
  ``FloatParam``, ``BooleanParam``, ``StringParam`` or ``EnumParam``, with
  labels, help texts and defaults from the ``Field`` declarations;
* a **Streaming** section with the polling interval in seconds (default 10);
* the usual parallelization section of Scipion protocols.

Fields of other types raise ``NotImplementedError`` when the form is built;
inputs whose type no Scipion object resolves into raise ``TypeError``, with an
explanation of the resolver that is missing.

Execution
---------

The generated protocol is a streaming Scipion protocol. When it runs, it:

1. wires a container with Scipion implementations of the services: commands
   run through the protocol's ``runJob`` (and appear in its log), temporary
   files are created in the protocol's ``tmp`` directory, and parameter
   values are read from the form;
2. compiles the protocol's pipeline, with a sink that writes the outputs.
   Like on every backend, this runs ``validate_protocol_configuration()`` and
   ``setup()`` and seals the protocol;
3. runs ``validate_protocol_configuration()`` again as its first step, so
   that the validation appears among the steps of the Scipion protocol;
4. polls its inputs. Whenever new items appear in an input set, it inserts a
   step that converts only those items (identified by their ids) into the
   input type and sends them into the pipeline;
5. when every input set is closed and all of its items have been sent,
   flushes the pipeline and closes the output sets.

Commands of a domain created with ``isolated=True`` run after activating the
conda environment given as ``conda_env``; other commands run in Scipion's
environment. Commands of the ``XMIPP`` domain get Xmipp's environment
variables when the Xmipp plugin is installed.

A profile of the run is written to ``profiler_trace.html`` in the protocol's
``extra`` directory.

Outputs
-------

Every output dictionary that reaches the sink is converted into Scipion
objects and merged into persistent output sets, following the reduction
semantics of the types:

* ``SetOfParticles`` and ``SetOfParticlesFlex`` outputs are appended to: new
  particles are added to the protocol's output set, and its sampling rate,
  acquisition and dimensions are taken from the input or the first batch;
* ``SetOfClasses2D`` outputs are merged by class id: particles of an existing
  class are appended to it and its representative replaced; new classes are
  added;
* other Scipion sets replace the output, and other objects are defined as
  outputs directly.

Output sets stay open while the protocol runs, so downstream streaming
protocols can consume them, and are closed at the end. Each output is linked
to the protocol's inputs as its source.

Type requirements
-----------------

The form and the conversions are derived from the type resolution graph:

* **Inputs.** Each input type needs a resolver from a Scipion object, either
  for the container type itself or for its element type. The pointer class of
  the input parameter is the Scipion set involved. If several sets qualify,
  the most general one is used, with a warning.
* **Outputs.** Each output type needs a resolver whose source is exactly the
  output type and whose target is a Scipion object. A path through an
  intermediate type does not count.

The built-in resolvers cover ``Set[Particle]`` and ``SetOfParticles`` in both
directions, ``Set[FlexParticle]`` and ``SetOfParticlesFlex`` in both
directions, and ``Collection[Class2D]`` and ``SetOfClasses2D`` in both
directions. Resolvers for new types receive a ``PyWorkflowResolutionContext``
as ``metadata``, which carries the Scipion protocol, the name of the output and
the ids of the items to convert; see :doc:`/integration/deployment` for an
example, and :doc:`/developer/pyworkflow_backend` for the internals.

Choosing chunk sizes
--------------------

Every batch handed to an external program is written to temporary files, so
the cost per batch includes this I/O. Under the pyworkflow backend, prefer
large chunks, for example 10,000 particles. With the Ray backend, protocols
that work on in-memory data can use smaller chunks, sized for the GPU. Expose
the chunk size as a field so that each environment can choose its value.
