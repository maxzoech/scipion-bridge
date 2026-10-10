Step 5: Running and Deploying
=============================

.. currentmodule:: scipion_bridge

A tested protocol can be put to use in three ways: run from a Python script
on Ray, run as a command line program, and registered as a protocol in a
Scipion plugin. This page covers each of them, how to read and tune the
performance of a running protocol, and what the Scipion backend requires of a
protocol's types.

The backends themselves are described in :doc:`/user_guide/backends/index`.

.. contents:: On this page
   :local:
   :depth: 2

Running from Python
-------------------

``RayPipelineRunner`` compiles a protocol for Ray, streams its inputs from
files, and hands the outputs to a sink:

.. code-block:: python

    # scipion_denoise/run.py
    from pathlib import Path

    from scipion_bridge.backend.ray import RayPipelineRunner

    from scipion_denoise.protocols import Denoise
    from scipion_denoise.sinks import OutputsToZarr

    with RayPipelineRunner(
        Denoise(),
        parameters={"chunk_size": 512, "weights": "/opt/models/cryodenoise.pt"},
        sink=OutputsToZarr("denoised.zarr"),
    ) as runner:
        runner.run(particles=Path("particles.star"))

When the runner is created, it validates the parameter names and precomputes
a resolver for every input, from ``Path`` (or the type given in
``origin_types``) to the declared input type. The pipeline is compiled on the
first ``run``. ``run`` reads every input in chunks, interleaving the inputs,
sends the chunks into the pipeline, flushes it when the inputs are exhausted,
and prints the throughput and a table of per-stage statistics. Leaving the
``with`` block releases the Ray actors.

Choose the sink deliberately: **without a sink, the outputs are validated and
then discarded**. The sink receives one output dictionary per batch.
``OutputsToZarr`` is the small writer shown in
:ref:`Persisting outputs <ray-persisting-outputs>`: it hands every output to
a ``TensorStoreSinkWriter``, which writes each leaf field to a Zarr array under
``denoised.zarr/<output name>/``, appending the columns of sets and
overwriting other arrays. ``TensorStoreSinkWriter`` needs dense columns: an
output field whose elements differ in shape raises ``ValueError``.

Running as a command line program
---------------------------------

``launch_as_terminal_application`` builds an ``argparse`` parser from the
protocol's declarations and runs the protocol with the parsed arguments:

.. code-block:: python

    # scipion_denoise/run.py
    from scipion_bridge.backend.ray import RayPipelineRunner

    from scipion_denoise.protocols import Denoise
    from scipion_denoise.sinks import OutputsToZarr

    if __name__ == "__main__":
        RayPipelineRunner(
            Denoise(),
            sink=OutputsToZarr("denoised.zarr"),
        ).launch_as_terminal_application()

.. code-block:: bash

    python -m scipion_denoise.run --help
    python -m scipion_denoise.run --particles particles.star \
        --weights /opt/models/cryodenoise.pt --chunk-size 512

The protocol's docstring becomes the program description. Inputs become path
arguments; field names become kebab-case options (``chunk_size`` becomes
``--chunk-size``); ``bool`` fields become ``--name`` / ``--no-name``; ``Enum``
fields accept the member names. A field that is not optional is a required
option, even when it has a default; declare ``optional=True`` to make such an
option optional on the command line. Every program also accepts
``--profile [PATH]``.

Running on a cluster
--------------------

``RayBackend`` starts a local Ray instance if none is running. To use an
existing Ray cluster, connect to it before the runner compiles the pipeline:

.. code-block:: python

    import ray

    ray.init(address="auto")

Every node of the cluster must be able to import the protocol's package and
its dependencies, and to run its external programs. Install the package into
the same environment on all nodes, or use Ray's runtime environments to ship
it. Input files must be readable from the driver, which reads them and sends
the chunks to the cluster; temporary files of external programs are created on
the node that runs the call.

The stage actors are named ``<pipeline id>:<stage index>:<description>``,
which makes them easy to find in the Ray dashboard.

Reading performance
-------------------

After a run, the runner prints one row per stage:

.. code-block:: text

    stage                         in    out   idle_s  process_s  blocked_s  emit_s  buffered  spilled  fetch_s  spill_s
    0:source(particles)           40     40     0.00       0.01      12.30    0.40         0        0     0.00     0.00
    1:chunk(512)                  40     78     1.10       0.80       9.70    0.30         4        0     0.10     0.00
    ...

The columns show where each stage spends its time:

* ``idle_s``: waiting for input. A stage with high idle time is *starved* by
  the stages before it.
* ``process_s``: running the stage logic.
* ``blocked_s``: waiting until the maps after it accept more items. A stage
  with high blocked time is *throttled* by what comes after it.
* ``buffered``: the most items that waited for the stage at once.

The bottleneck is usually the step that blocks the stages before it and
starves the stages after it. For a GPU protocol that is normally the model:
add GPUs, increase the batch size, or use lower precision. If a CPU map is the
bottleneck instead, move its work into ``map_element`` or reserve more cores.

For a timeline of every process, profile the run with ``profile=True`` (or
``--profile``) and open the trace in https://ui.perfetto.dev. The trace is
written while the pipeline runs, so it can be inspected during a long run.
:doc:`/user_guide/backends/ray` explains the lanes of the trace, and the
buffering options that bound memory use.

Registering the protocol in Scipion
-----------------------------------

A Scipion plugin lists its protocols in its ``protocols`` module. Convert the
Scipion Bridge protocol there with ``convert_protocol_to_scipion3_protocol``:

.. code-block:: python

    # scipion-em-denoise/denoise/protocols/__init__.py
    from scipion_bridge.backend.pyworkflow import convert_protocol_to_scipion3_protocol

    from scipion_denoise.protocols import Denoise

    Denoise = convert_protocol_to_scipion3_protocol(
        Denoise(),
        label="denoise particles",
        conda_env="cryodenoise",
    )

The generated class keeps the name and module of the original protocol, so
Scipion's plugin discovery lists it under the given label. It is a streaming
Scipion protocol:

* **The form** has an *Input* section with a pointer parameter per input, a
  *Parameters* section with a widget per field (labels and help texts come from
  the declarations), and a *Streaming* section with the polling interval.
* **While running**, it polls its inputs. Every time new items appear in an
  input set, it converts only those items into the input type and sends them
  into the pipeline. When all inputs are closed and processed, it flushes the
  pipeline and closes the outputs.
* **Outputs** are converted into Scipion objects and appended to persistent
  output sets (``SetOfParticles``, ``SetOfParticlesFlex``, ``SetOfClasses2D``),
  so that downstream protocols can consume them while the protocol is still
  running.
* **Commands** run through the protocol's ``runJob`` and appear in its logs.
  Commands of an isolated domain run after activating the conda environment
  given as ``conda_env``.
* **Temporary files** are created in the protocol's ``tmp`` directory.
* The ``setup()`` and ``validate_protocol_configuration()`` hooks of the
  protocol run as the first steps.
* A profile of the protocol's run is written to ``profiler_trace.html`` in the
  protocol's ``extra`` directory.

What the Scipion backend requires
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generated form and the data conversion are derived from the type
resolution graph. Two conditions must hold:

* **Every input type needs a resolver from a Scipion object.** The pointer
  class of an input parameter is the Scipion set whose items resolve into the
  input type. For ``Set[Particle]``, the built-in resolvers select
  ``SetOfParticles``. An input type without such a resolver raises
  ``TypeError`` when the form is built, with an explanation of the missing
  resolver.
* **Every output type needs a direct resolver to a Scipion object.** The
  backend looks for a resolver whose source is exactly the output type, so a
  chain through an intermediate type is not enough.

Fields must be ``int``, ``float``, ``bool``, ``str`` or an ``Enum``; other
field types raise ``NotImplementedError``.

The built-in resolvers cover ``Set[Particle]`` and ``SetOfParticles`` in both
directions, ``Set[FlexParticle]`` and ``SetOfParticlesFlex``, and
``Collection[Class2D]`` and ``SetOfClasses2D``. The ``Denoise`` protocol
outputs ``Set[DenoisedParticle]``, which none of them accepts, so the plugin
adds a resolver. The simplest one converts to a ``Set[Particle]`` and
delegates to the built-in conversion:

.. code-block:: python

    # scipion_denoise/scipion_resolvers.py, imported by the plugin
    from typing import Optional

    from pwem.objects import SetOfParticles

    import scipion_bridge as B
    from scipion_bridge import single_particle as spa
    from scipion_bridge.backend.pyworkflow.resolvers import PyWorkflowResolutionContext

    from .types import DenoisedParticle


    @B.resolver
    def denoised_to_set_of_particles(
        value: B.Set[DenoisedParticle],
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> SetOfParticles:
        particles = B.Set[spa.Particle](capacity=len(value))
        particles["pixels"] = value["pixels"]
        particles["sampling_rate"] = value["sampling_rate"]
        return B.resolve(particles, astype=SetOfParticles, metadata=metadata)

The ``metadata`` argument carries the Scipion protocol instance and the name
of the output, which the built-in resolvers need to create output sets in the
protocol's directory. Keeping the ``snr`` score in Scipion would require
writing it as an attribute of each Scipion particle in a resolver of its own.

Choosing parameters per environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same protocol class runs on Ray and in Scipion, but the best parameter
values differ. In Scipion, new data typically arrives in streaming batches and
every batch handed to an external program is written to temporary files, so
larger chunks (thousands of particles) amortize the per-batch cost. On Ray,
protocols that work on in-memory data can use smaller chunks sized for the
GPU. Exposing the chunk size as a field lets each deployment choose.

Checklist
---------

* The run script passes a sink; outputs without a sink are discarded.
* All nodes of a cluster can import the protocol's package and run its
  programs.
* The statistics table and, if needed, a profile have been used to find the
  bottleneck.
* For Scipion: every input type has a resolver from a Scipion object, every
  output type a direct resolver to one, and the protocol is converted in the
  plugin's ``protocols`` module.

This completes the integration. For a larger, stateful example, continue with
:doc:`/tutorials/2d_classification`.
