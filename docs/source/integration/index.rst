Integrating a Protocol
======================

This guide describes how to make a processing method available as a Scipion
Bridge protocol: a class that runs as a streaming pipeline on Ray, as a
command line program, and as a native Scipion 3 protocol. It is written for
method developers and facility staff who want to integrate their own code or an
existing program, and it assumes that you have worked through the
:doc:`/getting_started/quickstart`.

The guide is divided into five steps, one page each. This page explains what
an integration consists of, which decisions to make before writing code, and
how to organize the result.

.. contents:: On this page
   :local:
   :depth: 2

What an integration consists of
-------------------------------

An integrated protocol is made of up to five parts. Not every protocol needs
all of them.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Part
     - Purpose
     - Covered in
   * - Data types
     - ``Struct`` classes describing the elements the protocol consumes and
       produces. Often the built-in single particle types suffice.
     - :doc:`data_model`
   * - Program wrappers and resolvers
     - ``@B.shell_command`` declarations for external programs, proxy types for
       their file formats, and resolvers converting between those files and
       the data types.
     - :doc:`external_programs`
   * - The protocol class
     - Inputs, parameters, resources, compute requirements, outputs, and the
       streaming graph in ``steps()``.
     - :doc:`writing_protocols`
   * - Tests
     - In-process tests of the stage functions, resolvers and commands, plus
       a small end-to-end test on Ray.
     - :doc:`testing`
   * - Deployment
     - A command line entry point, a Ray launch script, and the registration
       of the protocol in a Scipion plugin.
     - :doc:`deployment`

The first decision: in-process or external program
---------------------------------------------------

The most important decision is how the method's computation is called. The
library supports two styles, and the rest of the integration follows from the
choice.

**In-process.** The method is a Python library (for example a PyTorch model
or a NumPy routine). The protocol imports it and calls it directly on
in-memory batches. Data never touches the disk between stages, and expensive
state such as model weights is loaded once per worker through a
``B.Resource``. This is the most efficient style, and the one to prefer when
the method can be imported into the protocol's environment.

**External program.** The method is a command line program, possibly in a
conda environment of its own, possibly written in another language. The
protocol declares the program with ``@B.shell_command`` and lets type
resolution write each batch to the files the program expects. Every batch
then round-trips through the file system, which costs I/O but requires no
changes to the program.

The two styles can be mixed. A protocol may preprocess batches in Python and
pass them to an external program, or call a program to produce intermediate
files that it reads back into memory.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - In-process
     - External program
   * - Data between stages
     - In memory (Arrow buffers)
     - Written to temporary files per batch
   * - Expensive state (weights)
     - Loaded once per worker (``Resource``)
     - Loaded by the program on every call
   * - Typical chunk size
     - Sized for the GPU (hundreds)
     - Large, to amortize I/O (thousands)
   * - Environment
     - Method importable from the protocol
     - Program on ``PATH`` or in its own conda env
   * - Work needed
     - Call the library in a ``map``
     - Declare the command, proxy types, and resolvers

Organizing the code
-------------------

Protocols, structs and resolvers are imported by name in other processes (Ray
workers, Scipion), so they must live in an importable package rather than in a
script or notebook. A layout that has worked well:

.. code-block:: text

    scipion_denoise/
    ├── __init__.py
    ├── types.py          Struct definitions
    ├── programs.py       Domains, @shell_command declarations, proxy types
    ├── resolvers.py      @B.resolver functions for the program's file formats
    ├── protocols.py      Protocol classes
    └── run.py            Command line entry point (RayPipelineRunner)
    tests/
    ├── test_resolvers.py
    ├── test_programs.py
    └── test_protocols.py

Keeping the parts in separate modules has a practical reason besides clarity:
type resolution only considers resolvers registered in modules that are
*visible* from the call site (see :ref:`resolver-visibility`). A package
``__init__.py`` that imports ``resolvers`` makes them visible wherever the
package is imported.

The running example
-------------------

The following pages use one example throughout: a denoising method for
particle images. It exists in two forms:

* a PyTorch model, importable as ``cryodenoise.load_model(weights)``, which
  maps a batch of images to denoised images and an estimated signal-to-noise
  ratio per image;
* a command line program from the same project, run as

  .. code-block:: bash

      cryodenoise predict --input particles.star --weights model.pt \
          --output denoised.mrcs --scores scores.csv

  which reads a RELION STAR file with its MRC stack and writes the denoised
  stack and a CSV file with one score per particle.

The goal is a protocol ``Denoise`` that consumes a stream of particles and
produces a stream of denoised particles with their scores, runs on Ray with
one GPU, and appears in Scipion as *denoise particles*.

The workflow
------------

The pages of this guide follow the order in which an integration is best
developed. Each step can be tested before the next one starts.

1. :doc:`data_model`: decide what the elements of the input and output
   streams are, and define structs for them.
2. :doc:`external_programs`: if the method is a program, declare it, define
   proxies for its file formats, and write resolvers to read and write them.
   Skip this step for in-process methods.
3. :doc:`writing_protocols`: write the protocol class: declarations, the
   streaming graph, and compute requirements.
4. :doc:`testing`: test the stage functions, resolvers and commands
   in-process, and the whole protocol once on Ray.
5. :doc:`deployment`: run the protocol on a workstation or cluster, tune its
   throughput, and register it in a Scipion plugin.

Checklist
---------

An integration is complete when:

* every struct, resolver and protocol is defined in an importable module;
* every input type has a resolution path from ``Path`` that contains a
  slice-aware resolver, so that the Ray runner can stream it from disk;
* for Scipion, every input type has a resolver from a Scipion object, and
  every output type a resolver to one;
* the last stage of ``steps()`` returns a dictionary matching ``outputs()``;
* expensive state is a ``Resource``, and the compute requirements are
  declared with ``@B.resources``;
* the stage functions, resolvers and commands have in-process tests, and one
  test runs the protocol end to end.
