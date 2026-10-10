Developer Guide
===============

This guide is for contributors who extend or maintain Scipion Bridge itself.
Where the user guide explains what the library does, this guide explains how
it does it: the structure of the code, the mechanisms behind each layer, the
contracts between them, and how changes are tested. It assumes familiarity
with the concepts described in the user guide.

.. contents:: On this page
   :local:
   :depth: 2

Design principles
-----------------

A few principles run through the whole code base. They explain many
structural decisions and should guide new code.

**Side effects go through injected services.**
   Library code never runs a process, creates a file or reads a parameter
   directly. It asks for a service, and a backend container provides it. This
   is what makes protocols portable between backends, and what makes the
   library testable without external programs (see
   :doc:`dependency_injection`).

**Small contracts between layers.**
   The streaming surface API lowers to five IR primitives; backends implement
   only those. Type conversions are edges of one graph; producers and
   consumers of data never know each other. Each layer can change behind its
   contract.

**Data is columnar and serialized as Arrow.**
   Containers are views over columns. Every component that moves data between
   processes relies on the Arrow form of the containers, never on pickling
   Python objects element by element.

**Fail early and consistently.**
   Invalid declarations raise when a class is defined (unannotated
   attributes, non-empty shell command bodies), invalid graphs when they are
   built or lowered, invalid wirings when protocols are chained. Errors in
   stages fail the pipeline immediately. Code does not guess around missing
   data.

Repository layout
-----------------

.. code-block:: text

    src/scipion_bridge/
    ├── core/
    │   ├── struct/        Struct, Set, Collection, schemas, storage engine, Arrow conversion
    │   ├── streaming/     Ops (surface API), IR, lowering, Pipeline, sinks, spill stores
    │   ├── typed/         Type resolution graph, Dijkstra search, proxies
    │   ├── protocol/      Protocol, Input / Field / Resource, chaining, @resources
    │   ├── environment/   Service interfaces: shell exec, temp files, storage,
    │   │                  configuration, resources, compute declarations
    │   └── utils/         ARC, shell_command, Marker, AST and signature helpers
    ├── backend/
    │   ├── standalone/    Default DI container
    │   ├── ray/           Ray streaming backend, runner, mailboxes, profiling, sinks
    │   └── pyworkflow/    Scipion 3 protocol generation, container and resolvers
    ├── single_particle/   Particle types, proxies and resolvers
    └── visualize/         Visualization helpers
    tests/                 pytest suite mirroring the package structure
    docs/                  This documentation (Sphinx)

Dependencies between packages largely point downwards: ``core.streaming``
builds on ``core.struct``, ``core.protocol`` on ``core.streaming``, and the
Ray and pyworkflow backends on ``core``. (The proxy and ARC modules refer to
the providers of the standalone container by name.) Import cycles are avoided
by this structure, not by deferred imports or ``TYPE_CHECKING`` guards.

Where to start for a change
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Change
     - Read
   * - A bug in reading or writing container data
     - :doc:`struct_storage`
   * - A new streaming operation
     - :doc:`streaming_internals`
   * - A new execution backend
     - :doc:`streaming_internals`, :doc:`dependency_injection`
   * - Scheduling, buffering or profiling on Ray
     - :doc:`ray_backend`
   * - Conversions, proxies, file cleanup, shell commands
     - :doc:`typed_internals`
   * - Support for a new Scipion object type
     - :doc:`pyworkflow_backend`
   * - Tests and CI
     - :doc:`testing`
   * - Code style and submitting changes
     - :doc:`contributing`

Articles in this guide
----------------------

* :doc:`dependency_injection`: services, containers, wiring, and how Ray
  workers obtain their services.
* :doc:`struct_storage`: the storage engine behind ``Struct``, ``Set`` and
  ``Collection``: key paths, the column invariant, reads, writes,
  serialization and thread safety.
* :doc:`streaming_internals`: lowering, the IR primitives, the execution
  contract, adding an operation and writing a backend.
* :doc:`ray_backend`: how the Ray backend compiles and runs the IR: stage
  actors, hosted maps, compute groups, mailboxes, ``group_by`` children,
  errors and profiling.
* :doc:`typed_internals`: the resolver registry and search, proxies, ARC,
  shell commands and pickling of specialized classes.
* :doc:`pyworkflow_backend`: how a Scipion 3 protocol is generated and how
  it streams data.
* :doc:`testing`: the test suite, the Ray test cluster, and how to write fast
  tests.
* :doc:`contributing`: conventions and the checklist for changes.
