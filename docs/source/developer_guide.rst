Developer Guide
===============

This guide is for contributors who extend or maintain Scipion Bridge. It
covers the internal architecture, the streaming intermediate representation,
how to write a backend, and how to test the library.

.. contents:: On this page
   :local:
   :depth: 2

Repository layout
-----------------

.. code-block:: text

    src/scipion_bridge/
    ├── core/
    │   ├── struct/        Struct, Set, Collection, schemas, storage, Arrow conversion
    │   ├── streaming/     Ops (surface API), IR, lowering, Pipeline, sinks
    │   ├── typed/         Type resolution graph, Dijkstra search, proxies
    │   ├── protocol/      Protocol, Input / Field / Resource, chaining, @resources
    │   ├── environment/   Service interfaces: shell exec, temp files, storage,
    │   │                  configuration, resources, compute declarations
    │   └── utils/         ARC, shell_command, AST helpers, Marker
    ├── backend/
    │   ├── standalone/    Default DI container
    │   ├── ray/           Ray streaming backend, runner, resource provider, sinks
    │   └── pyworkflow/    Scipion 3 protocol generation and resolvers
    ├── single_particle/   Particle types, proxies and resolvers
    └── visualize/         Visualization helpers
    tests/                 pytest suite mirroring the package structure

Coding conventions are in ``AGENTS.md`` at the repository root. In short:
Python 3.11, ``black`` formatting with trailing commas, ``match`` statements
over ``if`` / ``elif`` chains, guard clauses, no defensive ``getattr``
fallbacks, no ``TYPE_CHECKING`` or inline imports to avoid import cycles, and
no special cases for tests.

Dependency injection
--------------------

All side effects go through services provided by a
``dependency_injector`` container. Library code declares what it needs with
``Provide["service_name"]`` and ``@inject``:

.. code-block:: python

    @inject
    def _get_value(
        self,
        config_provider: ProtocolConfigurationProvider = Provide["protocol_config_provider"],
    ):
        ...

A backend defines a ``DeclarativeContainer`` with the same service names and
wires it into the ``scipion_bridge`` package:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Service
     - Standalone
     - Ray
     - PyWorkflow
   * - ``shell_exec``
     - ``StandaloneExecProvider``
     - ``StandaloneExecProvider``
     - ``runJob`` of the protocol
   * - ``temp_file_provider``
     - system temp dir
     - system temp dir
     - protocol ``tmp`` dir
   * - ``storage_provider``
     - Arrow
     - Arrow
     - Zarr in the protocol directory
   * - ``protocol_config_provider``
     - abstract
     - ``StaticProtocolConfigurationProvider``
     - Scipion form values
   * - ``resource_provider``
     - ``DefaultResourceProvider``
     - ``RayResourceProvider``
     - ``DefaultResourceProvider``
   * - ``streaming_backend``
     - ``RayBackend`` if installed
     - ``RayBackend``
     - ``RayBackend`` if installed

Ray workers are separate processes. Every Ray actor and task wires a
``RayContainer`` with the pipeline's parameter values before it runs user code,
so ``Field.value`` and resources work inside stages.

The streaming layer
-------------------

The streaming module has two layers: the user-facing **ops** and a minimal
**IR**. Backends implement only the IR.

Surface ops and lowering
^^^^^^^^^^^^^^^^^^^^^^^^

Every op is a ``Node`` with ``upstream`` and ``downstream`` edges and a
``lower(ctx)`` method. ``lower(nodes)`` walks the graph upstream from the
given sinks with a ``LoweringContext``. The context memoizes lowered nodes, so
shared subgraphs are lowered once, and it deduplicates sources by name. Each
node lowers itself to one IR node, and the context wires the IR edges in the
same order as the surface edges.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface op
     - Lowered to
   * - ``Source(name)``
     - ``IRSource(name)``
   * - ``map``, ``map_batch``
     - ``IRMap(func, compute)``
   * - ``map_element``
     - ``IRMap(ElementMapper(func, config))``. Worker pools are cached per
       process.
   * - ``chunk``, ``collect``, ``flatten``
     - ``IRAccumulate`` with the op's accumulate, initial-state and flush
       functions
   * - ``combine_latest``
     - ``IRAccumulate(tag_inputs=True)``
   * - ``group_by``
     - ``IRDemux(key_fn, template, source_name, max_keys)``
   * - ``sink``, ``write_to``, ``checkpoint``
     - ``IRSink(writer)``

The IR primitives
^^^^^^^^^^^^^^^^^

``IRSource(name)``
   An ingestion point. Items arrive through ``CompiledPipeline.send(name, item)``.

``IRMap(func, name, compute)``
   A stateless 1:1 transformation. ``compute`` is the ``ComputeAssignment``
   (resources plus group) that every call requires, or ``None``.

``IRAccumulate(accumulate_fn, initial_state_fn, flush_fn, tag_inputs)``
   A stateful N:M transformation. The backend keeps one ``state`` per stage:

   .. code-block:: python

       state = initial_state_fn()
       state, outputs = accumulate_fn(state, item)    # for every item
       state, outputs = flush_fn(state)               # on FLUSH (optional)

   With ``tag_inputs=True``, items are wrapped as ``Tagged(port, item)``, where
   ``port`` is the position of the sending stage among the node's upstream
   nodes. A node therefore cannot consume the same stream on two inputs, and
   lowering rejects that case.

``IRSink(writer)``
   A terminal node. It awaits ``writer.write(item)`` for every item and
   ``writer.finalize()`` on FLUSH.

``IRDemux(key_fn, template, source_name, max_keys)``
   Per-key routing for ``group_by``. ``template`` is the exit node of a
   separately lowered child graph whose only source is ``source_name``. For
   every new key, the backend compiles a copy of the template (``clone_ir``) and
   emits each child's results as ``Keyed(key, result)``.

Execution contract
^^^^^^^^^^^^^^^^^^

A backend must guarantee:

* **Per-stage ordering.** A stage processes items one at a time, in arrival
  order. Accumulator state and stage logic never run concurrently.
* **FLUSH barrier.** ``flush()`` returns only after all items sent before it
  have passed through every stage, every ``flush_fn`` has run, and every sink
  has been finalized. A stage with several upstream stages forwards FLUSH once
  it has received FLUSH from all of them.
* **Error propagation.** An exception in any stage fails the pipeline, and the
  driver's next ``send`` or ``flush`` raises it.
* **Backpressure.** Buffering between stages is bounded.

Writing a backend
^^^^^^^^^^^^^^^^^

Implement two interfaces from ``scipion_bridge.core.streaming.backend``:

.. code-block:: python

    class MyBackend(StreamingBackendProvider):
        def compile(self, ir_sinks: list[IROp]) -> CompiledPipeline:
            ...   # traverse upstream from ir_sinks, build stages, wire routes


    class MyCompiledPipeline(CompiledPipeline):
        def send(self, source_name: str, value) -> None: ...
        def flush(self) -> None: ...
        def stats(self) -> dict[str, StageStats]: ...
        def close(self) -> None: ...

Then provide it as the ``streaming_backend`` service of a container. A
single-threaded reference backend is a good way to test new ops: run every
``IRMap`` inline, keep accumulator state in a dict, and call writers with
``asyncio.run``.

The Ray backend
^^^^^^^^^^^^^^^

``backend/ray/backend.py`` compiles the IR as follows:

* every source, accumulator, demux and sink becomes a named actor, a
  ``_PipelinedStage`` with a bounded inbox, a compute task, and an emitter task;
* maps are not actors. The stage upstream of a map submits each call as a Ray
  task, with up to ``queue_size`` calls in flight, and forwards the results in
  order. Long-running compute groups run their calls on a shared executor
  actor, which holds the GPUs until FLUSH;
* resource claims (``gpus``, ``min_vram``, ``cpus``) are translated into Ray
  task options by ``gpu_claim`` and the backend's ``_claim``;
* a failing stage aborts the sources, so the driver fails fast.

Type resolution internals
-------------------------

``core/typed/resolve.py`` keeps a single ``Registry`` holding a
``networkx.DiGraph``. Nodes are types. Edges carry the resolver class, its
module (namespace), and whether it accepts ``metadata`` or ``slice``.

* ``@resolver`` reads the ``value`` and return annotations and adds an edge.
  Function resolvers are wrapped in a generated class with a ``forward``
  method.
* For every registered type, *downcast* edges to its bases in the MRO are
  added, so resolvers for a base class also apply to its subclasses.
* A lookup restricts the graph to the visible namespaces (see
  :doc:`user_guide/type_resolution`) and runs a custom Dijkstra search
  (``dijkstra.py``). Ties are broken in this order: path weight, then
  resolvers local to the caller, then slice-aware resolvers, then the more
  specific module path.
* The result is a ``ComposedResolver``, a list of steps that can be called
  repeatedly or iterated in chunks without searching again.

Proxies and ARC
---------------

``core/utils/arc.py`` implements a process-wide ``FileReferenceCounter``:

* ``register_temporary_file`` / ``new_managed_file`` start tracking a path with
  a count of 1;
* ``add_reference`` / ``remove_reference`` are called when managed proxies for
  the path are created and garbage collected;
* when a count reaches 0, the file is deleted through the
  ``temp_file_provider``.

Adding a reference to a path that is not tracked yet emits a
``DeprecationWarning``: such paths are usually persistent user files, which
reference counting must not delete.

``@shell_command`` checks the AST of the decorated function at definition
time: the body must consist of a single ``pass`` statement. Calling the
original function first lets Python validate the arguments before any command
is built.

Testing
-------

Tests live in ``tests/``, mirror the package layout, and run with ``pytest``.
Run only the affected test files during development. The full suite takes
several minutes.

.. code-block:: bash

    pytest tests/scipion_bridge/struct/test_set.py
    pytest -m "not ray"          # skip tests that need the Ray test cluster
    pytest -m "not slow"

``tests/conftest.py`` wires the standalone container for every test.
Tests that request the ``ray_cluster`` fixture run on a session-wide local
Ray cluster with 2 CPUs and 2 logical GPUs, and they are marked ``ray``
automatically.

To test shell commands without running them, override the ``shell_exec``
provider with a mock and assert on the generated arguments:

.. code-block:: python

    @B.shell_command
    def my_command(i: str, *, value: int):
        pass


    def test_command(mocker, setup_test_container):
        exec_mock = mocker.Mock()
        with setup_test_container.shell_exec.override(exec_mock):
            my_command("/in", value=42)

        _, _, args, _ = exec_mock.call_args.args
        assert args == ["my_command", "-i", "/in", "--value", "42"]

Struct and set tests use a mock storage. Do not add code paths that exist only
for tests: give the mock the methods the library calls, and rely on duck
typing.

Before submitting a change:

.. code-block:: bash

    black .
    pyright
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
