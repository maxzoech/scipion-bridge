Testing
=======

The test suite lives in ``tests/``, mirrors the package layout, and runs with
``pytest``. Almost all of its runtime is spent in the Ray streaming tests,
so the practical question for every change is which subset to run. This
article describes the organization of the suite, the shared Ray test cluster,
how to write tests that stay fast, and how coverage and CI work.

.. contents:: On this page
   :local:
   :depth: 2

Setup
-----

Install the test dependencies into the development environment:

.. code-block:: bash

    pip install pytest pytest-mock pytest-cov pytest-xdist ".[ray]"

``tests/conftest.py`` wires the standalone container once, before the test
modules are collected, so that structs defined at the top level of test
modules work, and provides it to every test as the autouse fixture
``setup_test_container``.

Choosing what to run
--------------------

Pick the smallest run that covers a change:

.. list-table::
   :header-rows: 1
   :widths: 45 35 20

   * - Command
     - Runs
     - Time
   * - ``pytest -m "not ray" tests``
     - struct, typed, protocol and all in-process tests; Ray never starts
     - ~15 s
   * - ``pytest tests/scipion_bridge/streaming/test_<file>.py``
     - the Ray tests of one module
     - seconds to ~2 min
   * - ``pytest -n auto --dist loadgroup tests``
     - the full suite in parallel
     - ~3 min
   * - ``pytest tests``
     - the full suite in one process
     - ~6 min

Run the full suite only when a change touches shared backend code that the
targeted modules cannot cover. On machines with limited memory, limit the
number of parallel workers (``-n 8`` instead of ``-n auto``), since every
worker starts a Ray cluster of its own.

Ray needs local networking to start its cluster. In a sandbox that blocks it,
``ray.init()`` times out during startup and every Ray test errors; the
non-Ray tests still run there.

The Ray test cluster
--------------------

* Tests that need Ray request the session fixture ``ray_cluster`` from
  ``tests/conftest.py``, through
  ``pytestmark = pytest.mark.usefixtures("ray_cluster")`` for a module or
  ``@pytest.mark.usefixtures("ray_cluster")`` for a single test. Test modules
  do not call ``ray.init()`` themselves.
* Every test using the fixture is marked ``ray`` automatically, so
  ``-m "not ray"`` and ``-m ray`` select the two halves of the suite.
* The cluster has 2 CPUs and 2 logical GPUs. Tests that assert scheduling
  (overlap of calls, queuing on GPUs) rely on these numbers.
* Its object store is capped at 512 MiB, since every xdist worker starts a
  cluster of its own. Keep the data of Ray tests small.
* Ray's auto-init is disabled (``RAY_ENABLE_AUTO_CONNECT=0``): a test that
  uses Ray without the fixture fails instead of starting an unconfigured
  cluster.
* The fixture exports the ``PYTHONPATH`` of the test modules to the raylet
  rather than passing a runtime environment, so that workers can reuse the
  processes Ray prestarts.
* Every actor of a pipeline starts a worker process of its own, which imports
  ``scipion_bridge`` (about 1.5 s). The worker count, not the item count, makes
  a Ray test slow.

Writing tests
-------------

**Test logic in-process first.** Operations lower to plain functions: the
``accumulate_fn``, ``initial_state_fn`` and ``flush_fn`` of an
``IRAccumulate`` can be called directly (see
``tests/scipion_bridge/streaming/test_chunk_op.py``). Keep one Ray
integration test per feature for the behavior that only exists across
actors, such as flush ordering, backpressure or resource scheduling.

**Mock services, not code paths.** Override providers of the container to
replace external effects:

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

Struct and set tests use a mock storage. Never add code paths that exist only
for tests: give the mock the methods the library calls, and rely on duck
typing.

**Keep timing tests short and relational.** Keep sleeps as short as the
assertion allows, and assert relations (overlap, order) rather than absolute
durations. Modules with timing assertions are marked
``pytest.mark.xdist_group("timing")`` so that parallel runs execute them on
one worker, one after another. Add new timing-sensitive modules to that group.
Their tests are also marked ``timing`` automatically, to run them alone
(``-m "ray and timing"``) when other clusters on the same CPUs distort their
timings.

**Batch subprocesses.** Starting a Python subprocess costs about 2 s for the
imports. Batch work into one subprocess where possible (see
``test_marker_pickle.py``).

**Reproduce storage bugs at the engine level.** ``test_root_engine.py`` tests
``RootEngine`` directly with explicit schema entries, which isolates a storage
bug from the containers.

Coverage
--------

Line coverage (``--cov=scipion_bridge``) only measures the pytest process.
Code that runs inside Ray actors and tasks is not counted: Ray stops actors
with ``ray.kill``, which discards the coverage data of the worker. Coverage of
the Ray backend's worker-side code therefore comes only from in-process tests,
which is one more reason to test logic outside of Ray. Do not remove a Ray
test because coverage reports no unique lines for it.

Continuous integration
----------------------

CI (``.github/workflows/python-app.yml``) runs on every push with Python 3.11:

1. ``flake8`` for syntax errors and undefined names (failing), and for style
   and complexity (warnings only);
2. the non-Ray tests with coverage;
3. the Ray tests in one process (``-m ray``), appending to the same coverage
   data. Every xdist worker would start a Ray cluster of its own: parallel
   clusters ran the 16 GB runner out of memory and distorted the timing
   tests.

Benchmarks
----------

Benchmarks are scripts in ``benchmarks/``, outside of the test suite, because
they need real data and a quiet machine. ``bench_mrc_read.py`` compares readers
for chunks of an MRC particle stack: a memory-mapped copy, the sequential
reader of ``scipion_bridge.single_particle.mrc`` and, optionally, the
``vitrio`` package. It reads from blocks of the stack that are cold in every
cache, so use a stack that is much larger than the memory of the machine:

.. code-block:: bash

    MRC_STACK=/path/to/stack.mrcs python benchmarks/bench_mrc_read.py out.jsonl \
        --readers mmap,A --conc 1,2,4 --chunks 1000,256

Every measurement is appended to ``out.jsonl`` as one JSON object with
throughput in MB/s and particles/s.
