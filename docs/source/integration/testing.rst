Step 4: Testing the Protocol
============================

.. currentmodule:: scipion_bridge

A protocol consists of ordinary Python functions arranged in a graph, so most
of it can be tested without a cluster, without GPUs and without the external
program. This page describes a layered testing strategy: test the pieces
in-process first, where tests are fast and failures are easy to read, and run
the whole protocol on Ray once, for the behavior that only exists across
processes.

The examples use ``pytest`` and ``pytest-mock``.

.. contents:: On this page
   :local:
   :depth: 2

Setting up the test environment
-------------------------------

The library reaches its services (command execution, temporary files,
parameter values, resources) through a dependency injection container. Tests
wire the standalone container once, in a ``conftest.py``, and override
individual services where needed:

.. code-block:: python

    # tests/conftest.py
    import pytest
    from scipion_bridge.backend import configure_default_env

    _container = configure_default_env(packages=["scipion_bridge", "scipion_denoise"])


    @pytest.fixture
    def container():
        return _container

``configure_default_env`` wires the ``scipion_bridge`` package by default;
listing the protocol's package as well makes the container's services
available to code there that uses ``Provide[...]``.

Testing resolvers
-----------------

Resolvers are plain functions over files and data, and the easiest part to
test. Write a small input file into pytest's ``tmp_path``, wrap it in its
proxy, and resolve it:

.. code-block:: python

    import numpy as np
    import scipion_bridge as B

    from scipion_denoise.programs import ScoreTable


    def test_read_score_table(tmp_path):
        path = tmp_path / "scores.csv"
        path.write_text("0.5\n1.5\n")

        scores = B.resolve(ScoreTable(path), astype=np.ndarray)

        np.testing.assert_allclose(scores, [0.5, 1.5])

For slice-aware resolvers, also test chunked resolution, since that is how the
Ray runner reads inputs:

.. code-block:: python

    def test_npy_stack_streams_in_chunks(tmp_path):
        path = tmp_path / "stack.npy"
        np.save(path, np.zeros((10, 8, 8), np.float32))

        chunks = list(
            B.resolve_iter(NpyImageStack(path), astype=B.Set[spa.Particle], chunk_size=4),
        )

        assert [len(c) for c in chunks] == [4, 4, 2]

Testing command declarations
----------------------------

A command declaration can be tested without the program by replacing the
``shell_exec`` service with a mock. The mock receives the program name, the
domain, the complete argument list and the subprocess options:

.. code-block:: python

    from scipion_denoise.programs import cryodenoise_predict


    def test_predict_arguments(mocker, container):
        exec_mock = mocker.Mock(return_value=0)
        with container.shell_exec.override(exec_mock):
            cryodenoise_predict(
                particles="in.star",
                weights="w.pt",
                output="out.mrcs",
                scores="s.csv",
            )

        _, _, args, _ = exec_mock.call_args.args
        assert args == [
            "cryodenoise", "predict",
            "--input", "in.star",
            "--weights", "w.pt",
            "--output", "out.mrcs",
            "--scores", "s.csv",
        ]

For a proxified command, a mock can also write the files the real program
would write, which tests the whole round trip (input files written, outputs
read back) without the program:

.. code-block:: python

    def fake_cryodenoise(func_name, domain, args, run_args):
        output = args[args.index("--output") + 1]
        scores = args[args.index("--scores") + 1]
        mrcfile.write(output, np.ones((3, 8, 8), np.float32), overwrite=True)
        np.savetxt(scores, [1.0, 2.0, 3.0], delimiter=",")
        return 0


    def test_denoise_batch(container, particles):
        with container.shell_exec.override(fake_cryodenoise):
            out = denoise_batch(particles, weights="w.pt")

        assert len(out) == 3
        np.testing.assert_allclose(out["snr"].ravel(), [1.0, 2.0, 3.0])

Testing stage functions
-----------------------

The functions passed to ``map`` are methods of the protocol and can be called
directly. Two services must be provided for them:

* **Parameter values.** ``Field.value`` asks the ``protocol_config_provider``
  service, which the standalone container leaves abstract. Provide the
  values for the test with a ``StaticProtocolConfigurationProvider``.
* **Resources.** The standalone container provides a process-local resource
  provider, so resources are built on first access, as in a worker. For a
  model that needs a GPU, test with a small stand-in, or override the
  builder's dependencies with a mock.

.. code-block:: python

    from dependency_injector import providers
    from scipion_bridge.core.environment.protocol_config import (
        StaticProtocolConfigurationProvider,
    )


    @pytest.fixture
    def parameters(container):
        values = StaticProtocolConfigurationProvider({"chunk_size": 4})
        with container.protocol_config_provider.override(providers.Object(values)):
            yield


    def test_to_output(parameters, particles):
        protocol = Denoise()
        pixels = np.zeros((len(particles), 8, 8), np.float32)
        snr = np.arange(len(particles), dtype=np.float32)

        outputs = protocol._to_output((particles, pixels, snr))

        assert set(outputs) == {"particles"}
        assert isinstance(outputs["particles"], B.Set[DenoisedParticle])

Validating outputs in-process
-----------------------------

``Protocol._verify_outputs`` is the function that checks the result of the
last stage against ``outputs()``. Calling it on the output of the last stage
function catches type mismatches, the most common integration error, before
the protocol runs anywhere:

.. code-block:: python

    def test_outputs_match_declaration(parameters, particles):
        protocol = Denoise()
        outputs = protocol._to_output((particles, pixels, snr))
        protocol._verify_outputs(outputs)        # raises ValidationError on mismatch

Testing stateful logic
----------------------

The stateful operations lower to plain functions, which can be driven
in-process. Lower the graph, find the accumulator node, and feed it items as
its stage would:

.. code-block:: python

    from scipion_bridge.core.streaming import IRAccumulate, Source, lower


    def run_accumulator(op, items):
        (sink,) = lower([op.sink(print)])
        (accumulate,) = sink.upstream
        assert isinstance(accumulate, IRAccumulate)

        state = accumulate.initial_state_fn()
        emitted = []
        for item in items:
            state, out = accumulate.accumulate_fn(state, item)
            emitted.extend(out)
        if accumulate.flush_fn is not None:
            _, out = accumulate.flush_fn(state)
            emitted.extend(out)
        return emitted

This is useful for checking that a ``chunk`` size and the input batch sizes
produce the batches a model expects. The library's own operator tests use the
same technique.

An end-to-end test on Ray
-------------------------

Some behavior only exists when the protocol runs as a pipeline: parameter
values reaching the workers, resources being built in the workers, outputs
validated after every batch, and the final flush. One end-to-end test per
protocol covers it. Run it with a small input, a callback sink, and Ray
started with a fixed, small number of CPUs and logical GPUs:

.. code-block:: python

    import ray
    from scipion_bridge.backend.ray import RayPipelineRunner


    @pytest.fixture(scope="session")
    def ray_cluster():
        ray.init(num_cpus=2, num_gpus=1)    # logical GPUs need no driver
        yield
        ray.shutdown()


    @ray.remote
    class Collector:
        def __init__(self):
            self.sizes = []

        def append(self, n):
            self.sizes.append(n)

        def get(self):
            return self.sizes


    def test_denoise_end_to_end(ray_cluster, tmp_path, monkeypatch):
        monkeypatch.setenv("SCIPION_CHUNK_SIZE", "4")
        star_path = write_sample_particles(tmp_path, n=8)
        collector = Collector.remote()

        with RayPipelineRunner(
            DenoiseExternal(),
            parameters={"chunk_size": 4},
            sink=lambda out: ray.get(collector.append.remote(len(out["particles"]))),
        ) as runner:
            runner.run(particles=star_path)

        assert ray.get(collector.get.remote()) == [4, 4]

The sink runs in a Ray actor, not in the test process, so results are
collected through a small actor rather than a list in the test.

The stages run in Ray worker processes, which wire their own containers.
Services overridden in the test process, such as a mocked ``shell_exec``, do
not apply there: the workers run the real program. To run an external program
variant end to end without the program, place a fake executable named like
the program on the ``PATH`` before Ray starts, so that the workers inherit it.
Logical GPUs let Ray schedule GPU stages on a machine without GPUs; stage code
that actually uses CUDA needs a CPU stand-in in such a test.

Keep end-to-end tests few and small. Every stage of a Ray pipeline starts a
worker process that imports the library, which takes on the order of a second
per process; the number of stages, not the number of items, dominates the
runtime of such a test.

Checklist
---------

* Resolvers have tests with small files, including chunked resolution for
  slice-aware resolvers.
* Command declarations are tested with a mocked ``shell_exec``, and the round
  trip with a fake program.
* Stage functions are called directly with parameter values provided, and
  their results pass ``_verify_outputs``.
* One end-to-end test per protocol runs it on a small Ray cluster.

Next, :doc:`deployment` runs the protocol on real data and registers it in
Scipion.
