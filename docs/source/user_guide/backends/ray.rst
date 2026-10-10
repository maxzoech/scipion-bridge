Ray Backend
===========

.. currentmodule:: scipion_bridge

The Ray backend runs streaming pipelines in parallel on one machine or
distributed across a `Ray <https://www.ray.io>`_ cluster. This article
describes how a pipeline is mapped onto Ray, how to run protocols with
``RayPipelineRunner`` from Python and from the command line, how buffering
and spilling bound memory use, how to profile a pipeline, and how to use an
existing cluster.

.. contents:: On this page
   :local:
   :depth: 2

How a pipeline runs on Ray
--------------------------

The backend compiles a streaming graph into Ray actors and tasks:

* sources, stateful operations (``chunk``, ``collect``, ``combine_latest``,
  ...), ``group_by`` routers and sinks become **actors**, one process each;
* every call of a ``map`` runs as a **Ray task**, submitted by the stage before
  it and scheduled by the compute resources declared with ``@B.resources``.
  Long-running protocols run their calls on a dedicated executor actor that
  holds their GPUs.

Stages run concurrently and forward results in order. Results of maps stay in
Ray's shared-memory object store and travel between stages as references;
the receiving stage fetches an item only when it processes it. ``Set``
batches are stored there as Arrow buffers, which processes on the same node
read without copying.

Each actor is named ``<pipeline id>:<stage index>:<description>``, which
identifies it in the Ray dashboard. ``RayBackend`` starts a local Ray
instance on first use if none is running.

Running a protocol: RayPipelineRunner
-------------------------------------

.. code-block:: python

    from pathlib import Path
    from scipion_bridge.backend.ray import RayPipelineRunner

    with RayPipelineRunner(
        FoundationModelInference(),
        parameters={"chunk_size": 512},
        sink=OutputsToZarr("latents.zarr"),      # see "Persisting outputs" below
    ) as runner:
        runner.run(particles=Path("particles.star"))

The runner wires the Ray container with the parameter values, compiles the
protocol's pipeline on the first ``run``, and feeds it with inputs resolved
from files.

Constructor arguments
^^^^^^^^^^^^^^^^^^^^^

``RayPipelineRunner(protocol, *, parameters=None, origin_types=None, sink=None,
queue_size=2, max_in_flight=None, buffer_size=4, spill_threshold=4,
spill_store=None, profile=None, profile_log_level=logging.INFO)``

``parameters``
   Values of the protocol's fields. Unknown names raise ``ValueError``, and so
   do missing required fields when the pipeline is compiled. Parameters
   without a value use their default. ``set_parameters`` replaces them before
   the first run.

``origin_types``
   The type each input is given as, by input name. The default is ``Path``.
   For every input, a resolver from this type to the declared input type is
   precomputed when the runner is created. It must contain a slice-aware step
   so that the input can be streamed in chunks (see
   :doc:`/user_guide/type_resolution/chunked_resolution`).

``sink``
   Where the protocol's output dictionaries go: a ``Sink``, a ``SinkWriter``,
   or a callback. **Without a sink, outputs are discarded.**

``queue_size``
   The number of items each stage reads ahead of its computation.

``max_in_flight``, ``buffer_size``, ``spill_threshold``, ``spill_store``
   How stages buffer the items waiting for them; see `Buffering and
   spilling`_.

``profile``, ``profile_log_level``
   Profile the pipeline into a trace file; see `Profiling`_.

Running
^^^^^^^

``run(**inputs)`` resolves every input in chunks and sends the chunks into the
pipeline, interleaving the inputs one chunk at a time. Only inputs for which a
resolver was precomputed are streamed, so every input is given as a value of
its origin type. When all inputs are exhausted, ``run`` flushes the pipeline
and prints the throughput and a table of per-stage statistics (see
:doc:`/user_guide/streaming/pipelines`). Use the runner as a context manager,
or call ``close()``, to release the Ray actors.

.. _ray-persisting-outputs:

Persisting outputs
^^^^^^^^^^^^^^^^^^

The sink receives the **dictionaries** returned by the protocol's last stage,
after validation, not the containers inside them. ``TensorStoreSinkWriter``
writes one container per item and rejects dictionaries, so it is used through
a small writer that dispatches each output to a writer of its own:

.. code-block:: python

    from pathlib import Path

    from scipion_bridge.backend.ray.sinks import TensorStoreSinkWriter


    class OutputsToZarr:
        """Write every output of a protocol to a Zarr group of its own."""

        def __init__(self, root: str | Path):
            self.root = Path(root)
            self.writers: dict[str, TensorStoreSinkWriter] = {}

        async def write(self, outputs: dict) -> None:
            for name, value in outputs.items():
                if name not in self.writers:
                    self.writers[name] = TensorStoreSinkWriter(self.root / name)
                await self.writers[name].write(value)

        async def finalize(self) -> None:
            for writer in self.writers.values():
                await writer.finalize()

Each output then appears under ``<root>/<output name>/<field path>``. Any
other ``SinkWriter`` or callback works the same way (see
:doc:`/user_guide/streaming/sinks`).

Command line applications
-------------------------

``launch_as_terminal_application()`` builds an ``argparse`` parser from the
protocol's declarations and runs the protocol with the parsed arguments.
Inputs become path arguments. Field names are converted to kebab case, so
``chunk_size`` becomes ``--chunk-size``. The protocol's docstring becomes the
program description.

.. code-block:: python

    # embed.py
    if __name__ == "__main__":
        RayPipelineRunner(FoundationModelInference()).launch_as_terminal_application()

.. code-block:: bash

    python embed.py --help
    python embed.py --particles particles.star --chunk-size 512 --model-type CRYO_IEF_SMALL

``bool`` fields become ``--name`` / ``--no-name``, and ``Enum`` fields accept
the member names. A field that is not optional is a required option, even
when it has a default. Every application also accepts ``--profile [PATH]``.

Buffering and spilling
----------------------

Every stage has a **mailbox**: items sent to it wait there until the stage is
ready for them. A stage therefore keeps working while the stage downstream of
it is busy, as long as that stage's mailbox has room. When a mailbox is full,
the stage upstream of it waits, and so on as far as the driver: ``send`` (and
``RayPipelineRunner.run``) then waits too. A slow stage throttles the pipeline
instead of piling up its input.

Waiting items are held in Ray's object store, and Ray needs that store for
every transfer between processes, including the inputs and results of a GPU
model. An unbounded backlog fills the store, after which every stage waits
for Ray to spill and restore objects at disk speed. Bound the backlog, or
spill it:

``buffer_size``
   The most items that may wait for a stage (default 4). ``None`` removes the
   cap, so a slow stage never stalls the stages upstream of it; combine that
   with spilling.

``spill_threshold``
   The most waiting items a stage keeps in the object store (default 4).
   Further items go to a **spill store** and are read back when the stage gets
   to them. Spilling only happens when the threshold is below
   ``buffer_size``, so with the defaults nothing spills. ``None`` never
   spills.

``spill_store``
   Creates the spill store of a stage from its label. The default store
   pickles every item into a temporary directory that is removed when the
   pipeline closes. A custom ``SpillStore`` decides what to persist; it could,
   for example, write only the columns of a ``Set`` that changed.

``max_in_flight``
   The number of items on their way out of a stage at once, through all the
   maps between it and the next stage (default 16). An item keeps its place
   until it has passed every map, so a slow map behind a fast one needs enough
   places for the fast map to keep working. A long-running map has two calls
   in flight: one computes while the next one's input is transferred.

Both caps count **items, not bytes**. A chunk of 64 particles of 336×336
pixels (33 MB) and a single particle (0.4 MB) each count as one. Besides its
mailbox, a stage holds up to ``queue_size`` items in its inbox, ``queue_size``
in its outbox and ``max_in_flight`` on its routes. Size the caps for the
largest items of the pipeline.

Some operators keep items in their own state, outside of these caps:
``combine_latest`` keeps the items of its first input until the second input
has produced its first item, and ``collect`` keeps its sample. Their
mailboxes keep accepting items meanwhile, so they never block the pipeline,
but their state can grow.

A ``group_by`` starts a worker (or the pipeline of a key) when the first item
assigned to it arrives, without waiting for it: the items wait until it has
started, and the items of other keys are routed on. Workers of a burst of new
keys start in parallel. Up to ``buffer_size`` items wait per worker; beyond
that, the ``group_by`` waits like any other stage. ``RayBackend(group_by_workers=4)``
sets the default number of workers (see :doc:`/user_guide/streaming/group_by`).

The statistics table shows per stage the most items that waited
(``buffered``), the items spilled, and the time spent reading waiting items
(``fetch_s``) and spilling them (``spill_s``). A stage with many waiting items
is the bottleneck of the pipeline.

Environment variables
---------------------

Every buffering option of the runner can be overridden by an environment
variable, which is convenient for tuning a protocol without changing code:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Effect
   * - ``SCIPION_CHUNK_SIZE``
     - The number of elements per input chunk. The default is estimated from
       the size of one element.
   * - ``SCIPION_TARGET_BYTE_SIZE``
     - The target size of an input chunk in bytes, used when the chunk size is
       estimated (default 32 MiB).
   * - ``SCIPION_STREAM_QUEUE_SIZE``
     - Overrides ``queue_size``.
   * - ``SCIPION_STREAM_MAX_IN_FLIGHT``
     - Overrides ``max_in_flight``.
   * - ``SCIPION_STREAM_BUFFER_SIZE``
     - Overrides ``buffer_size``; ``none`` removes the cap.
   * - ``SCIPION_STREAM_SPILL_THRESHOLD``
     - Overrides ``spill_threshold``; ``none`` disables spilling.
   * - ``SCIPION_STREAM_SPILL_DIR``
     - Spills by pickling into this directory instead of ``spill_store``.
   * - ``SCIPION_STREAM_PROFILE``
     - Overrides ``profile``: a path for the trace, ``1`` for a timestamped
       file in the working directory, ``0`` for no profiling.
   * - ``SCIPION_STREAM_PROFILE_LOG_LEVEL``
     - Overrides ``profile_log_level``, e.g. ``debug`` to log every item.
   * - ``SCIPION_BRIDGE_CPUS``
     - *Set by the backend* inside calls that reserved CPU cores.
   * - ``SCIPION_BRIDGE_GPU_FRACTION``
     - *Set by the backend* inside calls that claimed a fraction of a GPU,
       together with ``XLA_PYTHON_CLIENT_MEM_FRACTION`` and
       ``TF_FORCE_GPU_ALLOW_GROWTH`` (see :doc:`/user_guide/protocols/compute`).
   * - ``NUMPY_MADVISE_HUGEPAGE``
     - *Set by the backend* to ``0`` in every Ray worker, unless set in the
       driver's environment, whose value is passed on (see
       :doc:`/user_guide/sharp_bits`).

Profiling
---------

A profiled pipeline records what every process does: the stage actors, the
Ray tasks running maps, the executors of long-running compute groups, the
``group_by`` children and the driver. Enable it with ``profile=<path>`` (or
``True`` for a generated file name) on ``RayPipelineRunner`` or
``RayBackend``, with ``SCIPION_STREAM_PROFILE``, or with ``--profile``:

.. code-block:: bash

    python embed.py --particles particles.star --profile embed_trace.json
    SCIPION_STREAM_PROFILE=1 SCIPION_STREAM_PROFILE_LOG_LEVEL=debug python embed.py ...

The events go to two places:

* **The log.** Every process logs its events on the logger
  ``scipion_bridge.profile``, and Ray forwards the output of the workers to
  the driver, prefixed with the actor class and process id::

      (RayAccumulatorActor pid=81234) [2:chunk] compute/process 12.3ms seq=4 port=0

  Lifecycle events (stage start, FLUSH, executor start and kill, start of a
  ``group_by`` child, errors) are logged at INFO, the events of every item at
  DEBUG. Ray merges matching lines from different processes; set
  ``RAY_DEDUP_LOGS=0`` in the shell that starts the driver to see all of them.
* **The trace file**, in the Trace Event Format of Chrome. Open it in
  https://ui.perfetto.dev or ``chrome://tracing``. It is written **while the
  pipeline runs**, so it can be opened at any time and reloaded to see newer
  events. Workers send their events at least once per second, and every
  event before a FLUSH is in the file when ``flush`` returns. ``close``
  completes the file.

In the trace, each process is one row, named after its stage label and
process id. Each part of a stage's work has its own lane:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Lane
     - Events
   * - ``mailbox``
     - ``backpressure``: a ``push`` waiting for room in a full buffer.
   * - ``fetch``
     - ``fetch``: reading the next item, with its ``source`` (``inline``,
       ``object_store`` or ``spill``).
   * - ``spill``
     - ``spill``: writing an item to the spill store.
   * - ``compute``
     - ``process`` per item (``seq``, ``port``), ``on_flush``, and ``blocked``
       while the outbox is full.
   * - ``forward``
     - ``forward`` per item, including its maps; ``flush``.
   * - ``map:<label>``
     - On the stage, a ``call`` of a map from submission until its result is
       ready, including the wait for resources. On the task worker or
       executor, its ``execute`` (with the GPU ids) and, on an executor, the
       ``lock_wait``.
   * - ``group_by``
     - ``start_child``: starting a worker or the child pipeline of a key.
   * - ``driver``
     - ``send``: the driver waiting for a source to accept an item; ``flush``.

Profiling costs one extra Ray call per map call, because a task worker sends
its events before the task returns. Timestamps are wall-clock times, so on a
cluster they are only as accurate as the clocks of the nodes are in sync.

Running on a cluster
--------------------

To use an existing cluster, connect before creating the runner:

.. code-block:: python

    import ray
    ray.init(address="auto")

Every node must be able to import the protocol's package and run its
external programs. When the backend starts a local Ray instance itself, it
adds the working directory and ``src`` to the ``PYTHONPATH`` of the workers.

For ``min_vram`` claims, the backend probes the memory of the cluster's GPUs
with ``nvidia-smi``. Pass ``RayBackend(gpu_memory=[...])`` to set it
explicitly.

Lower-level use
---------------

The Ray backend also runs plain streaming graphs, without a protocol:

.. code-block:: python

    from scipion_bridge.backend.ray import RayBackend
    from scipion_bridge.core.streaming import Pipeline

    backend = RayBackend(queue_size=4, buffer_size=8, profile=True)
    with Pipeline.from_sink(sink, backend=backend) as pipeline:
        pipeline.send(particles=batch)
    pipeline.close()

``RayBackend`` accepts the buffering and profiling options of the runner, plus
``parameters`` (values served to ``Field.value`` in the workers),
``gpu_memory``, ``group_by_workers`` and ``init_ray`` (whether to start a
local Ray instance if none is running).
