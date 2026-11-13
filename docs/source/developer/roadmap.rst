Roadmap: scheduling, provenance and checkpointing
=================================================

This article records the plan for three features of the Ray backend: automatic
scheduling, data provenance and checkpointing. It lists which steps are done,
the design of the open ones, and why they come in this order. Steps marked as
planned are designs, not implemented behavior; check the code and the
changelog before relying on them.

.. contents:: On this page
   :local:
   :depth: 2

Motivation
----------

The 2D-classification pipeline
``RayFoundationModelInference() | RayClusterProtocol() | RayRefineClassesAlignPCAJax()``
was written the way a user thinks about the problem. Tuning made it 1.8 to 2.1
times faster, but most of the gains came from hand-written decisions inside
the protocols:

* switching alignment to ``LONG_RUNNING`` to avoid a fresh process per chunk;
* partitioning the batches in the protocol (a hash of the key, ``group_by`` and
  ``unkey``) to spread inference over several GPUs;
* keeping inference and alignment on separate GPUs, and choosing the number of
  workers by hand, with claims that had to add up to exactly the GPUs of the
  cluster, or a process waited for a GPU and stalled the stream;
* raising the batch size so the ``group_by`` router had fewer items to route;
* reading MRC stacks sequentially instead of through a memory map.

None of this belongs in a protocol. The goal is that the plain three-line
pipeline reaches the same result on its own: **protocols say what they need,
and the backend decides how many copies run and where.** Two more needs sit on
top of that: every run should be traceable (which inputs, protocols and
parameters produced an output), and a killed run should resume without
redoing committed work.

Architecture
------------

The plan separates the logical plan, which protocols write, from the physical
plan, which the backend derives:

.. code-block:: text

    Protocol layer (API unchanged)       IR (+ metadata)                       Ray backend
    @resources(gpus/min_vram/task)  →  IRMap.compute, stage_fp             →  executor pools per compute group
    .map/.chunk/.group_by/.astype/.apply   IRDemux.keys (planned)              placement planner (planned)
    sealed protocol instances                                                   data path: references, push_many (planned)
                                                                                checkpoints: barriers, commit log (planned)
                                                                                runtime controller (planned, optional)

A compute group (one protocol) runs on a **pool of executor replicas** on GPUs
the backend chooses. The number of replicas and their placement are decisions
of the backend; protocols can still pin them with ``replicas=n``.

The three features share foundations:

* **Sealing** makes maps stateless, so they can be replicated and moved. It
  also makes the fingerprint of a stage stable, which lineage, checkpoints and
  profiles of earlier runs need.
* **Key-path provenance** gives the cost model its byte terms, gives transport
  and spilling their references, and gives checkpoints compact snapshots.
* **One barrier and ordering mechanism** (``FlushSignal``, and a checkpoint
  barrier): transport batching must never reorder them against items.

Status
------

.. list-table::
   :header-rows: 1
   :widths: 6 44 14 36

   * - #
     - Step
     - Status
     - Depends on
   * - 1
     - Sealed protocols and stage fingerprints (``stage_fp``)
     - Done, except ``MethodHandle`` (see below)
     - —
   * - 1b
     - Ship the map functions once (function tables)
     - Done
     - 1
   * - 2
     - Executor pools: ``replicas``, warm-up, ephemeral GPU pools, pools
       shared by ``group_by`` children
     - Done
     - 1, 1b
   * - 3
     - Placement planner and placement groups
     - Done, except phases inside a ``group_by`` template
     - 2
   * - 4
     - Provenance: key paths, ``.astype``, ``.apply``, storage domains
     - In progress: reference columns in the struct storage, and particle
       resolvers referencing their stacks, done
     - 1
   * - 5
     - Lineage
     - Planned
     - 1, 4
   * - 6
     - Checkpointing: manifest and commit log, barrier snapshots, inspection
     - Planned
     - 4, 5
   * - 7
     - Cost model, lazy and local I/O
     - Planned
     - 4
   * - 8
     - Batched transport, profile-guided sizing
     - Planned
     - 6, 7
   * - 9
     - Runtime controller
     - Planned, optional
     - 3, 7

Steps 4 and 5 only need step 1, so they can run in parallel to steps 2 and 3.
Checkpointing (6) comes before the cost model (7) because it is about
correctness: later performance work should be tested against a backend that
can already resume. Batched transport (8) is easier to design once it is known
which barriers it must keep in order.

Done
----

Sealed protocols and fingerprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A protocol is sealed once ``get_pipeline()`` has built its pipeline: attributes
cannot change, mutable containers are rejected and arrays become read-only (see
:doc:`/user_guide/protocols/declarations`). This is what makes replication
safe. ``Protocol.fingerprint()`` digests the code, parameter values and sealed
state of a protocol, and every stage carries a ``stage_fp`` in the IR.

Not done: ``MethodHandle``. The idea was to replace bound methods by small
handles at lowering. The function tables below achieve the main part of it
(no protocol per call) on the Ray backend only; handles for every backend are
still open.

Function tables
^^^^^^^^^^^^^^^

A pipeline puts the functions of its maps, its parameters and its profiling
setup in the object store once. Calls carry a reference and a label, and every
worker process unpacks a table once (see :doc:`ray_backend`). A call with a
1 MB attribute on the protocol went from about 397 to about 690 calls per
second.

Executor pools
^^^^^^^^^^^^^^

* ``@B.resources(..., replicas=n)`` gives a protocol a pool of ``n`` executors
  that its maps share; calls are spread over them and results keep their order.
* The executors of long-running pools start when the pipeline is compiled and
  build the protocol's process-scope resources before the first item.
* Ephemeral calls claiming GPUs run on pools too, which start executors on
  demand and stop them after an idle time (``ephemeral_idle_s``). By default,
  the ephemeral pools of a pipeline share the free GPUs in equal parts.
* Long-running pools that do not fit the cluster at once run in phases (see
  the placement planner below).
* The children of ``group_by`` share the pools of their protocol, so
  ``group_by(workers=n)`` only partitions keys.

Planned
-------

Placement planner (step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Done:** the planner (``plan_placement``) and its ``placement`` layer in
pipeline reports (``--explain``, ``--explain-gpus``); pools get the replicas
it plans. Phases run: pools release their GPUs when FLUSH has passed their
own stages, later phases wait for the earlier pools on their GPUs (gates),
and the stage behind a boundary spills without bound. Phase boundaries are
where checkpointing will put its barriers (step 6), and what provenance
makes cheap to spill (step 4). The plan is enforced with a placement group
per long-running pool, a bundle per slot pinned to its node, reserved while
the pool holds its GPUs; ``placement="ray"`` leaves placement to Ray.
Open: phases inside a ``group_by`` template; pinning ephemeral executors;
the GPU memory report below.

What was planned, and what changed when it was built:


A module ``backend/ray/placement.py`` of pure functions, testable in-process.
Input: every compute assignment of the graph, including ``group_by``
templates, the GPUs of the cluster and, later, costs. Output: for every
group, the slots (GPU bundle and share) of its replicas. Rules, in order:

1. **Gang-schedule.** The replicas of a pool hold their GPUs at the same time,
   which prevents a process from waiting for a GPU another one holds. Pools
   that do not fit at once run in phases; only claims no phase satisfies
   fail to compile.
2. **Anti-affinity.** Different long-running groups do not share a GPU, because
   processes on one GPU time-slice between their CUDA contexts (a reference
   build took 65 s instead of 5 to 10 s). Fractional claims are filled with
   replicas of the same group.
3. **Leftover GPUs** go to the costliest group; before costs exist, GPUs are
   split in proportion to the claims.

The plan was to enforce it with one Ray placement group with a one-GPU
bundle per GPU. Phases need reservations that change hands at every
boundary, and a claim of several GPUs needs one bundle, so every
long-running pool reserves a placement group of its own instead. A mode
that leaves placement to Ray remains, for shared clusters (``placement="ray"``).
``replicas=n`` stays a manual override.
Executors would report their peak GPU memory, and the planner would warn when
it exceeds the claim.

Options for GPU sharing without dedicated GPUs, to be decided when the planner
is built: use NVIDIA MPS when a daemon runs (which also enforces memory
claims), start the daemon on request (never on GPUs of other users), or run
several groups on threads and CUDA streams of one host executor.

Data provenance (step 4)
^^^^^^^^^^^^^^^^^^^^^^^^

Provenance is tracked per key path of an item. A path is either a
**reference** (a proxy plus a slice of it, and the storage domain where it can
be read) or **computed** by the map of a compute group. References attach to
the existing slice-aware resolvers. Two new operations create exactly the
changes that provenance tracks:

* ``.astype(T)`` casts a struct or ``Set`` to ``T`` by copying the matching
  key paths; each copied path is a reference to its source, and no bytes move.
* ``.apply(path, pipeline)`` runs a sub-pipeline on an element and writes its
  result to ``path``. The sub-pipeline receives the whole element and returns
  only the value of ``path``, so "only ``path`` was computed" holds by
  construction, and the planner knows the bytes in and out exactly.

**Storage domains** keep this correct on several nodes. Every proxy carries a
domain (for example ``shared:/data`` or ``node:<id>``), nodes advertise the
domains they reach, and the planner decides where each reference is read: in
the consumer if it is inside the domain, otherwise the item is materialized at
the last stage inside it. Compilation fails if a stage would receive a
reference it cannot read.

**Spilling** would write only computed paths, and references whose domain does
not contain the stage's node. There is also a known bug to fix on the way: the
spill directory comes from the driver's temporary directory and is removed
only there, so spill directories on other nodes leak. Each stage actor should
create its directory in its own node's temporary directory and remove it when
it closes.

Lineage (step 5)
^^^^^^^^^^^^^^^^

Lineage records which inputs, protocols and parameters produced an output. The
unit is the chunk (a stream item), not the particle. An item id is the hash of
its source and input reference, or of the stage fingerprint, the parent ids and
the emission index. Ids travel in an envelope that stage actors remove before
user code runs, as ``Keyed`` and ``Tagged`` do. Every kind of IR node gets a
pure lineage policy: a map's parent is its input, ``.apply`` keeps the id and
adds a "computed by" edge, an accumulator's parents are the items consumed
since its last emission, and so on. A recorder, following the pattern of the
profiler, appends to a run-local Parquet file, and a query walks it back to
the input references, fingerprints and parameters.

Checkpointing (step 6)
^^^^^^^^^^^^^^^^^^^^^^

After sealing, only accumulators and the ``group_by`` router hold state; map
pools are rebuilt by warm-up. That bounds what a checkpoint captures.

* **Manifest and commit log** (stateless pipelines). A manifest written at
  compile time holds the run id, the stages with their fingerprints, the inputs
  as references and the parameters; resume is refused if the fingerprints
  differ. Sink writers record the ids and input references they durably
  wrote, data first and record second. An input is done once every sink
  reachable from it has committed it, and resume skips it.
* **Barrier snapshots** (``scan``, accumulators, ``group_by``). A checkpoint
  barrier flows like ``FlushSignal`` without flushing. Accumulators snapshot
  their state, the router its key assignment, sources their offset. State is
  stored with references kept as references. The store must be durable and
  visible from every node (a shared storage domain), or compilation fails.
  A snapshot is complete once every sink acknowledged the barrier.
* **Inspection.** ``Op.checkpoint(writer)`` outputs become readable while the
  stream runs, with a manifest of the committed ids.

Cost model and local I/O (step 7)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three unrelated cost signals exist today: resolver hop counts, proxy size
estimates that only size chunks, and GPU claims without throughput. The plan is
one module (``core/environment/cost.py``) with a per-element cost: bytes in
memory and transport, bytes read from a source (sequential or random), compute
seconds per resource class, and a network term. Sources of values: the bytes
per row of a schema, resolver edges priced in seconds (the hop count remains
the tie-breaker, so existing resolutions stay), and profiles of earlier runs
stored per stage fingerprint. The planner would balance the rates of the
stages, derive the resolution chunk size from the bytes, and decide how many
readers the source needs. With provenance, consumers can read references
themselves, which replaces the synchronous reader on the driver.

Transport (step 8)
^^^^^^^^^^^^^^^^^^

Item sizes stay with the protocol (``.chunk(n)``); the backend only batches
RPCs. A stage would send everything already waiting in its outbox in one
``push_many``, capped by bytes and by item count, without waiting for more
items, so an idle stream gains no latency. ``push_many`` never reorders
``FlushSignal`` or a checkpoint barrier against the items around it. The
``group_by`` router would compute keys for a whole batch at once.

Runtime controller (step 9, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A controller watching mailbox depths and busy fractions would move replicas
from idle groups to the bottleneck group, for example freed inference GPUs to
the alignment tail. Maps are stateless, so moves need no checkpoint; each move
rebuilds a model, so the controller acts rarely and with hysteresis. It sits on
top of the static planner.

Risks and open decisions
------------------------

* **State that sealing cannot see:** module globals, class attributes and
  mutable objects held by the protocol are covered by the documented contract.
  Hardening that was discussed and not built: a static check of method bodies
  at class definition, a check of what methods reach through closures and
  globals, and a digest of the protocol compared at FLUSH to detect writes
  that bypass the seal.
* **Placement groups reserve GPUs** while a pool holds them, which suits a
  dedicated pipeline and not a shared cluster; ``placement="ray"`` leaves
  placement to Ray there.
* **Storage domains must be configured** on multi-node clusters. Without them,
  every reference belongs to the driver's node, which is correct but
  materializes everything at the source. Checkpointing also needs a shared
  domain.
* **Overhead of lineage and barriers** must stay small; keep lineage per chunk,
  barriers periodic, and measure against the tuned pipeline.
* **Moving replicas at runtime** rebuilds models, so the controller stays
  conservative.
* **No per-test timeout in the suite:** a test that deadlocks hangs instead of
  failing. The pool tests showed this; ``pytest-timeout`` would help.

Verification targets
--------------------

For the open steps, tests follow the existing pattern: logic in-process first
(planner decisions on synthetic graphs, lineage policies, snapshot and
restore of accumulator state), and one Ray test per feature for what only
exists across actors (killing a pipeline midway and resuming it without
committed inputs running again or duplicate rows, lineage of a sink output
resolving to its input reference and stage fingerprints, GPU placement on the
two-GPU test cluster). End to end, the unchanged three-line pipeline runs at
100K and 1M particles against the tuned baseline, once without and once with
lineage and periodic barriers (overhead a few percent at most), and a 1M run
killed halfway must resume with the same output.
