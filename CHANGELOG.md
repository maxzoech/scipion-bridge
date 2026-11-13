# Changelog

## Unreleased

### Protocols
- Protocols are sealed once `get_pipeline()` has built their pipeline:
  setting or deleting an attribute raises `B.SealedProtocolError`, `list`,
  `dict`, `set` and `bytearray` attributes are rejected and NumPy arrays
  become read-only. State across items belongs in `.scan`, expensive objects
  in `B.Resource`.
- `get_pipeline()` runs `validate_protocol_configuration()` and `setup()` on
  every backend, before `steps()`. The Scipion backend no longer runs
  `setup()` as a step of its own. A pipeline can be built once per instance.
- Protocols have a `fingerprint()` of their code, parameter values and state;
  every stage of a protocol's pipeline carries a `stage_fp` in the IR.
- The standalone backend serves the declared defaults of parameters, as the
  Ray backend does for parameters without a value.
- `B.ValidationError` and `B.SealedProtocolError` are exported.

### Imports and public API
- `import scipion_bridge` no longer loads Ray or any backend (about 0.4 s
  faster): the core injects its services by name.
- `B.` is grouped by layer and adds the streaming API: `B.Source`, `B.Sink`,
  `B.Pipeline`, `B.SinkWriter`, `B.CallbackSinkWriter`.
- Removed from `B.`: `ParticleStackProxy`, `StarfileProxy` and
  `MRCStackProxy` (use `scipion_bridge.single_particle`), `ResourceProvider`
  and `DefaultResourceProvider` (use `scipion_bridge.core.environment`), and
  the module attributes `B.protocol` and `B.struct`.
- Every subpackage exports its API from its `__init__`: `core.environment` is
  new; `core.typed` exports the resolution and proxy API (except `resolve`,
  which would hide its module; use `B.resolve`); `core.streaming` no longer
  exports `Node`, `replace_node` and the op node classes; `core.struct` no
  longer exports `_BaseStorage`; `core.utils` no longer re-exports `Domain`;
  `core.protocol` exports `ChainError` instead of `ResourceScope`;
  `backend.ray` no longer exports its actor classes, `RayResourceCoordinator`
  and the alias `configure_ray_container`.

### Streaming
- Long-running protocols run on a pool of executors:
  `@B.resources(..., task=B.TaskType.LONG_RUNNING, replicas=n)` spreads the
  calls of their maps over `n` processes, in order. Compiling fails if the
  pools of a pipeline do not fit the cluster at once.
- The executors of long-running protocols start when the pipeline starts
  and build the protocols' process-scope resources (e.g. load models) before
  the first item.
- Compiling a pipeline on Ray only plans it and starts no actors, so
  `pipeline.report()` (with the placement of its pools) and `stats()` cost
  nothing on the cluster. The actors start with `Pipeline.start()`, when the
  pipeline is entered with `with`, or on the first `send()` or `flush()`.
  `RayBackend.compile_async()` is replaced by
  `RayCompiledPipeline.start_async()`.
- Calls of ephemeral protocols that claim GPUs run on a pool of executors
  instead of a new process per call, so models are loaded once per executor.
  Executors start on demand and stop after `RayBackend(ephemeral_idle_s=...)`
  seconds without calls; by default the ephemeral pools of a pipeline share
  the free GPUs in equal parts.
- The children of a `group_by` run on the pools of their protocols instead of
  starting executors of their own: `group_by(workers=n)` only partitions the
  keys, and the number of GPU processes is the protocol's `replicas`. An
  executor runs one call of each map of its protocol at a time, however many
  children share it. The pipeline releases its pools after a flush.
- Map calls on Ray no longer carry their function: a pipeline puts the
  functions of its maps, its parameters and its profiling setup in the object
  store once, and every worker process unpacks them once. A protocol's map
  methods no longer ship the protocol with every call (1 MB of state was 1 MB
  per call).
- Leaving the `with` block of a `Pipeline` closes it after the flush, and
  `RayPipelineRunner.run()` closes its pipeline: their Ray actors were never
  terminated, since the stages hold references to each other, and a test
  session accumulated over a hundred live actors. `stats()` remains available
  after closing; `send()` and `flush()` on a closed pipeline raise.

### Pipeline reports
- `--explain [PATH]` on every protocol launched with `RayPipelineRunner`
  prints the stages of its pipeline and writes an HTML report of its graph
  (and the graph in DOT), without starting Ray; `--report PATH` writes it
  after a run, with the stats of every stage. `Pipeline.report()` and
  `RayPipelineRunner.report()` return the report.
- Reports (`scipion_bridge.core.streaming.report`) are the graph of a
  pipeline annotated by layers, rendered as text, DOT, SVG or a
  self-contained HTML page; laid out with Graphviz if installed.
- The stages of the children of a `group_by` are labelled under their
  `group_by`: `1:group_by[a]/2:chunk(2)` instead of `group_by[a]:2:chunk(2)`,
  also in the names of their actors.
- `ComputeAssignment.name` names the protocol of a group.
- Stage stats record how long items waited before a stage took them
  (`wait_s`, `wait_max_s`: in its mailbox, or for an executor) and when the
  stage received its first item and forwarded its last (`first_in_t`,
  `last_out_t`). The `process_s` of a map on an executor pool no longer
  includes the wait for an executor.
- The stats layer derives `throughput`, `s_per_item`, `busy` and
  `wait_mean_s` (`throughput_of`), colours by `busy`, lists the stages in a
  *Throughput* panel and names the bottleneck; the runner's table shows
  items/s, busy and `wait_s`. *Colour by* in the HTML report offers every
  numeric annotation of a layer.

### Provenance
- Columns of Sets can reference rows of a source (an input file) instead of
  owning them, internally to the type system: their public API and behaviour
  are unchanged. A reference is read once and cached, any write makes the
  column owned, and views and `concat` keep references. Every row gets an
  xxh3 digest when it is read, and rereading detects changed inputs exactly
  (`StaleReferenceError`). Pickling sends unread references without their
  rows; spilling can drop the rows of verified references.
- New dependency: `xxhash`.

### Placement
- A placement planner (`backend/ray/placement.py`) decides the executors of
  every pool: long-running protocols get GPUs of their own where the cluster
  has enough (with a warning where they must share), the executors of a
  protocol fill a GPU before the next, and ephemeral protocols split the GPUs
  the others leave. Reports and `--explain` show the plan; `--explain-gpus`
  plans for another cluster.
- `@B.resources(..., replicas="auto")` sizes a long-running pool to the GPUs
  the other long-running protocols leave.
- Compiling fails for claims no cluster state satisfies, now also for a claim
  above one GPU that is not a whole number.
- Protocols that do not fit the cluster at once run in phases: a later phase
  starts once the pools of the earlier ones have flushed, and its items wait
  on disk in front of it, while backpressure stays bounded within a phase.
  Long-running pools release their GPUs as soon as FLUSH has passed their
  own stages. Before, such pipelines failed to compile, or hung when an
  ephemeral protocol found no GPU beside the long-running ones.
- The plan is enforced: every long-running pool reserves the GPUs planned
  for it in a Ray placement group, pinned to its node, from its first
  executor until its release, so that processes of other protocols cannot
  land on them (before, Ray could put two models on one GPU while another
  stayed idle). A pool with a GPU to itself reserves it whole. Ephemeral
  pools reserve nothing. `RayBackend(placement="ray")`, the runner's
  `--placement ray` or `SCIPION_STREAM_PLACEMENT=ray` leave placement to
  Ray, for clusters shared with other jobs. Requires Ray 2.49.
- A pool of a later phase waits only for the pools of earlier phases on its
  GPUs (with `placement="ray"`, for all of them). Reports show what every
  pool reserves and waits for.
- Long-running pools with `replicas="auto"` grow by whole executors and
  leave room for one executor of every ephemeral protocol of their phase;
  before, a share below one executor still took a whole GPU, and an
  ephemeral protocol could be left without one.
- A map behind a phase boundary runs on an actor of its own; the stats of
  such maps (and of merging maps) include the buffering of their actor.
- Every stage spills into the temporary directory of its own node and
  removes it when the pipeline closes; before, spill directories on other
  nodes than the driver's were left behind.

### Tooling
- The actors of the Ray backend moved from `backend/ray/backend.py` to
  `backend/ray/actors.py`. Ray workers wire a `RayWorkerContainer`
  (`backend/ray/worker_env.py`), without the streaming backend.
- AGENTS.md: `match` is for destructuring and dispatch on types; booleans,
  `None` checks and tuples of conditions use `if`/`else`.
- Static tensor columns are read from their Arrow storage, so null cells of a
  sparse `Collection` load with pyarrow 26, whose `to_numpy_ndarray` rejects
  them.
- CI runs the Ray tests in one process, with one Ray cluster, and the test
  cluster's object store is capped at 512 MiB, against out-of-memory kills on
  the hosted runner. Tests marked `xdist_group("timing")` are marked `timing`
  automatically.

## Branch `basic-ray-streaming` (2026-10-01 to 2026-10-30)

### Streaming operations
- `chunk`, `collect`, `flatten`, `map_element`, `combine_latest`, `scan` and
  `merge`. `scan(..., on_flush=)` emits a final item at the end of the stream;
  `merge` joins two branches of a pipeline.
- `group_by` child pipelines are shared between keys.
- Errors surface on the next `send` instead of on the flush signal.
- Same-named sources are unified and Ray actors are named.

### Ray backend
- Run pipelines on Ray, including a CLI and shared cluster resources.
- `@resources` schedules protocol stages by GPUs and CPUs; `min_vram` claims
  fractional GPUs sized for the smallest GPU in the cluster.
- Mailboxes, spilling and profiling; GPU map calls run in fresh processes.
- `clone_ir` and child-pipeline hooks (name, abort targets, source handles).
- Lists of input files load in parallel into a pipeline.
- Protocols can be combined.

### Types and data
- `Set`, `Struct` and `Collection` pickle generically as Arrow; any iterable
  can be flattened.
- Type resolution is decomposed with `ComposedResolver` and slice forwarding;
  large files resolve lazily and iteratively; proxies estimate chunk sizes.
- Particle stack resolvers, proxy group extensions and protocol dtype
  propagation.
- MRC stacks are read with sequential reads. Particles extracted by Xmipp are
  read from `.xmd` metadata with MRC or SPIDER stacks.
- `RootEngine.is_initialized` caches the field prefixes it queries.

### Tooling and documentation
- The test suite runs faster and the Ray tests run in parallel in CI.
- Documentation for the streaming API, backends, structs and the Xmipp
  particle stacks.
- `benchmarks/bench_mrc_read.py` compares MRC chunk readers.
- Unused imports and flake8/black findings removed.
