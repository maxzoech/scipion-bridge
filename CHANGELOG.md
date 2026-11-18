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

### Element ids
- Every element of a Set, and every Struct, can carry an element id, a
  column of the storage that is not part of the schema (`id_column.py`,
  `identity.py`, internal). Ids are compact segments (a namespace plus a
  range or positions), follow the rows through views, `concat`, `astype`,
  row writes, `Set([...])`, Struct fields and Collection slots, and survive
  pickling.
- Resolvers set element ids: particles of input stacks are `(stack, row)`
  (the stack's path, size and mtime), particles of Scipion sets
  `(table, item id)`. Particles of temporary stacks have none.
- On Ray, every item between stages carries a deterministic 16-byte item id
  (`core/streaming/lineage.py`): sources name items by their element ids,
  maps by their parent, accumulators by the items since their last emission
  and their previous emission. Rows a stage creates get element ids from the
  item they are emitted in. Ids survive spills, phase boundaries and
  `group_by`; `RayCompiledPipeline.received_ids()` lists the ids of a sink.
- Run stores (`core/streaming/run_store.py`): with `run_store=FileRunStore(path)`
  (`RayPipelineRunner`, `RayBackend`) or `--run-dir PATH`, a run records its
  manifest (protocols and fingerprints, parameters, inputs, chunking), its
  stages with their `stage_fp`, and a lineage record per item a stage made
  (Parquet, written in the background every 2000 records or 30 s, and at
  every FLUSH). Stats have a `lineage_s` column. `LineageGraph` answers
  `ancestry(item_id)` (input files, stages) and
  `trace_element(namespace, position)` (an input row, or the item a stage
  created the element in).
- A lineage viewer: `python -m scipion_bridge.core.streaming.report.lineage_view RUN_DIR`
  serves a page browsing the run, its stages (with the share of input rows
  their items hold), the outputs, the ancestry of every item, the members of
  every class down to their input file and row, every item holding an
  element, and the rows read from every input file. `LineageGraph` is
  columnar and indexes element ids by namespace: a run of a million
  particles (435K records, 57 MiB) loads in about 4 s and answers every page
  in under 0.3 s (`benchmarks/bench_lineage.py`).
- The lineage viewer takes a directory of runs, like TensorBoard's
  `--logdir`: it lists every run below it (running ones marked, new ones
  appear) and browses the one picked; its API is `/api/runs` and
  `/api/run/<name>/...`.
- Sinks commit: with a run store, a sink records a commit per item it wrote
  (`commits/` of the run directory), and a run continuing in the store of an
  earlier run skips the items committed there that it receives again with
  the same ids (`skipped` in the stats). Ids follow arrival order, so this
  needs the order of the earlier run (one loader, no order-dependent stage
  before the sink); skipping by input rows comes with reruns from
  checkpoints. A run store of a run with other protocols, parameters, inputs,
  chunking or stages is refused (`RunStoreMismatchError`).
- `CommittingSinkWriter`: writers that undo the writes no commit covers
  after a crash. `TensorStoreSinkWriter` is one: it records the rows of its
  arrays and cuts them back. `StoreSinkWriter(path)` keeps every item in a
  file named by its id.
- Sinks can check the element ids they write and warn about, or refuse,
  elements written a second time: `RayPipelineRunner(unique_elements=...)` or
  `--unique-elements warn|raise` for the sink the runner attaches, and
  `write_to(writer, unique_elements=)` in pipelines built by hand.
- Sinks keep no state across items: state belongs in `scan` or `accumulate`
  stages, where lineage and checkpoints see it (Sinks guide).
- The lineage viewer marks the items sinks wrote and counts them per sink.
- `stream.checkpoint(name)` and `stream.tap(name)` keep the items passing
  through in the run store, by item id, while the stream goes on
  (`checkpoints/` of the run directory, complete once FLUSH passed them).
  Every protocol ends with a checkpoint of its outputs
  (`checkpoint(<Protocol>)`); `--no-checkpoint-outputs` leaves them out.
  Without a run store they are left out of the pipeline, keeping the labels
  of the other stages. Checkpoints are the points reruns will replay. Rows
  read from input files are kept as references to the files, not as bytes
  (`StoreSinkWriter` too); rows computed in the pipeline as bytes.
- **Breaking:** `Op.checkpoint(writer)` (a side sink) is replaced by
  `checkpoint()`/`tap()`, whose writer the backend provides; to write to a
  writer of your own on a side branch, use `write_to(writer)` and pass the
  sink to `Pipeline.from_sink`.
- The lineage viewer lists the checkpoints of a run and their items, and
  previews a kept item: its fields, small values and thumbnails of 2D images
  (e.g. class averages).
- The lineage viewer lists the items the sinks wrote as the outputs of a
  run, and the items nothing was made from that no sink wrote as its dead
  ends (often a branch leading nowhere), instead of mixing both.
- `B.astype(value, T)` changes the type of a Set or Struct without copying
  its columns, as `Op.astype` does for a stream: e.g. dropping the pixels of
  the members of a class keeps their element ids and file references.
- Arrays with an axis of size 0 (e.g. the members of an empty class) no
  longer kill the process (SIGFPE in PyArrow) when converted to Arrow or
  pickled.
- The preview of a kept item shows unset fields (e.g. the CTF of a class
  average) as unset instead of failing.
- The lineage viewer works on a single run directory in browsers: the run is
  named after its directory instead of `.`, which browsers dropped from
  URLs, and links into a single run no longer jump to its outputs.
- The lineage viewer draws the graph of the pipeline (the runner records its
  report in the run directory) and overlays the ancestry or descendants of an
  item, or the items holding an element: items per stage as badges, and the
  edges and pass-through stages (`group_by` routers, the sources of their
  children) they went along.
- Element ids of short runs (rows of small stacks, members of classes) are
  stored as mixed segments, keeping the segments of a column a fraction of
  its elements. Maps that keep the element ids of their item record
  `inherited` instead of repeating them.
- `Set.tag(name, discriminator)` and `Struct.tag(...)` give elements derived
  from others (e.g. augmented views) ids of their own, linked to theirs.
  `B.ids_rng(elements, salt)` returns random generators seeded from the ids,
  for augmentations that repeat when a stage runs again.
- A Set whose elements hold nested Sets of elements with ids (e.g. a
  `Set[Class2D]` of particles a pipeline read) raises `NotImplementedError`;
  use a `Collection` (Known Issues).
- Sets assigned into Struct fields and Structs assigned into Collection
  slots keep the references of their columns (unread rows stay unread), and
  Structs and Collections pickle references like Sets, so spills behind a
  classification write references instead of pixels.
- Fingerprints and `stage_fp` use `xxh3_128` instead of SHA-256
  (`core/utils/hashing.py`), so all fingerprints change once.

### Indexing of nested fields
- Fixed: selecting rows of a struct field after an outer selection selects
  rows of the Set: `s[1:]["ctf"][:1]` is one row (it was three),
  `s[1:]["ctf"][[1, 0]]` and `s[1:]["ctf"][0]` no longer raise.
- Fixed: assigning a Struct with a nested Set into a preallocated Set
  (`groups[i] = Group(...)`) writes its members; it raised for more than one
  member and silently dropped the members axis for one.
- Fixed: a scalar of a member of a ragged nested Set
  (`groups[1].members[0].value`) reads as a number.
- Views over the members of every element of a Set (`groups["members"]`)
  select along the members of each element: `groups["members"][:2]` is the
  first two members of every group, also after a full outer slice
  (`groups[0:]["members"][:2]`, which selected the first two groups before).
  They read ragged columns and accept batch writes. They have no single
  length: `len()`, iteration and `bool` raise `TypeError`, and negative
  indices raise `ValueError`.
- **Breaking:** `KeyPath` records which components open an axis (the root of
  a Set and nested Set fields): `append(name, *, axis)` requires the keyword,
  and a component without an axis has the index `None` (`IndexType` includes
  `None`; its string omits the brackets, `root[1:3].ctf`). Narrowing selects
  along the innermost axis; `axis_indices()` and `spans_several_axes()` are
  new.

### Streaming
- `Op.astype(T)` changes the type of the Sets or Structs of a stream
  without copying their columns: fields `T` adds stay uninitialized, fields
  it lacks are dropped, and fields of both must match in kind, dtype and
  shape. Columns are shared read-only and references to input files stay
  references, so a spill after it does not write them. It lowers to a
  fusable map: after `chunk` it runs inside the chunk stage
  (`chunk(256) → astype(Set[FlexParticle])`). Casting Collections raises
  `NotImplementedError`. The provenance warning for copied values now points
  to it.
- A whole-column write of the leaf of a nested Set accepts ragged data.

### Documentation
- New page *Known Issues* (`user_guide/known_issues.rst`): Sets of structs
  with nested Sets of different sizes cannot be pickled, the capacity of
  nested Sets is not checked on whole-column writes, Sets assigned into
  Struct fields or Collection slots lose their references (and reports do
  not flag it), Struct specialization is not implemented, and `astype` does
  not cast Collections. The two bugs
  have strict xfail tests.
- Lowering ends with an operator-fusion pass (`core/streaming/fusion.py`):
  maps marked `IRMap(fusable=True)` (cheap, pure, without resources; only
  set by operators of the library) run inside the stage upstream of them,
  or else the map downstream of them, instead of as a Ray task of their
  own. The fused stage keeps the host's fingerprint; its name lists the
  operators it runs (`chunk(256) → astype(...)`), and `IROp.fused` and
  `ReportNode.fused` hold their titles. A `FusionPolicy` decides on every
  legal move; `lower(nodes, fusion=NoFusion())` keeps every stage.
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
  (`wait_s`, `wait_max_s`: in its mailbox, for an executor, for the other
  call in flight on it, or in Ray's queue), when the stage received its
  first item and finished processing its last (`first_in_t`,
  `last_done_t`), and its `capacity` (1 for a stage, the executors of a
  pool map). Maps account the time their calls executed, as measured by
  the executor or task running them (returned next to the result), not the
  time from submitting a call to its result.
- The stats layer derives `throughput` (items processed per second, also of
  sinks), `s_per_item`, `busy`, `utilization` (`busy` over `capacity`) and
  `wait_mean_s` (`throughput_of`), colours by `utilization`, lists the
  stages in a *Throughput* panel and names the most utilized stage whose
  items wait as the bottleneck; maps running as Ray tasks have no capacity
  and are not compared. The runner's table shows items/s, busy, utilization
  and `wait_s`. *Colour by* in the HTML report offers every numeric
  annotation of a layer.
- After a run, the GPU maps of the placement layer name the GPUs by the
  devices the executors ran on (`nvidia-smi` numbering). `--explain` and the
  GPU memory of claims count only the GPUs `CUDA_VISIBLE_DEVICES` shows.

### Provenance
- Columns of Sets can reference rows of a source (an input file) instead of
  owning them, internally to the type system: their public API and behaviour
  are unchanged. A reference is read once and cached, any write makes the
  column owned, and views and `concat` keep references. Every row gets an
  xxh3 digest when it is read, and rereading detects changed inputs exactly
  (`StaleReferenceError`). Pickling sends unread references without their
  rows; spilling can drop the rows of verified references.
- New dependency: `xxhash`.
- The particle resolvers (RELION and Xmipp stacks, MRC and SPIDER) make
  `pixels` reference the images of the stack instead of reading them: they
  are read on first access, with the same values. Errors of the stack
  (volumes, truncated files) are still raised when resolving, from the
  header. Temporary stacks are read at once. Loaders (and the runner for a
  single input) read the pixels before pushing a chunk, so files are still
  read by the parallel loaders, and stages receive pixels as read references.
- Spilling writes reference columns as references to their rows when the
  stage can read the files again, unchanged (checked by size, mtime, and on
  reading by the row digests): a phase boundary in front of a particle
  stage spills metadata instead of images. `SCIPION_STREAM_SPILL_REFERENCES=0`
  spills the bytes. Items sent between stages always carry their rows.
- `StageStats` gain `spill_bytes` and `reread_s`.
- References stay as small as their rows when Sets are concatenated and
  sliced again and again (`Op.chunk`, the loaders): before, every batch
  carried the references of all earlier ones, which made transfers slower
  over a run and could fail to pickle.
- Reports have a `provenance` layer: per stage, the bytes per item that are
  references to input files, read references or owned; where files are read
  (`read_here`) and where columns lose their reference (`computed_here`,
  with red edges and a warning, which says when the values are only copied;
  that check hashes the columns and only runs with `__debug__`).
  The panels *Key paths* and *Spill estimate* show the state of the
  referenced paths at every stage and the bytes a spill writes at every
  phase boundary. Stages sample their first 3 items in and out.

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
