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

### Tooling
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
