# Changelog

## Unreleased (branch `basic-ray-streaming`, 2026-10-01 to 2026-10-30)

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
