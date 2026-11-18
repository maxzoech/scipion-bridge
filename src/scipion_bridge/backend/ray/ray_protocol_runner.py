from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Sized
from dataclasses import dataclass
import datetime
import glob
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import ray

from ...core.environment.compute import numpy_hugepage_env
from ...core.struct import references
from ...core.struct.set import Set
from ...core.typed.resolve import ComposedResolver, find_resolver
from ...core.protocol.protocol_base import ChainedProtocol, Protocol
from ...core.protocol.run_config import (
    add_input_arguments,
    add_parameter_arguments,
    argname,
    check_required,
    default_parameters,
    inputs_of,
    parameter_values,
    parameters_of,
)
from ...core.streaming.backend import StageStats, throughput_of
from ...core.streaming.ir import IROp
from ...core.streaming.node import lower
from ...core.streaming.pipeline import Pipeline
from ...core.streaming.run_store import (
    FileRunStore,
    RunStore,
    RunStoreMismatchError,
)
from ...core.streaming.report.model import Layer, PipelineReport, report_of
from ...core.streaming.report.render import render_dot, render_html, render_text
from ...core.streaming.sink import Sink
from ...core.streaming.sink_writer import SinkWriter, UniqueElements
from ...core.streaming.spill import SpillStoreFactory, pickle_spill_store
from .backend import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_SPILL_THRESHOLD,
    PlacementMode,
    RayBackend,
    RayCompiledPipeline,
    read_gpu_memory,
)
from .placement import ClusterSpec, explain_placement
from .container import configure_ray_env
from .loader import LoaderStats, RayInputLoader, expand_pattern, shard


def _format_stats(stats: Dict[str, StageStats]) -> str:
    """Format cumulative per-stage metrics as a table, with the throughput,
    busy, utilization and waits of the stages that were active."""
    header = (
        f"{'stage':<40} {'in':>6} {'out':>6} {'items/s':>8} {'busy':>6} {'util':>6} "
        f"{'idle_s':>8} {'process_s':>10} {'wait_s':>8} {'blocked_s':>10} "
        f"{'emit_s':>8} {'buffered':>8} {'spilled':>8} {'fetch_s':>8} "
        f"{'spill_s':>8} {'lineage_s':>9} {'skipped':>8}"
    )
    rows = [
        f"{label:<40} {s.items_in:>6} {s.items_out:>6} "
        f"{_derived(s, 'throughput', 8, 2)} {_derived(s, 'busy', 6, 2)} "
        f"{_derived(s, 'utilization', 6, 2)} "
        f"{s.idle_s:>8.2f} {s.process_s:>10.2f} {s.wait_s:>8.2f} "
        f"{s.blocked_s:>10.2f} {s.emit_s:>8.2f} {s.buffered_peak:>8} "
        f"{s.spilled:>8} {s.fetch_s:>8.2f} {s.spill_s:>8.2f} {s.lineage_s:>9.2f} "
        f"{s.skipped:>8}"
        for label, s in stats.items()
    ]
    return "\n".join([header, *rows])


def _derived(stats: StageStats, metric: str, width: int, digits: int) -> str:
    """A metric of ``throughput_of`` as a column, ``-`` for inactive stages."""
    value = throughput_of(stats).get(metric)
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.{digits}f}"


def _format_loader_stats(stats: Sequence[LoaderStats]) -> str:
    """Format the metrics of the input loaders as a table."""
    header = (
        f"{'loader':<40} {'files':>6} {'chunks':>6} {'items':>8} "
        f"{'read_s':>8} {'push_s':>8}"
    )
    rows = [
        f"{f'loader {i}':<40} {s.files:>6} {s.chunks:>6} {s.items:>8} "
        f"{s.read_s:>8.2f} {s.push_s:>8.2f}"
        for i, s in enumerate(stats)
    ]
    return "\n".join([header, *rows])


def _expand_input(value: str) -> List[Path]:
    """The files matching a glob pattern, or the path itself."""
    match glob.has_magic(value):
        case True:
            return expand_pattern(value)
        case False:
            return [Path(value)]


def _parse_inputs(values: List[str]) -> Union[Path, List[Path]]:
    """A command line input: one path, or the files of several paths and patterns.

    A single path is resolved on the driver; a list is resolved by the loaders.
    """
    match values:
        case [value] if not glob.has_magic(value):
            return Path(value)
        case _:
            return [path for value in values for path in _expand_input(value)]


def _env_int(name: str, value: Optional[int]) -> Optional[int]:
    """The environment variable ``name`` as an int if it is set, else ``value``."""
    match os.environ.get(name):
        case None:
            return value
        case override:
            return int(override)


def _env_limit(name: str, value: Optional[int]) -> Optional[int]:
    """Like ``_env_int``; the value ``none`` sets no limit (``None``)."""
    match os.environ.get(name):
        case None:
            return value
        case override if override.lower() == "none":
            return None
        case override:
            return int(override)


def _env_spill_store(value: Optional[SpillStoreFactory]) -> Optional[SpillStoreFactory]:
    """Pickled spilling into ``SCIPION_STREAM_SPILL_DIR`` if it is set, else ``value``."""
    match os.environ.get("SCIPION_STREAM_SPILL_DIR"):
        case None:
            return value
        case directory:
            return pickle_spill_store(Path(directory))


def _env_profile(value: Union[None, bool, str, Path]) -> Optional[Path]:
    """Like ``_profile_path``; ``SCIPION_STREAM_PROFILE`` overrides ``value``."""
    return _profile_path(os.environ.get("SCIPION_STREAM_PROFILE", value))


def _profile_path(value: Union[None, bool, str, Path]) -> Optional[Path]:
    """The trace path of profiling.

    ``1`` or ``True`` profile into a timestamped file in the working
    directory, ``0`` or ``False`` turn profiling off.
    """
    match value:
        case None | False | "0" | "":
            return None
        case True | "1":
            return Path(f"ray_profile_{time.strftime('%Y%m%d-%H%M%S')}.json")
        case path:
            return Path(path)


def _env_log_level(name: str, value: int) -> int:
    """The level named by the environment variable ``name`` if it is set, else ``value``."""
    match os.environ.get(name):
        case None:
            return value
        case level:
            return logging.getLevelNamesMapping()[level.upper()]


def _default_sink_handler(outputs: Dict[str, Any]) -> None:
    """Default sink handler for compiled Ray pipelines."""
    pass


# What a run continuing in the run store of an earlier run must share with
# it: its sinks skip the items the earlier run committed.
_CONTINUED = ("protocols", "parameters", "inputs", "chunking")


def _check_continues(previous: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Refuse a run in a run store whose earlier run (``previous``, empty for
    a new store) computed something else.

    Raises:
        RunStoreMismatchError: For the parts of the manifest that changed.
    """
    if not previous:
        return
    # As the store keeps it: JSON values.
    stored = json.loads(json.dumps(dict(manifest), default=str))
    changed = [key for key in _CONTINUED if stored[key] != previous[key]]
    if changed:
        raise RunStoreMismatchError(
            f"The run store holds a run with other {changed}; its sinks would "
            "skip items of that run. Use a new run directory.",
        )


_DEFAULT_REPORT = "pipeline_report.html"


def _placement_mode(value: str) -> PlacementMode:
    match value:
        case "managed":
            return "managed"
        case "ray":
            return "ray"
        case _:
            raise ValueError(f"placement must be 'managed' or 'ray', got {value!r}.")


def _explain_requested(argv: Sequence[str]) -> bool:
    """Whether ``argv`` asks for ``--explain``, before it is parsed."""
    peek = ArgumentParser(add_help=False)
    peek.add_argument("--explain", nargs="?", const=True)
    known, _ = peek.parse_known_args(argv)
    return known.explain is not None


def local_cluster() -> ClusterSpec:
    """This machine as a cluster: its GPUs, as nvidia-smi reports them, and CPUs."""
    memory = tuple(read_gpu_memory())
    return ClusterSpec(
        node_gpus=(len(memory),),
        cpus=float(os.cpu_count() or 1),
        gpu_memory=memory,
    )


def _cluster_of(gpus: Optional[str]) -> Optional[ClusterSpec]:
    """The cluster of ``--explain-gpus``, e.g. ``4,4`` for two nodes of 4 GPUs,
    with the CPUs and GPU memory of this machine."""
    match gpus:
        case None:
            return None
        case _:
            local = local_cluster()
            node_gpus = tuple(int(count) for count in gpus.split(","))
            memory = min(local.gpu_memory, default=None)
            return ClusterSpec(
                node_gpus=node_gpus,
                cpus=local.cpus * len(node_gpus),
                gpu_memory=() if memory is None else (memory,) * sum(node_gpus),
            )


def _protocols_of(protocol: Protocol) -> List[Dict[str, str]]:
    """The protocols of ``protocol`` in order, with their fingerprints."""
    if isinstance(protocol, ChainedProtocol):
        return [*_protocols_of(protocol.first), *_protocols_of(protocol.second)]
    else:
        return [
            {
                "protocol": type(protocol).__qualname__,
                "name": protocol.name,
                "fingerprint": protocol.fingerprint(),
            },
        ]


def _write_report(report: PipelineReport, path: Path) -> None:
    """Write ``report`` to ``path`` as HTML and next to it as DOT."""
    path.write_text(render_html(report))
    dot = path.with_suffix(".dot")
    dot.write_text(render_dot(report))
    print(f"Pipeline report: {path.resolve()} (graph: {dot.resolve()})")


class RayPipelineRunner:
    """Ray pipeline runner for Scipion Bridge protocols.

    Precomputes resolvers for protocol inputs upon initialization,
    compiles the protocol streaming DAG into a Ray pipeline,
    and executes it on input chunks.
    """

    def __init__(
        self,
        protocol: Protocol,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
        origin_types: Optional[Dict[str, Type]] = None,
        intermediate_types: Optional[Dict[str, Type]] = None,
        sink: Optional[Union[Sink, SinkWriter, Callable[[Any], Any]]] = None,
        queue_size: int = 2,
        max_in_flight: Optional[int] = None,
        buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE,
        spill_threshold: Optional[int] = DEFAULT_SPILL_THRESHOLD,
        spill_store: Optional[SpillStoreFactory] = None,
        profile: Union[None, bool, str, Path] = None,
        profile_log_level: int = logging.INFO,
        num_loaders: int = 4,
        loader_cpus: float = 0,
        placement: PlacementMode = "managed",
        run_store: Optional[RunStore] = None,
        checkpoint_outputs: bool = True,
        unique_elements: UniqueElements = UniqueElements.OFF,
    ) -> None:
        """
        Every buffering option is overridden by an environment variable, see
        ``RayBackend`` for their meaning.

        Args:
            parameters: Values of the parameters by protocol name,
                ``{name: {parameter: value}}`` (see ``Protocol.name``), served
                to ``Field.value`` on the driver and inside every pipeline
                stage. Enums may be given by name. Parameters without a value
                resolve to the value given to the constructor of their
                protocol, else to their declared default.
            origin_types: Type each input is given as, by input name; ``Path``
                by default.
            intermediate_types: Type each input is resolved through, by input
                name, when several routes lead to its declared type; e.g.
                ``XmippParticleStackProxy`` for ``.xmd`` files.
            queue_size: Number of items each pipeline stage reads ahead
                (``SCIPION_STREAM_QUEUE_SIZE``).
            max_in_flight: Number of items on the routes out of a stage at
                once (``SCIPION_STREAM_MAX_IN_FLIGHT``).
            buffer_size: Most items waiting for a stage; beyond it, the stages
                upstream and ``run`` wait. ``None`` lets them wait without
                limit (``SCIPION_STREAM_BUFFER_SIZE``, ``none`` for no limit).
            spill_threshold: Number of waiting items a stage keeps in the
                object store before spilling; ``None`` never spills
                (``SCIPION_STREAM_SPILL_THRESHOLD``, ``none`` to never spill).
            spill_store: Creates the spill store of a stage. Pickles into
                ``SCIPION_STREAM_SPILL_DIR`` if that is set.
            profile: Path of the profiling trace, written while the pipeline
                runs; ``True`` writes a timestamped file in the working
                directory (``SCIPION_STREAM_PROFILE``, ``1`` for the
                timestamped file, ``0`` for no profiling).
            profile_log_level: Level of the profiling log of the workers;
                ``DEBUG`` logs every item
                (``SCIPION_STREAM_PROFILE_LOG_LEVEL``, e.g. ``debug``).
            num_loaders: Number of loader actors that resolve an input given
                as a list of values in parallel
                (``SCIPION_STREAM_NUM_LOADERS``).
            loader_cpus: CPUs reserved for each loader actor
                (``SCIPION_STREAM_LOADER_CPUS``).
            placement: ``"managed"`` reserves the GPUs the planner gives
                every long-running protocol; ``"ray"`` lets Ray place them,
                for clusters shared with other jobs
                (``SCIPION_STREAM_PLACEMENT``).
            run_store: Where the run records its manifest and the lineage
                of its items (``--run-dir``); ``None`` records nothing.
            checkpoint_outputs: Whether the run store keeps the outputs of
                every protocol, to inspect them and to rerun from
                (``--no-checkpoint-outputs`` turns it off).
            unique_elements: What the sink of the run does with an element it
                writes a second time: nothing, warn, or fail the pipeline
                (``--unique-elements``). A ``sink`` given as a ``Sink`` keeps
                its own setting.
        """
        self.protocol = protocol
        self.parameters = parameter_values(protocol, parameters or {})
        self.origin_types = origin_types or {}
        self.intermediate_types = intermediate_types or {}
        self.num_loaders = int(
            os.environ.get("SCIPION_STREAM_NUM_LOADERS", num_loaders)
        )
        self.loader_cpus = float(
            os.environ.get("SCIPION_STREAM_LOADER_CPUS", loader_cpus)
        )
        self.queue_size = int(os.environ.get("SCIPION_STREAM_QUEUE_SIZE", queue_size))
        self.placement: PlacementMode = _placement_mode(
            os.environ.get("SCIPION_STREAM_PLACEMENT", placement)
        )
        self.max_in_flight = _env_int("SCIPION_STREAM_MAX_IN_FLIGHT", max_in_flight)
        self.buffer_size = _env_limit("SCIPION_STREAM_BUFFER_SIZE", buffer_size)
        self.spill_threshold = _env_limit(
            "SCIPION_STREAM_SPILL_THRESHOLD",
            spill_threshold,
        )
        self.spill_store = _env_spill_store(spill_store)
        self.profile = _env_profile(profile)
        self.profile_log_level = _env_log_level(
            "SCIPION_STREAM_PROFILE_LOG_LEVEL",
            profile_log_level,
        )
        self.sink = sink
        self.run_store = run_store
        self.checkpoint_outputs = checkpoint_outputs
        self.unique_elements = unique_elements
        self._input_resolvers: Dict[str, ComposedResolver] = {}
        self._pipeline: Optional[Pipeline] = None
        self._ir_sinks: Optional[List[IROp]] = None

        self._precompute_resolvers()

    def _precompute_resolvers(self) -> None:
        """Precompute ComposedResolver instances for all protocol inputs."""
        config = self.protocol.configuration
        for name, field in config.inputs.items():
            target_type = field.dtype
            assert isinstance(target_type, type)

            origin_type = self.origin_types.get(name, Path)
            if origin_type == target_type:
                continue

            resolver = find_resolver(
                origin_type,
                target_type,
                intermediate=self.intermediate_types.get(name),
            )
            self._input_resolvers[name] = resolver

    def _compile_pipeline(self) -> Pipeline:
        """Compile protocol streaming DAG into an executable Ray pipeline."""
        check_required(self.protocol, self.parameters)
        configure_ray_env(parameters=self.parameters)
        backend = RayBackend(
            queue_size=self.queue_size,
            max_in_flight=self.max_in_flight,
            buffer_size=self.buffer_size,
            spill_threshold=self.spill_threshold,
            spill_store=self.spill_store,
            parameters=self.parameters,
            profile=self.profile,
            profile_log_level=self.profile_log_level,
            placement=self.placement,
            run_store=self.run_store,
            checkpoint_outputs=self.checkpoint_outputs,
        )
        return Pipeline.from_ir(self.ir_sinks, backend=backend)

    @property
    def ir_sinks(self) -> List[IROp]:
        """The IR of the protocol's pipeline and its sink, lowered once.

        Builds the pipeline of the protocol, which seals it, without starting
        Ray.
        """
        if self._ir_sinks is None:
            configure_ray_env(parameters=self.parameters)
            sink_node = (
                self.sink
                if isinstance(self.sink, Sink)
                else Sink(
                    self.sink or _default_sink_handler,
                    unique_elements=self.unique_elements,
                )
            )
            self.protocol.get_pipeline().op(
                sink_node,
            )
            self._ir_sinks = lower([sink_node])

        return self._ir_sinks

    def report(self) -> PipelineReport:
        """The report of the pipeline: its graph, and the stats of its stages
        once it was compiled. Render it with
        ``scipion_bridge.core.streaming.report``."""
        match self._pipeline:
            case None:
                return report_of(self.ir_sinks)
            case pipeline:
                return pipeline.report()

    def explain(self, path: Path, cluster: Optional[ClusterSpec] = None) -> None:
        """Print the stages of the pipeline and their placement on
        ``cluster`` (default: this machine), and write its report to ``path``
        (HTML) and next to it (DOT), without running it.

        A placement that cannot be satisfied on ``cluster`` is reported as a
        warning instead of the placement.
        """
        cluster = local_cluster() if cluster is None else cluster
        try:
            report = explain_placement(
                self.ir_sinks,
                cluster,
                managed=self.placement == "managed",
            )
        except ValueError as error:
            report = report_of(self.ir_sinks).with_layer(
                Layer(
                    "placement",
                    "Placement",
                    warnings=(f"The pipeline cannot run on {cluster}: {error}",),
                ),
            )
        print(render_text(report))
        _write_report(report, path)

    @property
    def pipeline(self) -> Pipeline:
        """Return the streaming Pipeline, compiling it on first access."""
        if self._pipeline is None:
            self._pipeline = self._compile_pipeline()

        return self._pipeline

    def set_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Replace the parameter values. Only possible before the pipeline is compiled."""
        if self._pipeline is not None:
            raise RuntimeError(
                "Parameters cannot be changed after the pipeline has been compiled.",
            )

        self.parameters = parameter_values(self.protocol, parameters)

    def set_profile(self, profile: Union[bool, str, Path]) -> None:
        """Profile the pipeline. Only possible before the pipeline is compiled."""
        if self._pipeline is not None:
            raise RuntimeError(
                "Profiling cannot be enabled after the pipeline has been compiled.",
            )

        self.profile = _profile_path(profile)

    @property
    def input_resolvers(self) -> Dict[str, ComposedResolver]:
        """Return the precomputed input resolvers."""
        return dict(self._input_resolvers)

    def _start_loaders(
        self,
        batched: Mapping[str, Sequence[Any]],
        *,
        chunk_size: Optional[int],
        target_bytes: Optional[int],
    ) -> List[Tuple[Any, ray.ObjectRef]]:
        """Start the loader actors of the inputs given as lists.

        Returns every loader with the reference to the result of its run.
        """
        compiled = self.pipeline.compiled
        assert isinstance(compiled, RayCompiledPipeline)

        loaders: List[Tuple[str, Sequence[Any], Any]] = [
            (
                name,
                values,
                RayInputLoader.options(
                    num_cpus=self.loader_cpus,
                    runtime_env={"env_vars": numpy_hugepage_env()},
                ).remote(
                    self._input_resolvers[name],
                    self.parameters,
                    compiled.profile,
                    f"loader:{name}:{i}",
                ),
            )
            for name, all_values in batched.items()
            for i, values in enumerate(shard(all_values, self.num_loaders))
        ]
        return [
            (
                actor,
                actor.run.remote(
                    values,
                    compiled.source_handle(name),
                    chunk_size,
                    target_bytes,
                ),
            )
            for name, values, actor in loaders
        ]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Resolve the inputs in chunks and stream them through the compiled pipeline.

        Inputs given as a single value are resolved on the driver and
        interleaved one chunk at a time. An input given as a list or tuple of
        values, e.g. the files matching a pattern, is resolved by
        ``num_loaders`` loader actors in parallel, which regroup the chunks of
        all values into chunks of the target size.
        """
        all_inputs = dict(inputs or {})
        all_inputs.update(kwargs)

        env_chunk_size = (
            int(os.environ["SCIPION_CHUNK_SIZE"])
            if "SCIPION_CHUNK_SIZE" in os.environ
            else None
        )
        env_target_bytes = (
            int(os.environ["SCIPION_TARGET_BYTE_SIZE"])
            if "SCIPION_TARGET_BYTE_SIZE" in os.environ
            else None
        )

        _DONE = object()

        streamed = {
            name: value
            for name, value in all_inputs.items()
            if name in self._input_resolvers
        }
        batched = {
            name: value
            for name, value in streamed.items()
            if isinstance(value, (list, tuple))
        }
        active_iterators = {
            name: self._input_resolvers[name].iter(
                raw_value,
                chunk_size=env_chunk_size,
                target_bytes=env_target_bytes,
            )
            for name, raw_value in streamed.items()
            if name not in batched
        }

        if self.run_store is not None:
            manifest = self.manifest(all_inputs, env_chunk_size, env_target_bytes)
            _check_continues(self.run_store.read_manifest(), manifest)
            self.run_store.write_manifest(manifest)
            self.run_store.write_report(self.report().to_json())

        t_start = time.perf_counter()
        total_chunks = 0
        total_items = 0

        # The loaders push into the actors of the pipeline.
        pipeline = self.pipeline.start()
        loaders = self._start_loaders(
            batched,
            chunk_size=env_chunk_size,
            target_bytes=env_target_bytes,
        )
        try:
            with pipeline:
                while active_iterators:
                    for name in list(active_iterators.keys()):
                        chunk = next(
                            active_iterators[name],
                            _DONE,
                        )
                        if chunk is _DONE:
                            del active_iterators[name]
                            continue

                        assert isinstance(chunk, Sized)
                        if isinstance(chunk, Set):
                            references.prefetch(chunk)

                        total_chunks += 1
                        total_items += len(chunk)

                        pipeline.send(
                            **{name: chunk},
                        )

                loader_stats: List[LoaderStats] = ray.get(
                    [run for _, run in loaders],
                )
        finally:
            for actor, _ in loaders:
                ray.kill(actor)

        total_chunks += sum(s.chunks for s in loader_stats)
        total_items += sum(s.items for s in loader_stats)

        elapsed = time.perf_counter() - t_start
        throughput = total_items / elapsed if elapsed > 0 else 0.0
        print(
            f"Pipeline complete in {elapsed:.2f}s "
            f"({total_chunks} chunks, {total_items} items, "
            f"{throughput:.1f} items/s)",
        )
        print(_format_stats(pipeline.stats()))
        if self.run_store is not None:
            # The graph again, with the stats of its stages.
            self.run_store.write_report(self.report().to_json())
        if loader_stats:
            print(_format_loader_stats(loader_stats))
        match self.profile:
            case None:
                pass
            case path:
                print(f"Profiling trace: {path.resolve()}")

    def manifest(
        self,
        inputs: Mapping[str, Any],
        chunk_size: Optional[int],
        target_bytes: Optional[int],
    ) -> Dict[str, Any]:
        """What a run of the pipeline on ``inputs`` computes: the manifest of
        its run store."""
        # The protocol is sealed once its pipeline is built.
        _ = self.ir_sinks
        return {
            "run_id": uuid.uuid4().hex,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "library_version": importlib.metadata.version("scipion-bridge"),
            "protocols": _protocols_of(self.protocol),
            "parameters": dict(self.parameters),
            # The values as given; the files the run read, with their size
            # and modification time, are in the records of its sources.
            "inputs": {
                name: {
                    "value": value,
                    "count": len(value) if isinstance(value, (list, tuple)) else 1,
                }
                for name, value in inputs.items()
            },
            "chunking": {
                "chunk_size": chunk_size,
                "target_bytes": target_bytes,
                "num_loaders": self.num_loaders,
            },
        }

    def close(self) -> None:
        """Terminate all actors allocated for the pipeline."""
        if self._pipeline is not None:
            self._pipeline.close()

    def __enter__(self) -> "RayPipelineRunner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def launch(
        self,
        *,
        inputs: Optional[Mapping[str, Any]] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        **options: Any,
    ) -> None:
        """Run the protocol as a configuration of plain values describes it.

        The configuration is what any frontend gives: ``launch(**cfg)`` with
        Hydra's ``DictConfig``, a mapping read from YAML, or the command line
        (``launch_as_terminal_application``)::

            runner.launch(
                inputs={"particles": ["run_a/*.xmd"]},
                parameters={"Align": {"num_classes": 50}},
                run_dir="runs/a",
            )

        Args:
            inputs: The input files by input name: a path, or a list of
                paths and glob patterns (read by the loaders).
            parameters: Values by protocol name, ``{name: {parameter:
                value}}``; enums by name.
            options: The options of ``RUN_OPTIONS``: the runner's, which
                replace those given to the constructor, and ``explain``
                (write the report to this path instead of running),
                ``explain_gpus`` and ``report`` (write the report with the
                stats of the stages to this path after the run).

        Raises:
            TypeError: For an unknown option.
            ValueError: For unknown or missing inputs and parameters.
        """
        unknown = sorted(set(options) - set(RUN_OPTIONS))
        if unknown:
            raise TypeError(
                f"Unknown options {unknown}. Options: {list(RUN_OPTIONS)}.",
            )
        for name, value in options.items():
            RUN_OPTIONS[name].apply(self, value)
        self.set_parameters(parameters or {})
        explain = options.get("explain")
        if explain is not None:
            self.explain(Path(explain), _cluster_of(options.get("explain_gpus")))
        else:
            self.run(self._input_values(inputs or {}))
            report = options.get("report")
            if report is not None:
                _write_report(self.report(), Path(report))

    def default_config(self) -> Dict[str, Any]:
        """The configuration ``launch`` takes, with every value at its
        default as plain values: the inputs (``None``), the parameters by
        protocol name (the values given to the constructors, else the
        declared defaults) and the options of the runner. E.g. a node of
        Hydra's ``ConfigStore``, or a YAML file to start from."""
        return {
            "inputs": {name: None for name in self.protocol.configuration.inputs},
            "parameters": default_parameters(self.protocol),
            **{name: option.current(self) for name, option in RUN_OPTIONS.items()},
        }

    def argument_parser(self, required: bool = True) -> ArgumentParser:
        """The command line of the protocol: ``--<input>``,
        ``--<protocol name>.<parameter>`` and the options. Add arguments of
        your own to it; ``config_of`` leaves them out.

        Args:
            required: Whether inputs without a default are required.
        """
        parser = ArgumentParser(description=self.protocol.__doc__)
        add_input_arguments(parser, self.protocol, required)
        add_parameter_arguments(parser, self.protocol)
        group = parser.add_argument_group("options of the run")
        for name, option in RUN_OPTIONS.items():
            group.add_argument(
                f"--{argname(name)}",
                dest=f"options.{name}",
                default=None,
                help=option.help,
                **option.argument,
            )
        return parser

    def config_of(self, namespace: Namespace) -> Dict[str, Any]:
        """The configuration ``launch`` takes, from the arguments of
        ``argument_parser()`` parsed into ``namespace``: those given only."""
        return {
            "inputs": inputs_of(namespace, self.protocol),
            "parameters": parameters_of(namespace, self.protocol),
            **{
                name: value
                for name in RUN_OPTIONS
                if (value := getattr(namespace, f"options.{name}", None)) is not None
            },
        }

    def launch_as_terminal_application(
        self,
        argv: Optional[Sequence[str]] = None,
    ) -> None:
        """Run the protocol as the command line (``sys.argv``, or ``argv``)
        describes it: ``launch(**config_of(argument_parser().parse_args()))``.

        ``--explain`` writes the report of the pipeline instead of running
        it, so inputs and parameters are optional then.
        """
        argv = sys.argv[1:] if argv is None else list(argv)
        parser = self.argument_parser(required=not _explain_requested(argv))
        self.launch(**self.config_of(parser.parse_args(argv)))

    def _input_values(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """The inputs of a configuration as ``run`` takes them: one path, or
        the files of several paths and glob patterns.

        Raises:
            ValueError: For an unknown input, or a missing required one.
        """
        declared = self.protocol.configuration.inputs
        unknown = [name for name in inputs if name not in declared]
        missing = [
            name
            for name, field in declared.items()
            if not field.optional and field.default is None and name not in inputs
        ]
        if unknown or missing:
            raise ValueError(
                f"Unknown inputs {unknown} or missing inputs {missing}. "
                f"Inputs: {list(declared)}.",
            )
        return {
            name: _parse_inputs(
                (
                    [str(value)]
                    if isinstance(value, (str, os.PathLike))
                    else [str(item) for item in value]
                ),
            )
            for name, value in inputs.items()
            if value is not None
        }


@dataclass(frozen=True)
class RunOption:
    """An option of a run: its help, how it changes the runner, its current
    value as a plain value, and the kind of its command line argument."""

    help: str
    apply: Callable[[RayPipelineRunner, Any], None]
    current: Callable[[RayPipelineRunner], Any]
    argument: Mapping[str, Any]


def _set(attribute: str, convert: Callable[[Any], Any]) -> Callable[[Any, Any], None]:
    """Sets ``attribute`` of the runner to the converted value."""
    return lambda runner, value: setattr(runner, attribute, convert(value))


def _get(attribute: str) -> Callable[[Any], Any]:
    return lambda runner: getattr(runner, attribute)


def _no_runner_state(runner: RayPipelineRunner, value: Any) -> None:
    """For options of ``launch`` itself (``explain``, ``report``)."""


def _env(name: str, value: Any) -> Any:
    """The environment variable ``name`` if it is set, else ``value``."""
    return os.environ.get(name, value)


def _limit(value: str) -> Optional[int]:
    """A limit on the command line: a number, or ``none`` for no limit."""
    return None if value.lower() == "none" else int(value)


def _store_path(runner: RayPipelineRunner) -> Optional[str]:
    store = runner.run_store
    return str(store.path) if isinstance(store, FileRunStore) else None


# The options of a run, by name; values from the environment variables
# SCIPION_STREAM_* still override them, as they override the constructor.
RUN_OPTIONS: Dict[str, RunOption] = {
    "run_dir": RunOption(
        "Record the manifest of the run and the lineage of its items in this "
        "directory, readable while the run goes. Running again in it continues "
        "the run.",
        _set(
            "run_store",
            lambda value: None if value is None else FileRunStore(Path(value)),
        ),
        _store_path,
        {"type": str},
    ),
    "checkpoint_outputs": RunOption(
        "Keep the outputs of every protocol in the run directory, to inspect "
        "them and to rerun from (default: yes).",
        _set("checkpoint_outputs", bool),
        _get("checkpoint_outputs"),
        {"action": BooleanOptionalAction},
    ),
    "unique_elements": RunOption(
        "What the sink does with an element (e.g. a particle) it writes a "
        "second time: off (default), warn, or raise to fail the run.",
        _set("unique_elements", UniqueElements),
        lambda runner: runner.unique_elements.value,
        {"choices": [u.value for u in UniqueElements]},
    ),
    "num_loaders": RunOption(
        "Number of loader actors resolving the files of an input given as "
        "several paths or quoted glob patterns, e.g. 'run_a/extra/*.xmd' "
        "'run_b/extra/*.xmd'.",
        _set(
            "num_loaders",
            lambda value: int(_env("SCIPION_STREAM_NUM_LOADERS", value)),
        ),
        _get("num_loaders"),
        {"type": int},
    ),
    "loader_cpus": RunOption(
        "CPUs reserved for each loader actor.",
        _set(
            "loader_cpus",
            lambda value: float(_env("SCIPION_STREAM_LOADER_CPUS", value)),
        ),
        _get("loader_cpus"),
        {"type": float},
    ),
    "placement": RunOption(
        "managed (default) reserves the GPUs the planner gives every "
        "long-running protocol; ray lets Ray place them, for clusters shared "
        "with other jobs.",
        _set(
            "placement",
            lambda value: _placement_mode(
                _env("SCIPION_STREAM_PLACEMENT", value),
            ),
        ),
        _get("placement"),
        {"choices": ["managed", "ray"]},
    ),
    "queue_size": RunOption(
        "Number of items each pipeline stage reads ahead.",
        _set(
            "queue_size",
            lambda value: int(_env("SCIPION_STREAM_QUEUE_SIZE", value)),
        ),
        _get("queue_size"),
        {"type": int},
    ),
    "max_in_flight": RunOption(
        "Number of items on the routes out of a stage at once.",
        _set(
            "max_in_flight",
            lambda value: _env_int("SCIPION_STREAM_MAX_IN_FLIGHT", value),
        ),
        _get("max_in_flight"),
        {"type": int},
    ),
    "buffer_size": RunOption(
        "Most items waiting for a stage before the stages upstream wait; "
        "none for no limit.",
        _set(
            "buffer_size",
            lambda value: _env_limit("SCIPION_STREAM_BUFFER_SIZE", value),
        ),
        _get("buffer_size"),
        {"type": _limit},
    ),
    "spill_threshold": RunOption(
        "Number of waiting items a stage keeps in the object store before "
        "spilling; none to never spill.",
        _set(
            "spill_threshold",
            lambda value: _env_limit("SCIPION_STREAM_SPILL_THRESHOLD", value),
        ),
        _get("spill_threshold"),
        {"type": _limit},
    ),
    "profile": RunOption(
        "Profile the pipeline into a trace file at this path (a timestamped "
        "file without a path), which can be opened in https://ui.perfetto.dev "
        "while the pipeline runs.",
        _set("profile", _env_profile),
        lambda runner: None if runner.profile is None else str(runner.profile),
        {"nargs": "?", "const": True},
    ),
    "profile_log_level": RunOption(
        "Level of the profiling log of the workers, e.g. debug to log every item.",
        _set(
            "profile_log_level",
            lambda value: _env_log_level(
                "SCIPION_STREAM_PROFILE_LOG_LEVEL",
                logging.getLevelNamesMapping()[str(value).upper()],
            ),
        ),
        lambda runner: logging.getLevelName(runner.profile_log_level).lower(),
        {"type": str},
    ),
    "explain": RunOption(
        "Do not run the pipeline: print its stages and write its report to "
        f"this path (HTML, default {_DEFAULT_REPORT}) and a DOT file next to it.",
        _no_runner_state,
        lambda runner: None,
        {"nargs": "?", "const": _DEFAULT_REPORT},
    ),
    "explain_gpus": RunOption(
        "The GPUs of every node of the cluster --explain plans for, e.g. 4 or "
        "4,4 (default: the GPUs of this machine).",
        _no_runner_state,
        lambda runner: None,
        {"type": str},
    ),
    "report": RunOption(
        "Write the report of the pipeline with the stats of its stages to this "
        "path (HTML) after the run.",
        _no_runner_state,
        lambda runner: None,
        {"type": str},
    ),
}
