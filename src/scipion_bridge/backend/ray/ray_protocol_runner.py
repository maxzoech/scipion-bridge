from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sized
from enum import Enum
import glob
import logging
import os
from pathlib import Path
import re
import time
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
from ...core.typed.resolve import ComposedResolver, find_resolver
from ...core.protocol.protocol_base import Protocol
from ...core.streaming.backend import StageStats
from ...core.streaming.pipeline import Pipeline
from ...core.streaming.sink import Sink
from ...core.streaming.sink_writer import SinkWriter
from ...core.streaming.spill import SpillStoreFactory, pickle_spill_store
from .backend import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_SPILL_THRESHOLD,
    RayBackend,
    RayCompiledPipeline,
)
from .container import configure_ray_env
from .loader import LoaderStats, RayInputLoader, expand_pattern, shard


def _convert_to_argname(name: str) -> str:
    """Convert a Protocol field name (snake_case or camelCase) to a CLI argument name."""
    cleaned = name.lstrip("-")
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", cleaned).replace("_", "-").lower()
    return f"--{kebab}"


def _format_stats(stats: Dict[str, StageStats]) -> str:
    """Format cumulative per-stage metrics as a table."""
    header = (
        f"{'stage':<40} {'in':>6} {'out':>6} {'idle_s':>8} "
        f"{'process_s':>10} {'blocked_s':>10} {'emit_s':>8} "
        f"{'buffered':>8} {'spilled':>8} {'fetch_s':>8} {'spill_s':>8}"
    )
    rows = [
        f"{label:<40} {s.items_in:>6} {s.items_out:>6} {s.idle_s:>8.2f} "
        f"{s.process_s:>10.2f} {s.blocked_s:>10.2f} {s.emit_s:>8.2f} "
        f"{s.buffered_peak:>8} {s.spilled:>8} {s.fetch_s:>8.2f} {s.spill_s:>8.2f}"
        for label, s in stats.items()
    ]
    return "\n".join([header, *rows])


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


def _validate_parameters(
    protocol: Protocol,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject values for parameters that ``protocol`` does not declare."""
    declared = protocol.configuration.parameters

    unknown = [name for name in parameters if name not in declared]
    if unknown:
        raise ValueError(
            f"Unknown parameters {unknown} for protocol "
            f"{type(protocol).__name__}. Declared: {list(declared)}.",
        )

    return dict(parameters)


def _check_required_parameters(
    protocol: Protocol,
    parameters: Mapping[str, Any],
) -> None:
    """Fail if a required parameter without default has no value."""
    declared = protocol.configuration.parameters
    missing = [
        name
        for name, field in declared.items()
        if not field.optional and field.default is None and name not in parameters
    ]

    if missing:
        raise ValueError(
            f"Missing values for required parameters {missing} of protocol "
            f"{type(protocol).__name__}.",
        )


def _parse_cli_value(dtype: Any, value: Any) -> Any:
    """Convert a parsed command line value to the parameter type."""
    match dtype:
        case type() if issubclass(dtype, Enum) and isinstance(value, str):
            return dtype[value]

        case _:
            return value


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
    ) -> None:
        """
        Every buffering option is overridden by an environment variable, see
        ``RayBackend`` for their meaning.

        Args:
            parameters: Values of the protocol parameters, served to
                ``Field.value`` on the driver and inside every pipeline stage.
                Parameters without a value resolve to their declared default.
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
        """
        self.protocol = protocol
        self.parameters = _validate_parameters(protocol, parameters or {})
        self.origin_types = origin_types or {}
        self.intermediate_types = intermediate_types or {}
        self.num_loaders = int(
            os.environ.get("SCIPION_STREAM_NUM_LOADERS", num_loaders)
        )
        self.loader_cpus = float(
            os.environ.get("SCIPION_STREAM_LOADER_CPUS", loader_cpus)
        )
        self.queue_size = int(os.environ.get("SCIPION_STREAM_QUEUE_SIZE", queue_size))
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
        self._input_resolvers: Dict[str, ComposedResolver] = {}
        self._pipeline: Optional[Pipeline] = None

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
        _check_required_parameters(self.protocol, self.parameters)
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
        )
        steps = self.protocol.get_pipeline()

        sink_node = (
            self.sink
            if isinstance(self.sink, Sink)
            else Sink(
                self.sink or _default_sink_handler,
            )
        )
        steps.op(
            sink_node,
        )

        return Pipeline.from_sink(
            sink_node,
            backend=backend,
        )

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

        self.parameters = _validate_parameters(self.protocol, parameters)

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

        t_start = time.perf_counter()
        total_chunks = 0
        total_items = 0

        pipeline = self.pipeline
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
        if loader_stats:
            print(_format_loader_stats(loader_stats))
        match self.profile:
            case None:
                pass
            case path:
                print(f"Profiling trace: {path.resolve()}")

    def close(self) -> None:
        """Terminate all actors allocated for the pipeline."""
        if self._pipeline is not None:
            self._pipeline.close()

    def __enter__(self) -> "RayPipelineRunner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def launch_as_terminal_application(self):
        """CLI entry point for running the protocol from terminal arguments."""
        parser = ArgumentParser(description=self.protocol.__doc__)
        config = self.protocol.configuration

        for name, field in config.inputs.items():
            parser.add_argument(
                _convert_to_argname(name),
                dest=name,
                nargs="+",
                required=(not field.optional) or field.default is not None,
                default=field.default,
                help=field.help,
            )

        parser.add_argument(
            "--num-loaders",
            type=int,
            help=(
                "Number of loader actors resolving the files of an input "
                "given as several paths or quoted glob patterns, e.g. "
                "'run_a/extra/*.xmd' 'run_b/extra/*.xmd'."
            ),
        )

        parser.add_argument(
            "--profile",
            nargs="?",
            const=True,
            type=Path,
            help=(
                "Profile the pipeline into a trace file at this path (a "
                "timestamped file without a path), which can be opened in "
                "https://ui.perfetto.dev while the pipeline runs."
            ),
        )

        for name, field in config.parameters.items():
            dtype = field.dtype
            kwargs: dict[str, Any] = {
                "required": not field.optional,
                "help": field.help,
                "dest": name,
                "default": field.default,
            }

            match dtype:
                case _ if isinstance(dtype, type) and issubclass(dtype, Enum):
                    kwargs["choices"] = [e.name for e in dtype]
                    if isinstance(field.default, Enum):
                        kwargs["default"] = field.default.name

                case _ if dtype is bool:
                    kwargs["action"] = (
                        "store_true" if not field.default else "store_false"
                    )
                    kwargs["action"] = BooleanOptionalAction
                    kwargs.pop("required", None)

                case _:
                    kwargs["type"] = dtype

            parser.add_argument(
                _convert_to_argname(name),
                **kwargs,
            )

        args = parser.parse_args()
        if args.profile is not None:
            self.set_profile(args.profile)
        if args.num_loaders is not None:
            self.num_loaders = args.num_loaders

        inputs = {
            name: _parse_inputs(val) if isinstance(val, list) else val
            for name in config.inputs
            if (val := getattr(args, name, None)) is not None
        }

        self.set_parameters(
            {
                name: _parse_cli_value(field.dtype, val)
                for name, field in config.parameters.items()
                if (val := getattr(args, name, None)) is not None
            },
        )
        self.run(
            inputs,
        )
