from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sized
from enum import Enum
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional, Type, Union

from ...core.typed.resolve import ComposedResolver, find_resolver
from ...core.protocol.protocol_base import Protocol
from ...core.streaming.backend import StageStats
from ...core.streaming.pipeline import Pipeline
from ...core.streaming.sink import Sink
from ...core.streaming.sink_writer import SinkWriter
from .backend import RayBackend
from .container import configure_ray_env


def _convert_to_argname(name: str) -> str:
    """Convert a Protocol field name (snake_case or camelCase) to a CLI argument name."""
    cleaned = name.lstrip("-")
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", cleaned).replace("_", "-").lower()
    return f"--{kebab}"


def _format_stats(stats: Dict[str, StageStats]) -> str:
    """Format cumulative per-stage metrics as a table."""
    header = (
        f"{'stage':<40} {'in':>6} {'out':>6} {'idle_s':>8} "
        f"{'process_s':>10} {'blocked_s':>10} {'emit_s':>8}"
    )
    rows = [
        f"{label:<40} {s.items_in:>6} {s.items_out:>6} {s.idle_s:>8.2f} "
        f"{s.process_s:>10.2f} {s.blocked_s:>10.2f} {s.emit_s:>8.2f}"
        for label, s in stats.items()
    ]
    return "\n".join([header, *rows])


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
        sink: Optional[Union[Sink, SinkWriter, Callable[[Any], Any]]] = None,
        queue_size: int = 2,
    ) -> None:
        """
        Args:
            parameters: Values of the protocol parameters, served to
                ``Field.value`` on the driver and inside every pipeline stage.
                Parameters without a value resolve to their declared default.
            queue_size: Number of items each pipeline stage buffers. Overridden
                by the ``SCIPION_STREAM_QUEUE_SIZE`` environment variable.
        """
        self.protocol = protocol
        self.parameters = _validate_parameters(protocol, parameters or {})
        self.origin_types = origin_types or {}
        self.queue_size = int(os.environ.get("SCIPION_STREAM_QUEUE_SIZE", queue_size))
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
            )
            self._input_resolvers[name] = resolver

    def _compile_pipeline(self) -> Pipeline:
        """Compile protocol streaming DAG into an executable Ray pipeline."""
        _check_required_parameters(self.protocol, self.parameters)
        configure_ray_env(parameters=self.parameters)
        backend = RayBackend(
            queue_size=self.queue_size,
            parameters=self.parameters,
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

    @property
    def input_resolvers(self) -> Dict[str, ComposedResolver]:
        """Return the precomputed input resolvers."""
        return dict(self._input_resolvers)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Iterate over resolved inputs in lockstep interleaving and execute the compiled streaming pipeline."""
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

        active_iterators = {
            name: self._input_resolvers[name].iter(
                raw_value,
                chunk_size=env_chunk_size,
                target_bytes=env_target_bytes,
            )
            for name, raw_value in all_inputs.items()
            if name in self._input_resolvers
        }

        t_start = time.perf_counter()
        total_chunks = 0
        total_items = 0

        pipeline = self.pipeline
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

        elapsed = time.perf_counter() - t_start
        throughput = total_items / elapsed if elapsed > 0 else 0.0
        print(
            f"Pipeline complete in {elapsed:.2f}s "
            f"({total_chunks} chunks, {total_items} items, "
            f"{throughput:.1f} items/s)",
        )
        print(_format_stats(pipeline.stats()))

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
                type=Path,
                required=(not field.optional) or field.default is not None,
                default=field.default,
                help=field.help,
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

        inputs = {
            name: val
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
