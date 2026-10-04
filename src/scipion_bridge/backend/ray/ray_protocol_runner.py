from argparse import ArgumentParser
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sized
from enum import Enum
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, Optional, Type, Union

from ...core.typed.resolve import ComposedResolver, find_resolver
from ...core.protocol.protocol_base import Protocol
from ...core.streaming.backend import StreamingBackendProvider
from ...core.streaming.pipeline import Pipeline
from ...core.streaming.sink import Sink
from ...core.streaming.sink_writer import SinkWriter
from .backend import RayBackend


def _convert_to_argname(name: str) -> str:
    """Convert a Protocol field name (snake_case or camelCase) to a CLI argument name."""
    cleaned = name.lstrip("-")
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", cleaned).replace("_", "-").lower()
    return f"--{kebab}"


def _default_sink_handler(outputs: Dict[str, Any]) -> None:
    """Default sink handler for compiled Ray pipelines."""
    pass


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
        origin_types: Optional[Dict[str, Type]] = None,
        sink: Optional[Union[Sink, SinkWriter, Callable[[Any], Any]]] = None,
    ) -> None:
        self.protocol = protocol
        self.origin_types = origin_types or {}
        self.backend = RayBackend()
        self.sink = sink
        self._input_resolvers: Dict[str, ComposedResolver] = {}

        self._precompute_resolvers()
        self._pipeline: Pipeline = self._compile_pipeline()

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
            backend=self.backend,
        )

    @property
    def pipeline(self) -> Pipeline:
        """Return the compiled streaming Pipeline."""
        return self._pipeline

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

        with self._pipeline:
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

                    self._pipeline.send(
                        **{name: chunk},
                    )

        elapsed = time.perf_counter() - t_start
        throughput = total_items / elapsed if elapsed > 0 else 0.0
        print(
            f"Pipeline complete in {elapsed:.2f}s "
            f"({total_chunks} chunks, {total_items} items, "
            f"{throughput:.1f} items/s)",
        )

    def close(self) -> None:
        """Terminate all actors allocated for the pipeline."""
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
                required=not field.optional,
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

        args, unparsed_args = parser.parse_known_args()

        inputs = {
            name: val
            for name in config.inputs
            if (val := getattr(args, name, None)) is not None
        }

        self.run(
            inputs,
        )
