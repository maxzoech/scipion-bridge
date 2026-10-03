from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, Optional, Type

from tqdm import tqdm

import scipion_bridge as B
from ...core.protocol.protocol_base import Protocol


def _convert_to_argname(name: str) -> str:
    """Convert a Protocol field name (snake_case or camelCase) to a CLI argument name."""
    cleaned = name.lstrip("-")
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", cleaned).replace("_", "-").lower()
    return f"--{kebab}"


class RayPipelineRunner:
    """Ray pipeline runner for Scipion Bridge protocols.

    Precomputes resolvers for protocol inputs upon initialization,
    and iterates over resolved inputs on execution.
    """

    def __init__(
        self,
        protocol: Protocol,
        *,
        origin_types: Optional[Dict[str, Type]] = None,
        chunk_size: Optional[int] = None,
        target_bytes: Optional[int] = None,
    ) -> None:
        self.protocol = protocol
        self.origin_types = origin_types or {}
        self.chunk_size = chunk_size
        self.target_bytes = target_bytes
        self._input_resolvers: Dict[str, B.ComposedResolver] = {}

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

            resolver = B.find_resolver(origin_type, target_type)
            self._input_resolvers[name] = resolver

    @property
    def input_resolvers(self) -> Dict[str, B.ComposedResolver]:
        """Return the precomputed input resolvers."""
        return dict(self._input_resolvers)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        chunk_size: Optional[int] = None,
        target_bytes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Iterate over the resolved inputs and print the length of each chunk."""
        all_inputs = dict(inputs or {})
        all_inputs.update(kwargs)

        effective_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        effective_target_bytes = (
            target_bytes if target_bytes is not None else self.target_bytes
        )

        for name, raw_value in all_inputs.items():
            assert name in self._input_resolvers

            resolver = self._input_resolvers[name]
            for chunk in tqdm(
                resolver.iter(
                    raw_value,
                    chunk_size=effective_chunk_size,
                    target_bytes=effective_target_bytes,
                )
            ):
                print(len(chunk))

    def launch_as_terminal_application(self):
        """CLI entry point for running the protocol from terminal arguments."""
        parser = ArgumentParser()
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
