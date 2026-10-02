from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
import re
from typing import Any

import scipion_bridge as B
from ...core.protocol.protocol_base import Protocol


@dataclass
class ResolvedInput:

    value: Any
    dtype: type
    is_path: bool


def _convert_to_argname(name: str) -> str:
    """Convert a Protocol field name (snake_case or camelCase) to a CLI argument name."""
    cleaned = name.lstrip("-")
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", cleaned).replace("_", "-").lower()
    return f"--{kebab}"


class RayPipelineRunner:

    def __init__(self, protocol: Protocol) -> None:
        self.protocol = protocol

    def __call__(self, **kwds: Any):
        self.run(kwargs=kwds)

    def run(self, **kwargs):
        print("Run protocol here")

    def launch_as_terminal_application(self):
        parser = ArgumentParser()
        config = self.protocol.configuration

        def _parse_input(val: Any, dtype: type):
            inputs = Path(val)
            return ResolvedInput(inputs, dtype=dtype, is_path=True)

        for name, field in config.inputs.items():
            dtype = field.dtype
            assert isinstance(dtype, type)

            parser.add_argument(
                _convert_to_argname(name),
                dest=name,
                type=partial(_parse_input, dtype=dtype),
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

        particles: ResolvedInput = args.particles
        inputs = B.resolve(particles.value, astype=particles.dtype)

        print(inputs)
