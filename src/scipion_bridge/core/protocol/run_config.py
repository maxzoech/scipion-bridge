"""The inputs and parameters of a protocol as a mapping of plain values.

A run is configured by a mapping that any frontend can produce (Hydra, a
YAML file, a custom parser, the command line of a runner)::

    inputs:     {particles: ["run_a/*.xmd"]}
    parameters: {Align: {num_classes: 50, output_ctf: PHASE_FLIPPED}}

Parameters are scoped by the instance names of the protocols
(``Protocol.name``); enums are given by name. Mappings and sequences are
read through ``collections.abc``, so e.g. OmegaConf's ``DictConfig`` and
``ListConfig`` work as they are, without a dependency on them.

The command line form is generated from the same declarations: an input
``--particles``, a parameter ``--Align.num-classes``.
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import PurePath
from typing import Any, Dict, List

from .fields import BoundField
from .protocol_base import Protocol


def argname(name: str) -> str:
    """A field name (snake_case or camelCase) as the kebab-case of a flag."""
    cleaned = name.lstrip("-")
    return re.sub(r"(?<!^)(?=[A-Z])", "-", cleaned).replace("_", "-").lower()


def plain(value: Any) -> Any:
    """``value`` as a plain configuration value: enums by name, paths as
    strings, mappings as dicts and sequences as lists."""
    match value:
        case Enum():
            return value.name
        case PurePath():
            return str(value)
        case str() | bytes():
            return value
        case Mapping():
            return {str(key): plain(item) for key, item in value.items()}
        case Sequence():
            return [plain(item) for item in value]
        case _:
            return value


def default_parameters(protocol: Protocol) -> Dict[str, Dict[str, Any]]:
    """The defaults of the parameters of ``protocol`` by instance name: the
    values given to the constructors, else the declared defaults."""
    return {
        name: {key: plain(field.default) for key, field in fields.items()}
        for name, fields in protocol.parameters_by_name().items()
    }


def parameter_values(
    protocol: Protocol,
    parameters: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """The values of ``parameters`` (``{name: {parameter: value}}``) for the
    protocols of ``protocol``, with enums given by name converted.

    Raises:
        ValueError: For a protocol name or a parameter ``protocol`` does
            not have.
    """
    declared = protocol.parameters_by_name()
    unknown = [name for name in parameters if name not in declared]
    if unknown:
        raise ValueError(
            f"No protocols named {unknown}; the protocols are {list(declared)}.",
        )
    values: Dict[str, Dict[str, Any]] = {}
    for name, scoped in parameters.items():
        if not isinstance(scoped, Mapping):
            raise TypeError(
                f"The parameters of {name} must be a mapping of parameter names "
                f"to values, got {type(scoped).__name__}.",
            )
        fields = declared[name]
        unknown = [key for key in scoped if key not in fields]
        if unknown:
            raise ValueError(
                f"Unknown parameters {unknown} of protocol {name}. "
                f"Declared: {list(fields)}.",
            )
        values[name] = {
            key: _converted(fields[key], value) for key, value in scoped.items()
        }
    return values


def check_required(
    protocol: Protocol,
    values: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail if a required parameter has no value: none in ``values``, given
    to the constructor or declared.

    Raises:
        ValueError: Naming the parameters as ``name.parameter``.
    """
    missing = [
        f"{name}.{key}"
        for name, fields in protocol.parameters_by_name().items()
        for key, field in fields.items()
        if not field.optional
        and field.default is None
        and key not in values.get(name, {})
    ]
    if missing:
        raise ValueError(f"Missing values for required parameters {missing}.")


def _converted(field: BoundField, value: Any) -> Any:
    """A configuration value as the value of ``field``: an enum given by name
    becomes the enum, sequences become tuples (protocols hold immutable
    values)."""
    dtype = field.dtype
    match value:
        case str() if isinstance(dtype, type) and issubclass(dtype, Enum):
            return dtype[value]
        case str() | bytes():
            return value
        case Sequence():
            return tuple(value)
        case _:
            return value


# -- The command line -----------------------------------------------------------
def add_input_arguments(
    parser: ArgumentParser,
    protocol: Protocol,
    required: bool,
) -> None:
    """``--<input>`` for every input of ``protocol``: one or more paths or
    quoted glob patterns. ``required`` makes the inputs without a default
    required."""
    group = parser.add_argument_group("inputs")
    for name, field in protocol.configuration.inputs.items():
        group.add_argument(
            f"--{argname(name)}",
            dest=f"inputs.{name}",
            nargs="+",
            required=required and not field.optional and field.default is None,
            help=field.help,
        )


def add_parameter_arguments(parser: ArgumentParser, protocol: Protocol) -> None:
    """``--<name>.<parameter>`` for every parameter of the protocols of
    ``protocol``, in one group per protocol. Parameters left out keep their
    defaults, so none is required here (``check_required`` checks them)."""
    for name, fields in protocol.parameters_by_name().items():
        group = parser.add_argument_group(f"parameters of {name}")
        for key, field in fields.items():
            group.add_argument(
                f"--{name}.{argname(key)}",
                dest=f"parameters.{name}.{key}",
                default=None,
                help=_help(field),
                **_argument_kind(field),
            )


def inputs_of(namespace: Namespace, protocol: Protocol) -> Dict[str, List[str]]:
    """The inputs given on the command line."""
    return {
        name: values
        for name in protocol.configuration.inputs
        if (values := getattr(namespace, f"inputs.{name}", None)) is not None
    }


def parameters_of(
    namespace: Namespace, protocol: Protocol
) -> Dict[str, Dict[str, Any]]:
    """The parameters given on the command line, by protocol name."""
    given: Dict[str, Dict[str, Any]] = {}
    for name, fields in protocol.parameters_by_name().items():
        for key in fields:
            value = getattr(namespace, f"parameters.{name}.{key}", None)
            if value is not None:
                given.setdefault(name, {})[key] = value
    return given


def _help(field: BoundField) -> str:
    default = (
        "required"
        if field.default is None and not field.optional
        else plain(field.default)
    )
    return f"{field.help or ''} (default: {default})".strip()


def _argument_kind(field: BoundField) -> Dict[str, Any]:
    dtype = field.dtype
    if isinstance(dtype, type) and issubclass(dtype, Enum):
        return {"choices": [member.name for member in dtype]}
    elif dtype is bool:
        return {"action": BooleanOptionalAction}
    else:
        return {"type": dtype}
