"""Helpers to merge two protocols into a chain (``first | second``)."""

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    TypeVar,
    get_origin,
)

from ..streaming.node import Node, replace_node
from ..streaming.ops import Op, Source
from .fields import Field, Input, Resource


class ChainError(TypeError):
    """Raised when two protocols cannot be chained."""


class Select:
    """Picklable stage selecting one key from a protocol output dictionary."""

    def __init__(self, key: str) -> None:
        self.key = key

    def __call__(self, outputs: Dict[str, Any]) -> Any:
        return outputs[self.key]

    def __repr__(self) -> str:
        return f"Select({self.key!r})"


def types_compatible(produced: Any, expected: Any) -> bool:
    """Return whether an output of type ``produced`` can feed an input of type ``expected``."""
    if produced == expected:
        return True

    produced_origin = get_origin(produced) or produced
    expected_origin = get_origin(expected) or expected
    return (
        isinstance(produced_origin, type)
        and isinstance(expected_origin, type)
        and issubclass(produced_origin, expected_origin)
        and getattr(produced, "_dtype", None) == getattr(expected, "_dtype", None)
    )


def resolve_wires(
    output_types: Mapping[str, Any],
    inputs: Mapping[str, Input],
    mapping: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Decide which outputs of the first protocol feed which inputs of the second.

    Returns:
        Mapping from input name of the second protocol to the output key of the
        first protocol that feeds it.

    Resolution order: an explicit ``mapping``; else outputs and inputs with the
    same name; else, if the first protocol has a single output, the single
    type-compatible input of the second protocol.
    """
    match mapping:
        case None:
            wires = {name: name for name in inputs if name in output_types}
            wires = wires or _wire_single_output(output_types, inputs)

        case _:
            targets = list(mapping.values())
            if len(set(targets)) != len(targets):
                raise ChainError(
                    f"Several outputs are mapped to the same input: {dict(mapping)}.",
                )
            wires = {target: key for key, target in mapping.items()}

    for name, key in wires.items():
        if key not in output_types:
            raise ChainError(f"The first protocol has no output '{key}'.")

        if name not in inputs:
            raise ChainError(f"The second protocol has no input '{name}'.")

        if not types_compatible(output_types[key], inputs[name].dtype):
            raise ChainError(
                f"Output '{key}' of type {output_types[key]} cannot feed input "
                f"'{name}' of type {inputs[name].dtype}.",
            )

    return wires


def _wire_single_output(
    output_types: Mapping[str, Any],
    inputs: Mapping[str, Input],
) -> Dict[str, str]:
    if len(output_types) != 1:
        raise ChainError(
            "No output of the first protocol matches an input name of the second "
            "protocol. Pass an explicit mapping to pipe().",
        )

    ((key, produced),) = output_types.items()
    candidates = [
        name
        for name, field in inputs.items()
        if types_compatible(produced, field.dtype)
    ]
    match candidates:
        case [name]:
            return {name: key}

        case []:
            raise ChainError(
                f"Output '{key}' of type {produced} matches no input of the second protocol.",
            )

        case _:
            raise ChainError(
                f"Output '{key}' is compatible with several inputs {candidates}. "
                "Pass an explicit mapping to pipe().",
            )


_FieldT = TypeVar("_FieldT")


def _merge(
    kind: str,
    first: Mapping[str, _FieldT],
    second: Mapping[str, _FieldT],
    same: Any,
) -> "OrderedDict[str, _FieldT]":
    """Union of two named collections; equal duplicates are shared, others rejected."""
    clashes = [
        name for name in second if name in first and not same(first[name], second[name])
    ]
    if clashes:
        raise ChainError(
            f"Both protocols declare the {kind} {clashes} with different definitions. "
            "Rename one of them.",
        )

    return OrderedDict(
        [*first.items(), *((k, v) for k, v in second.items() if k not in first)]
    )


def _same_field(a: Field, b: Field) -> bool:
    return a == b and a.dtype == b.dtype


def _same_resource(a: Resource, b: Resource) -> bool:
    return a.builder == b.builder and a.scope == b.scope and a.dtype == b.dtype


def merge_inputs(
    first: Mapping[str, Input],
    second: Mapping[str, Input],
    wired: Mapping[str, str],
) -> "OrderedDict[str, Input]":
    """Inputs of the chain: those of ``first`` plus the unwired ones of ``second``."""
    remaining = {k: v for k, v in second.items() if k not in wired}
    return _merge("inputs", first, remaining, lambda a, b: a.dtype == b.dtype)


def merge_parameters(
    first: Mapping[str, Field],
    second: Mapping[str, Field],
) -> "OrderedDict[str, Field]":
    return _merge("parameters", first, second, _same_field)


def merge_resources(
    first: Mapping[str, Resource],
    second: Mapping[str, Resource],
) -> "OrderedDict[str, Resource]":
    return _merge("resources", first, second, _same_resource)


def _collect_sources(sink: Node) -> List[Source]:
    """All sources reachable upstream from ``sink``, in discovery order."""
    seen: set[Node] = set()
    stack: List[Node] = [sink]
    sources: List[Source] = []
    while stack:
        node = stack.pop()
        if node in seen:
            continue

        seen.add(node)
        match node:
            case Source():
                sources.append(node)

            case _:
                stack.extend(reversed(node.upstream))

    return sources


def merge_pipelines(first_out: Op, second_out: Op, wires: Mapping[str, str]) -> Op:
    """Connect the pipeline of the first protocol into the one of the second.

    Every source of ``second_out`` named in ``wires`` is replaced by a stage
    selecting the corresponding key from ``first_out``. Afterwards, sources with
    the same name are unified so that shared inputs are fed only once.
    """
    adapters: Dict[str, Op] = {
        name: first_out.map(Select(key)) for name, key in wires.items()
    }
    for source in _collect_sources(second_out):
        if source.name in adapters:
            replace_node(source, adapters[source.name])

    if isinstance(second_out, Source) and second_out.name in adapters:
        return adapters[second_out.name]

    canonical: Dict[str, Source] = {}
    for source in _collect_sources(second_out):
        original = canonical.setdefault(source.name, source)
        if original is not source:
            replace_node(source, original)

    return second_out
