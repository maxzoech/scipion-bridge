from __future__ import annotations
import inspect
import textwrap
import ast
import abc

from dataclasses import dataclass
from itertools import chain
from typing import Dict, get_type_hints, get_origin, get_args, Any, Type, OrderedDict

from ..streaming.ops import Op
from .fields import Field, Input

from ..utils.ast import parse_ast
from ..utils.type_annotation import has_untyped_class_definitions


@dataclass
class _ProtocolTypeConfiguration:
    inputs: OrderedDict[str, Type[Input]]
    parameters: OrderedDict[str, Type[Field]]
    states: OrderedDict[str, Type]


@dataclass
class ProtocolConfiguration:
    inputs: OrderedDict[str, Input]
    parameters: OrderedDict[str, Field]


class ValidationError(Exception):
    pass


class Protocol(metaclass=abc.ABCMeta):

    _configuration: _ProtocolTypeConfiguration

    @property
    def configuration(self) -> ProtocolConfiguration:
        def _build_fields(source_dict):
            def _get_item(key, type_hint):
                field = getattr(type(self), key, None)
                if field is None:
                    field = type_hint(optional=False)
                    field._bound_name = key
                    field.name = key
                elif (
                    getattr(field, "dtype", None) is None
                    and getattr(type_hint, "_dtype", None) is not None
                ):
                    field._dtype = type_hint._dtype
                return (key, field)

            return OrderedDict(_get_item(k, v) for k, v in source_dict.items())

        inputs = _build_fields(self._configuration.inputs)
        params = _build_fields(self._configuration.parameters)

        return ProtocolConfiguration(inputs, params)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls._configuration = _create_protocol_info(cls)

    def __init__(self) -> None:
        pass

    def setup(self):
        """Optional setup method for protocol initialization."""
        pass

    def get_pipeline(self) -> Op:
        """Return the pipeline of operations for this protocol."""

        def _verify_outputs(outputs: Dict):
            if not isinstance(outputs, dict):
                raise ValueError("Pipeline output needs to be a dictionary")

            output_types = self.outputs()

            for key, value in outputs.items():
                if key not in output_types:
                    raise ValueError(f"Output '{key}' is not in declared outputs.")

                if not isinstance(value, output_types[key]):
                    raise TypeError(
                        f"Type mismatch for output key '{key}': "
                        f"expected {output_types[key]}, got '{type(value)}' ({value!r})"
                    )

            return outputs

        steps_op = self.steps()
        pipeline: Op = steps_op.map(_verify_outputs)

        return pipeline

    @abc.abstractmethod
    def outputs(self) -> Dict[str, Type]:
        pass

    @abc.abstractmethod
    def steps(self) -> Op:
        pass

    def validate_protocol_configuration(self):
        pass


def _create_protocol_info(cls: type[Protocol]) -> _ProtocolTypeConfiguration:

    attributes = get_type_hints(cls)

    source = inspect.getsource(cls)
    source = textwrap.dedent(source)

    tree = ast.parse(source)
    class_def = tree.body[0]
    assert isinstance(class_def, ast.ClassDef), "Protocol must be a class"

    if has_untyped_class_definitions(tree):
        raise TypeError(
            f"The protocol {cls.__qualname__} has declared attributes without type annotation."
        )

    # Check if the user has defined state without a type annotation
    untyped_assign_ops = [
        target.id
        for a in class_def.body
        if isinstance(a, ast.Assign)
        for target in a.targets
        if isinstance(target, ast.Name)
    ]
    if len(untyped_assign_ops) > 0:
        invalid_state_list = ", ".join([f"'{s}'" for s in untyped_assign_ops])
        raise TypeError(
            f"The protocol states {invalid_state_list} in {cls.__qualname__} do not have type annotations."
        )

    # Configure states
    typed_assign_ops = [
        a.target.id
        for a in class_def.body
        if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)
    ]

    inputs = OrderedDict()
    parameters = OrderedDict()
    states = OrderedDict()
    for name in typed_assign_ops:
        value = attributes[name]
        origin = get_origin(value) or getattr(value, "__origin__", None)

        if origin == Input:
            inputs[name] = value
        elif origin == Field:
            parameters[name] = value
        else:
            states[name] = value

    return _ProtocolTypeConfiguration(
        inputs=inputs,
        parameters=parameters,
        states=states,
    )
