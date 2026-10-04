import abc
import ast
import inspect
import textwrap
import uuid

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    OrderedDict,
    Type,
    TypeVar,
    cast,
    get_origin,
    get_type_hints,
)

from ..streaming.ops import Op
from .fields import Field, Input, Resource


@dataclass
class _ProtocolTypeConfiguration:
    inputs: OrderedDict[str, Type[Input]]
    parameters: OrderedDict[str, Type[Field]]
    resources: OrderedDict[str, Type[Resource]]
    states: OrderedDict[str, Type]


@dataclass
class ProtocolConfiguration:
    inputs: OrderedDict[str, Input]
    parameters: OrderedDict[str, Field]
    resources: OrderedDict[str, Resource]


class ValidationError(Exception):
    pass


FieldT = TypeVar("FieldT", bound=Field)


def _bind_field(
    owner_cls: type,
    key: str,
    type_hint: Type[FieldT] | Any,
) -> tuple[str, FieldT]:
    field = getattr(owner_cls, key, None)
    if field is None:
        bound_field = type_hint(optional=False)
        bound_field._bound_name = key
        bound_field.name = key
        return (
            key,
            bound_field,
        )

    else:
        if (
            getattr(field, "dtype", None) is None
            and getattr(type_hint, "_dtype", None) is not None
        ):
            field._dtype = type_hint._dtype
        return (
            key,
            field,
        )


class Protocol(metaclass=abc.ABCMeta):

    _configuration: _ProtocolTypeConfiguration

    @property
    def configuration(self) -> ProtocolConfiguration:
        inputs = OrderedDict(
            _bind_field(type(self), k, v) for k, v in self._configuration.inputs.items()
        )
        params = OrderedDict(
            _bind_field(type(self), k, v)
            for k, v in self._configuration.parameters.items()
        )
        resources = OrderedDict(
            (k, getattr(type(self), k)) for k in self._configuration.resources
        )
        return ProtocolConfiguration(
            inputs=inputs,
            parameters=params,
            resources=resources,
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls._configuration = _create_protocol_info(cls)

    def __init__(self, protocol_id: Optional[str] = None) -> None:
        self.protocol_id: str = protocol_id or uuid.uuid4().hex

    def setup(self):
        """Optional setup method for protocol initialization."""
        pass

    def get_pipeline(self) -> Op:
        """Return the pipeline of operations for this protocol."""
        output_types = self.outputs()

        def _verify_outputs(outputs: Any) -> Dict[str, Any]:
            if not isinstance(outputs, dict):
                raise ValidationError(
                    f"Protocol steps output must be a dictionary, got {type(outputs).__name__}",
                )

            for key, value in outputs.items():
                if key not in output_types:
                    raise ValidationError(f"Output '{key}' is not in declared outputs.")

                if not isinstance(value, output_types[key]):
                    raise ValidationError(
                        f"Type mismatch for output key '{key}': "
                        f"expected {output_types[key]}, got '{type(value)}' ({value!r})",
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


def _extract_declaration_order_from_ast(cls: type) -> Optional[List[str]]:
    """Extract declaration order of annotated attributes from class source AST.

    Returns None if source code is not available on disk (e.g. cloudpickle reconstruction,
    dynamically generated classes, or interactive environments).
    """
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return None

    try:
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except (SyntaxError, IndentationError):
        return None

    if not tree.body or not isinstance(tree.body[0], ast.ClassDef):
        return None

    class_def = tree.body[0]
    return [
        a.target.id
        for a in class_def.body
        if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)
    ]


def _create_protocol_info(cls: type[Protocol]) -> _ProtocolTypeConfiguration:
    annotations: Dict[str, Any] = getattr(cls, "__annotations__", {})

    # Detect untyped class attributes (e.g. `state = 42` instead of `state: int = 42`)
    untyped = [
        k
        for k, v in cls.__dict__.items()
        if not (k.startswith("__") and k.endswith("__"))
        and not k.startswith("_abc_")
        and not callable(v)
        and not isinstance(v, (type, property, classmethod, staticmethod))
        and k not in annotations
    ]
    if untyped:
        invalid_list = ", ".join(f"'{k}'" for k in untyped)
        raise TypeError(
            f"The protocol {cls.__qualname__} has declared attributes without type annotation: {invalid_list}.",
        )

    match _extract_declaration_order_from_ast(cls):
        case list() as ast_order:
            ordered_names = [name for name in ast_order if name in annotations] + [
                name for name in annotations if name not in ast_order
            ]
        case _:
            ordered_names = list(annotations.keys())

    hints = get_type_hints(cls)

    inputs: OrderedDict[str, Type[Input]] = OrderedDict()
    parameters: OrderedDict[str, Type[Field]] = OrderedDict()
    resources: OrderedDict[str, Type[Resource]] = OrderedDict()
    states: OrderedDict[str, Type] = OrderedDict()

    def _is_type_or_origin_subclass(val: Any, target_type: type) -> bool:
        origin = get_origin(val)
        return (isinstance(val, type) and issubclass(val, target_type)) or (
            isinstance(origin, type) and issubclass(origin, target_type)
        )

    for name in ordered_names:
        value = hints.get(name, annotations[name])
        origin = get_origin(value)

        is_input = (isinstance(value, type) and issubclass(value, Input)) or (
            isinstance(origin, type) and issubclass(origin, Input)
        )
        is_field = (isinstance(value, type) and issubclass(value, Field)) or (
            isinstance(origin, type) and issubclass(origin, Field)
        )
        if _is_type_or_origin_subclass(value, Input):
            inputs[name] = cast(Type[Input], value)

        elif _is_type_or_origin_subclass(value, Field):
            parameters[name] = cast(Type[Field], value)

        elif _is_type_or_origin_subclass(value, Resource):
            field_obj = getattr(cls, name, None)

            if not isinstance(field_obj, Resource) or field_obj.builder is None:
                raise TypeError(
                    f"The protocol resource '{name}' in {cls.__qualname__} must be assigned a Resource instance with a builder callable.",
                )

            resources[name] = cast(Type[Resource], value)
        else:
            states[name] = value

        match (is_input, is_field):
            case (True, _):
                inputs[name] = cast(Type[Input], value)
            case (_, True):
                parameters[name] = cast(Type[Field], value)
            case _:
                states[name] = value

    return _ProtocolTypeConfiguration(
        inputs=inputs,
        parameters=parameters,
        resources=resources,
        states=states,
    )
