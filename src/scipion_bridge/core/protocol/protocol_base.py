import abc
import ast
import inspect
import textwrap
import uuid

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Type,
    TypeVar,
    cast,
    get_origin,
    get_type_hints,
)

from ..environment.compute import ComputeAssignment, ComputeResources, TaskType
from ..streaming.ops import MapElementOp, MapOp, Op, lineage
from .chain_utils import (
    merge_inputs,
    merge_parameters,
    merge_pipelines,
    merge_resources,
    resolve_wires,
)
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
    # Resources of the stages created by ``steps``, set with ``@resources``.
    compute_resources: ClassVar[Optional[ComputeResources]] = None

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
        return self._tagged_steps().map(self._verify_outputs)

    def _tagged_steps(self) -> Op:
        """Return ``steps()``, with its stages assigned this protocol's resources.

        Only the user functions (``map``, ``map_element``) are assigned; the
        stages of built-in ops need no resources. The stages of the protocol
        form one group, which shares its resources if it is long-running.
        """
        steps = self.steps()
        match self.compute_resources:
            case None:
                return steps
            case resources:
                assignment = ComputeAssignment(resources, group=self.protocol_id)
                # cpu_only stages run without GPUs and outside the
                # long-running executor of the protocol.
                cpu_assignment = ComputeAssignment(
                    ComputeResources(cpus=resources.cpus),
                    group=self.protocol_id,
                )
                for node in lineage(steps):
                    match node:
                        case MapOp() | MapElementOp() if node.compute is None:
                            node.compute = (
                                cpu_assignment if node.cpu_only else assignment
                            )
                return steps

    def _verify_outputs(self, outputs: Any) -> Dict[str, Any]:
        """Validate step outputs against the declared outputs.

        Implemented as a bound method so that ``outputs()`` is evaluated in the
        process executing the step, instead of being captured on the driver and
        pickled together with a closure.
        """
        if not isinstance(outputs, dict):
            raise ValidationError(
                f"Protocol steps output must be a dictionary, got {type(outputs).__name__}",
            )

        output_types = self.outputs()

        for key, value in outputs.items():
            if key not in output_types:
                raise ValidationError(f"Output '{key}' is not in declared outputs.")

            expected_type = output_types[key]
            origin = get_origin(expected_type) or expected_type

            if not isinstance(value, origin):
                raise ValidationError(
                    f"Type mismatch for output key '{key}': "
                    f"expected {expected_type}, got '{type(value)}' ({value!r})",
                )

            expected_dtype = getattr(expected_type, "_dtype", None)
            actual_dtype = getattr(value, "dtype", getattr(value, "_dtype", None))
            if expected_dtype is not None and actual_dtype != expected_dtype:
                raise ValidationError(
                    f"Type argument mismatch for output key '{key}': "
                    f"expected dtype '{expected_dtype}', got '{actual_dtype}'",
                )

        return outputs

    @abc.abstractmethod
    def outputs(self) -> Dict[str, Type]:
        pass

    @abc.abstractmethod
    def steps(self) -> Op:
        """Return the streaming pipeline of the protocol, built from its inputs.

        Each ``.map()`` runs as a separate pipeline stage. Split GPU work and
        CPU post-processing into separate maps so that they overlap on
        consecutive batches::

            def steps(self):
                return (
                    self.particles.chunk(256)
                    .map(self._forward)
                    .map(self._build_metadata)
                )
        """
        pass

    def validate_protocol_configuration(self):
        pass

    def pipe(
        self,
        other: "Protocol",
        mapping: Optional[Mapping[str, str]] = None,
    ) -> "ChainedProtocol":
        """Chain ``other`` after this protocol.

        The outputs of this protocol feed the inputs of ``other``. Unless
        ``mapping`` (output name -> input name of ``other``) is given, outputs
        are matched to inputs by name, or by type if this protocol has a single
        output. The result is again a protocol, whose inputs are the inputs of
        this protocol plus the unwired inputs of ``other``.
        """
        return ChainedProtocol(self, other, mapping=mapping)

    def __or__(self, other: Any) -> "ChainedProtocol":
        match other:
            case Protocol():
                return self.pipe(other)

            case _:
                return NotImplemented


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


class ChainedProtocol(Protocol):
    """Two protocols executed as one: the outputs of ``first`` feed ``second``.

    Created with ``first | second`` or ``first.pipe(second)``. The steps of both
    protocols are merged into a single streaming pipeline; each batch emitted by
    ``first`` is passed as one item to the inputs of ``second``. Inputs, parameters
    and resources of both protocols are merged; the outputs are those of ``second``.
    """

    def __init__(
        self,
        first: Protocol,
        second: Protocol,
        *,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__()
        self.first = first
        self.second = second

        first_config = first.configuration
        second_config = second.configuration
        self._wires = resolve_wires(first.outputs(), second_config.inputs, mapping)
        self._chain_configuration = ProtocolConfiguration(
            inputs=merge_inputs(first_config.inputs, second_config.inputs, self._wires),
            parameters=merge_parameters(
                first_config.parameters,
                second_config.parameters,
            ),
            resources=merge_resources(first_config.resources, second_config.resources),
        )

    @property
    def configuration(self) -> ProtocolConfiguration:
        return self._chain_configuration

    def setup(self) -> None:
        self.first.setup()
        self.second.setup()

    def validate_protocol_configuration(self) -> None:
        self.first.validate_protocol_configuration()
        self.second.validate_protocol_configuration()

    def outputs(self) -> Dict[str, Type]:
        return self.second.outputs()

    def steps(self) -> Op:
        return merge_pipelines(
            self.first.get_pipeline(),
            self.second._tagged_steps(),
            self._wires,
        )


ProtocolT = TypeVar("ProtocolT", bound=Type[Protocol])


def resources(
    *,
    gpus: float = 0,
    min_vram: Optional[float] = None,
    cpus: Optional[float] = None,
    task: TaskType = TaskType.EPHEMERAL,
) -> Callable[[ProtocolT], ProtocolT]:
    """Declare the compute resources of a protocol's stages.

    Every call of the protocol's ``map`` and ``map_element`` functions
    requires the resources::

        @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
        class Inference(B.Protocol): ...

    See :class:`ComputeResources` for the arguments.
    """
    compute = ComputeResources(gpus=gpus, min_vram=min_vram, cpus=cpus, task=task)

    def decorate(cls: ProtocolT) -> ProtocolT:
        if not issubclass(cls, Protocol):
            raise TypeError(
                f"@resources applies to Protocol classes, got '{cls.__qualname__}'.",
            )

        cls.compute_resources = compute
        return cls

    return decorate
