import inspect
import autopep8  # type: ignore
import ast
import abc

from dataclasses import dataclass
from collections import OrderedDict
from typing import get_type_hints, get_origin

from typing import Generic, TypeVar, Optional, Literal, overload, Any
from enum import Enum

T = TypeVar("T", str, int, float, Enum)


class Field(Generic[T]):

    @overload
    def __init__(self, default: T, optional: Literal[False] = False):
        ...

    @overload
    def __init__(self, default: Optional[T] = None, optional: Literal[True] = True):
        ...

    def __init__(self, default: Optional[T] = None, optional: Optional[bool] = None):
        super().__init__()

        if optional is None:
            optional = default is None

        self.default: Optional[T] = default
        self.optional: bool = optional

class Protocol(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any):
        pass


class ProtocolExecutionDescription:

    def __init__(
        self,
        protocol: Protocol,
        *,
        inputs: OrderedDict,
        configuration: OrderedDict,
        states: OrderedDict,
    ):
        
        self.protocol = protocol
        self.inputs = inputs
        self.configuration = configuration
        self.states = states


def create_protocol(protocol: Protocol) -> ProtocolExecutionDescription:
    def _is_run_method(el: ast.AST) -> bool:
        if not isinstance(el, ast.FunctionDef):
            return False
        
        return el.name == "run"
        
    attributes = get_type_hints(type(protocol))

    source = inspect.getsource(type(protocol))
    source = autopep8.fix_code(source)

    tree = ast.parse(source)
    class_def = tree.body[0]
    assert isinstance(class_def, ast.ClassDef), "Protocol must be a class"

    # Find inputs in run method
    inputs = OrderedDict()
    run_method_defs = [el for el in class_def.body if _is_run_method(el)]

    if len(run_method_defs) > 0:
        run_method_def = run_method_defs[0]
        assert isinstance(run_method_def, ast.FunctionDef)

        arg_def = run_method_def.args
        if arg_def.vararg is not None or arg_def.kwarg is not None:
            raise RuntimeError("Variational arguments are not supported in run method")

        args = list(zip(arg_def.args[1:], arg_def.defaults))
        kwargs = list(zip(arg_def.kwonlyargs, arg_def.kw_defaults))

        input_types = get_type_hints(protocol.run)
        invalid_inputs = []
        for arg, default in [*args, *kwargs]:
            if arg.annotation is None:
                invalid_inputs.append(arg.arg)

            elif default is not None:
                invalid_inputs.append(arg.arg)

            else:
                inputs[arg.arg] = input_types[arg.arg]

        if len(invalid_inputs) > 0:
            invalid_inputs_list = ", ".join(invalid_inputs)
            raise RuntimeError(f"Protocol inputs {invalid_inputs_list} either do not have type annotations or a default value.")


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
            f"The protocol states {invalid_state_list} do not have type annotations."
        )

    # Configure states
    typed_assign_ops = [
        a.target.id
        for a in class_def.body
        if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)
    ]

    configuration = OrderedDict()
    states = OrderedDict()
    for name in typed_assign_ops:
        value = attributes[name]

        if get_origin(value) == Field:
            if name in inputs:
                inputs[name] = value
            else:
                configuration[name] = value
        else:
            states[name] = value

    return ProtocolExecutionDescription(
        protocol,
        inputs=inputs,
        configuration=configuration,
        states=states,
    )