import inspect
import autopep8  # type: ignore
import ast
import abc

from itertools import zip_longest, chain
from collections import OrderedDict
from typing import get_type_hints, get_origin, get_args

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

class ProtocolExecutionDescription:

    def __init__(
        self,
        protocol: "Protocol",
        *,
        inputs: OrderedDict,
        configuration: OrderedDict,
        states: OrderedDict,
    ):
        
        self.protocol = protocol
        self.inputs = inputs
        self.configuration = configuration
        self.states = states


class ProtocolMetaclass(abc.ABCMeta):

    def __call__(cls, *args: Any, **kwargs: Any) -> ProtocolExecutionDescription:
        instance = super().__call__(*args, **kwargs)
        return create_protocol(instance)


class Protocol(metaclass=ProtocolMetaclass):

    def __new__(cls, *args: Any, **kwargs: Any) -> ProtocolExecutionDescription:  # type: ignore[misc]
        return super().__new__(cls)  # type: ignore

    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any):
        pass


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
            raise RuntimeError(f"Method run() in {type(protocol).__qualname__} has variational arguments")

        args = zip_longest(arg_def.args[1:], arg_def.defaults)
        kwargs = zip_longest(arg_def.kwonlyargs, arg_def.kw_defaults)

        input_types = get_type_hints(protocol.run)
        invalid_inputs = []
        for arg, default in chain(args, kwargs):
            if arg.annotation is None:
                invalid_inputs.append(arg.arg)
            elif default is not None:
                invalid_inputs.append(arg.arg)
            else:
                inputs[arg.arg] = input_types[arg.arg]

        if len(invalid_inputs) > 0:
            invalid_inputs_list = ", ".join(invalid_inputs)
            raise RuntimeError(f"Protocol inputs {invalid_inputs_list} in {type(protocol).__qualname__} either do not have type annotations or a default value.")


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
            f"The protocol states {invalid_state_list} in {type(protocol).__qualname__} do not have type annotations."
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
                if not inputs[name] == get_args(value)[0]:
                    raise TypeError(f"Type of input '{name}' in {type(protocol).__qualname__} must match with declared field")

                inputs[name] = value
            else:
                configuration[name] = value
        else:
            if name in inputs:
                raise TypeError(f"Input '{name}' in {type(protocol).__qualname__} cannot also be declared as a protocol state") 

            states[name] = value

    print("Inputs: ", inputs)
    return ProtocolExecutionDescription(
        protocol,
        inputs=inputs,
        configuration=configuration,
        states=states,
    )