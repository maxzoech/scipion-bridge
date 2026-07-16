from __future__ import annotations
import inspect
import autopep8  # type: ignore
import ast
import abc

from dataclasses import dataclass
from itertools import chain
from typing import get_type_hints, get_origin, Any, OrderedDict

from .fields import Field


@dataclass
class _ProtocolInfo:
    inputs: OrderedDict[str, Any]
    states: OrderedDict[str, Any]
    configuration: OrderedDict[str, Any]


class Protocol(metaclass=abc.ABCMeta):

    _exec_info: _ProtocolInfo

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls._exec_info = _find_protocol_info(cls)

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        obj_id = hex(id(self))

        def _fmt_type(t: Any) -> str:
            if hasattr(t, "__name__"):
                return t.__name__
            
            return "<Unnamed Type>"

        inputs_lines = []
        for k, v in self._exec_info.inputs.items():
            inputs_lines.append(f"    {k} ({_fmt_type(v)}):")
        for k, v in self._exec_info.configuration.items():
            inputs_lines.append(f"    {k} ({_fmt_type(v)}):")
        inputs_str = "\n".join(inputs_lines)

        states_str = ", ".join([f"{k}({_fmt_type(v)})" for k, v in self._exec_info.states.items()])

        return f"""<{class_name} object at {obj_id}>

Inputs:
{inputs_str}

States: {states_str}"""

    def _defineParams(self, form: Any) -> None:
        try:
            from pyworkflow.protocol.params import (
                StringParam, IntParam, FloatParam, BooleanParam
            )
        except ImportError:
            raise ImportError(
                "Defining Scipion parameters requires pyworkflow. "
                "Install it using: pip install \"scipion-bridge[pyworkflow]\""
            )

        from typing import get_args, get_origin, Union

        def _get_underlying_type(hint: Any) -> Any:
            origin = get_origin(hint)
            if origin == Field:
                args = get_args(hint)
                return args[0] if args else Any
            elif isinstance(hint, type) and issubclass(hint, Field):
                if hasattr(hint, "__orig_bases__") and hint.__orig_bases__:
                    args = get_args(hint.__orig_bases__[0])
                    return args[0] if args else Any
            return Any

        def _map_type(dtype: Any) -> Any:
            if get_origin(dtype) is Union:
                args = [a for a in get_args(dtype) if a is not type(None)]
                if args:
                    dtype = args[0]

            if dtype is str:
                return StringParam
            elif dtype is int:
                return IntParam
            elif dtype is float:
                return FloatParam
            elif dtype is bool:
                return BooleanParam
            else:
                return StringParam

        hints = get_type_hints(self.__class__)
        
        fields = OrderedDict()
        for name, hint in hints.items():
            is_field = False
            if get_origin(hint) == Field or hint == Field:
                is_field = True
            elif isinstance(hint, type) and issubclass(hint, Field):
                is_field = True

            if is_field:
                field_obj = getattr(self.__class__, name, None)
                if isinstance(field_obj, Field):
                    underlying_type = _get_underlying_type(hint)
                    fields[name] = (field_obj, underlying_type)

        for name, (field, dtype) in fields.items():
            param_cls = _map_type(dtype)
            param_kwargs = {}

            if field.default is not None:
                param_kwargs["default"] = field.default
            if field.help is not None:
                param_kwargs["help"] = field.help

            form.addParam(name, param_cls, **param_kwargs)

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any):
        pass


def _find_protocol_info(cls: type[Protocol]) -> _ProtocolInfo:
    def _is_run_method(el: ast.AST) -> bool:
        if not isinstance(el, ast.FunctionDef):
            return False

        return el.name == "run"

    attributes = get_type_hints(cls)

    source = inspect.getsource(cls)
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
            raise RuntimeError(
                f"Method run() in {cls.__qualname__} has variational arguments"
            )

        padding = len(arg_def.args[1:]) - len(arg_def.defaults)
        defaults = [None] * padding + list(arg_def.defaults)

        args = zip(arg_def.args[1:], defaults)
        kwargs = zip(arg_def.kwonlyargs, arg_def.kw_defaults)

        input_types = get_type_hints(cls.run)
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
            raise RuntimeError(
                f"Protocol inputs {invalid_inputs_list} in {cls.__qualname__} either do not have type annotations or a default value."
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

    configuration = OrderedDict()
    states = OrderedDict()
    for name in typed_assign_ops:
        value = attributes[name]

        is_field = False
        if get_origin(value) == Field or value == Field:
            is_field = True
        elif isinstance(value, type) and issubclass(value, Field):
            is_field = True

        if is_field:
            if name in inputs:
                inputs[name] = value
            else:
                configuration[name] = value
        else:
            states[name] = value

    return _ProtocolInfo(
        inputs=inputs,
        states=states,
        configuration=configuration,
    )
