import os
import sys
import re
from dataclasses import dataclass
from subprocess import Popen, PIPE
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from ..environment.container import Container
from ..environment.cmd_exec import ShellExecProvider

import ast
import inspect
import autopep8  # type: ignore
from typing import Dict, Any, Callable, Optional, Set, List, Protocol, TypeVar, overload

import itertools
import functools
from functools import partial

from .func_params import extract_func_params

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Domain:
    name: str
    command: List[str]
    isolated: bool = False

    @classmethod
    def default(cls) -> "Domain":
        return Domain(
            name="Default",
            command=[],
            isolated=False
        )


class ShellDecoratorProtocol(Protocol):

    @overload
    def __call__(
        self,
        func: F,
        *,
        name: Optional[str] = None,
        postprocess_fn: Optional[Callable] = None,
        **args_map,
    ) -> F:
        ...

    @overload
    def __call__(
        self,
        func: None = None,
        *,
        name: Optional[str] = None,
        postprocess_fn: Optional[Callable] = None,
        **args_map,
    ) -> "ShellDecoratorProtocol":
        ...

    def __call__(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        postprocess_fn: Optional[Callable] = None,
        **args_map,
    ) -> Any:
        ...


@overload
def shell_command(
    f: F,
    *,
    domain: Domain = Domain.default(),
    name: Optional[str] = None,
    postprocess_fn: Optional[Callable] = None,
    **args_map,
) -> F:
    ...


@overload
def shell_command(
    f: None = None,
    *,
    domain: Domain = Domain.default(),
    name: Optional[str] = None,
    postprocess_fn: Optional[Callable] = None,
    **args_map,
) -> ShellDecoratorProtocol:
    ...


def shell_command(
    f: Optional[Callable] = None,
    *,
    domain: Domain = Domain.default(),
    name: Optional[str] = None,
    postprocess_fn: Optional[Callable] = None,
    **args_map,
) -> Any:
    def _wrap(func: Optional[Callable] = None, **new_args) -> Any:
        if new_args:
            merged_name = new_args.pop("name", name)
            merged_postprocess_fn = new_args.pop("postprocess_fn", postprocess_fn)
            merged_args_map = {**args_map, **new_args}
            return shell_command(
                func,
                domain=domain,
                name=merged_name,
                postprocess_fn=merged_postprocess_fn,
                **merged_args_map,
            )

        if func is not None:
            return _shell_command_wrapper(
                func,
                domain,
                name,
                postprocess_fn,
                args_map,
            )
        return _wrap

    if f is None:
        return _wrap
    return _wrap(f)


def _func_is_empty(func):

    source = inspect.getsource(func)
    source = autopep8.fix_code(source)

    tree = ast.parse(source)

    # Find the function definition in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body

            return len(body) == 1 and isinstance(body[0], ast.Pass)

    return False  # In case no FunctionDef was found


def _param_to_cmd_args(
    param: inspect.Parameter, value: Any, args_map, boolean_params: Set[str]
):

    k = param.name
    arg_name = args_map[k] if k in args_map else k
    mapped_boolean_params = {args_map[p] if p in args_map else p for p in boolean_params}

    if arg_name in mapped_boolean_params:
        return (
            [f"--{arg_name}"] if value else []
        )  # Use `if value` to support implicit booleaness of Python
    else:
        is_keyword = param.kind == inspect.Parameter.KEYWORD_ONLY
        prefix = "--" if is_keyword else "-"

        return [prefix + arg_name, str(value)]


def _shell_command_wrapper(
    f,
    domain: Domain,
    name=None,
    postprocess_fn=None,
    args_map=None,
):

    is_empty = _func_is_empty(f)
    if not is_empty:
        raise RuntimeError(
            f"Forward declared external scipion function {f.__name__} must be only contain a single pass statement."
        )

    if args_map is None:
        args_map = {}

    func_name = name if name is not None else f.__name__

    signature = inspect.signature(f)
    params = signature.parameters
    boolean_params = {k for k, v in f.__annotations__.items() if v is bool}

    pos_args = {
        k
        for k, v in params.items()
        if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    if boolean_params.intersection(pos_args):
        raise RuntimeError("Positional arguments cannot be declared as boolean flags")

    run_args: Dict[str, Any] = {}
    run_args.setdefault("shell", True)
    run_args["stderr"] = PIPE

    @functools.wraps(f)
    @inject
    def wrapper(
        *args,
        __scipion_bridge_runner__: ShellExecProvider = Provide[Container.shell_exec],
        **kwargs,
    ):
        _ = f(
            *args, **kwargs
        )  # Call function for Python to throw error if args and kwargs aren't passed correctly

        merged_args = extract_func_params(args, kwargs, signature)

        # Filter args that are None to support optional arguments
        merged_args = {k: v for k, v in merged_args.items() if v is not None}

        raw_args = [
            _param_to_cmd_args(p, v, args_map, boolean_params)
            for p, v in merged_args.items()
        ]
        if postprocess_fn is not None:
            raw_args = postprocess_fn(raw_args)

        raw_args = list(itertools.chain.from_iterable(raw_args))
        raw_args = domain.command + [func_name, *raw_args]

        print(f"Runner: {__scipion_bridge_runner__}")

        return __scipion_bridge_runner__(func_name, domain, raw_args, run_args)

    return wrapper
