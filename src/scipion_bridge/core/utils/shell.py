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
from typing import Dict, Any, Callable, Optional, Set, List

import itertools
import functools
from functools import partial

from .func_params import extract_func_params


@dataclass
class Domain:
    name: str
    command: List[str]


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
    param = param.replace(name=args_map[k]) if k in args_map else param
    boolean_params = {args_map[p] if p in args_map else p for p in boolean_params}

    if param.name in boolean_params:
        return (
            [f"--{param.name}"] if value else []
        )  # Use `if value` to support implicit booleaness of Python
    else:
        is_keyword = param.kind == inspect.Parameter.KEYWORD_ONLY
        prefix = "--" if is_keyword else "-"

        return [prefix + param.name, str(value)]


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


def shell_command(
    f: Optional[Callable] = None,
    *,
    domain: Domain,
    name: Optional[str] = None,
    postprocess_fn: Optional[Callable] = None,
    **args_map,
):
    def _wrap(func: Callable) -> Callable:
        return _shell_command_wrapper(
            func,
            domain,
            name,
            postprocess_fn,
            args_map,
        )

    if f is None:
        return _wrap
    return _wrap(f)
