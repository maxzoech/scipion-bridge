import inspect
import ast
import autopep8 # type: ignore

from typing import Any

def parse_ast(cls: Any) -> ast.AST:
    source = inspect.getsource(cls)
    source = autopep8.fix_code(source)

    return ast.parse(source)