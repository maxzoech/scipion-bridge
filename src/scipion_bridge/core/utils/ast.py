import inspect
import ast
import textwrap

from typing import Any

def parse_ast(cls: Any) -> ast.AST:
    source = inspect.getsource(cls)
    source = textwrap.dedent(source)

    return ast.parse(source)