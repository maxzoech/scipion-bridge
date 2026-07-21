import ast
from typing import overload, Union, Any, Type

from .ast import parse_ast

@overload
def has_untyped_class_definitions(cls: ast.AST) -> bool:
    ...

@overload
def has_untyped_class_definitions(cls: Type[Any]) -> bool:
    ...

def has_untyped_class_definitions(cls: Union[ast.AST, Type[Any]]) -> bool:
    
    if isinstance(cls, ast.AST):
        tree = cls
    else:
        tree = parse_ast(cls)
        
    assert isinstance(tree, ast.Module),  "Parser results must be module"
    class_def = tree.body[0]
    assert isinstance(class_def, ast.ClassDef)

    # Check if the user has defined state without a type annotation
    untyped_assign_ops = [
        target.id
        for a in class_def.body
        if isinstance(a, ast.Assign)
        for target in a.targets
        if isinstance(target, ast.Name)
    ]

    return len(untyped_assign_ops) > 0