from typing import Generic, TypeVar, Optional, Literal, overload
from enum import Enum

T = TypeVar("T")

class Field(Generic[T]):

    @overload
    def __init__(self, default: T, optional: Literal[False] = False, help: Optional[str] = None): ...

    @overload
    def __init__(self, default: Optional[T] = None, optional: Literal[True] = True, help: Optional[str] = None): ...

    def __init__(self, default: Optional[T] = None, optional: Optional[bool] = None, help: Optional[str] = None):
        super().__init__()

        if optional is None:
            optional = default is None

        self.default: Optional[T] = default
        self.optional: bool = optional
        self.help: Optional[str] = help

