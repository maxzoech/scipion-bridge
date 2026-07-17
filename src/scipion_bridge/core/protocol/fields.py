from dataclasses import dataclass

from typing import Generic, TypeVar, Optional, Type, overload, Any
from enum import Enum

T = TypeVar("T")


@dataclass
class Field(Generic[T]):

    def __init__(
        self,
        *,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        group: Optional[str] = None,
        help: Optional[str] = None,
    ):
        super().__init__()

        if optional is None:
            optional = default is None

        self.default = default
        self.optional = optional
        self.label = label
        self.help = help


@dataclass
class Input(Field, Generic[T]):

    def __init__(
        self,
        *,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        help: Optional[str] = None,
    ):
        super().__init__(
            default=default, optional=optional, group="Input", label=label, help=help,
        )
