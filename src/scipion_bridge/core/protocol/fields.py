from dataclasses import dataclass

from typing import Generic, TypeVar, Optional, Type, overload, Any
from enum import Enum

T = TypeVar("T")

from ..streaming.ops import Source


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
        self.group = group
        self.help = help


    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Field):
            return False
        return (
            self.default == other.default
            and self.optional == other.optional
            and self.label == other.label
            and getattr(self, "group", None) == getattr(other, "group", None)
            and self.help == other.help
        )




class Input(Field, Generic[T], Source):

    def __set_name__(self, owner, name):
        self.name = name

    def __init__(
        self,
        *,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        help: Optional[str] = None,
    ):
        Field.__init__(
            self,
            default=default,
            optional=optional,
            group="Input",
            label=label,
            help=help,
        )

        Source.__init__(self)

    def __hash__(self):
        return Source.__hash__(self)