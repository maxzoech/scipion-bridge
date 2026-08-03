from enum import Enum
from dataclasses import dataclass
from dependency_injector.wiring import Provide, inject

from ..environment.protocol_config import ProtocolConfigurationProvider

from typing import Generic, TypeVar, Optional, Type, overload, Any

T = TypeVar("T")

FieldSelf = TypeVar("FieldSelf", bound="Field")
InputSelf = TypeVar("InputSelf", bound="Input")

from ..streaming.ops import Source


from ...backend.standalone.container import Container


class BoundField(Generic[T]):

    def __init__(
        self,
        name: str,
        *,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        group: Optional[str] = None,
        help: Optional[str] = None,
    ):
        self.name = name
        self.default = default
        self.optional = optional
        self.label = label
        self.group = group
        self.help = help

    @property
    def value(self) -> T:
        """Return the value of the field, or None if not set."""
        return self._get_value()

    @inject
    def _get_value(
        self,
        config_provider: ProtocolConfigurationProvider = Provide[Container.protocol_config_provider],
    ) -> T:
        return config_provider.get_value(self.name, default=self.default)


class Field(Generic[T]):

    def __set_name__(self, owner: Any, name: str) -> None:
        self._bound_name = name

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

    @overload
    def __get__(self: FieldSelf, instance: None, owner: Any) -> FieldSelf: ...

    @overload
    def __get__(self, instance: Any, owner: Any) -> BoundField[T]: ...

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self

        return BoundField(
            name=self._bound_name,
            default=self.default,
            optional=self.optional,
            label=self.label,
            group=self.group,
            help=self.help,
        )

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


class BoundInput(BoundField[T], Source):

    def __init__(
        self,
        name: str,
        *,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        help: Optional[str] = None,
    ):
        BoundField.__init__(
            self,
            name=name,
            default=default,
            optional=optional,
            label=label,
            group="Input",
            help=help,
        )
        Source.__init__(self, name=name)

    def __hash__(self) -> int:
        return Source.__hash__(self)


class Input(Field[T]):

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

    @overload
    def __get__(self: InputSelf, instance: None, owner: Any) -> InputSelf: ...

    @overload
    def __get__(self, instance: Any, owner: Any) -> BoundInput[T]: ...

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self

        return BoundInput(
            name=self._bound_name,
            default=self.default,
            optional=self.optional,
            label=self.label,
            help=self.help,
        )