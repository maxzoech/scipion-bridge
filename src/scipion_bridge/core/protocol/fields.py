from typing import Any, Callable, Optional, TypeVar, overload, Self
from dependency_injector.wiring import Provide, inject

from ..environment.protocol_config import ProtocolConfigurationProvider
from ..environment.resource_provider import ResourceProvider, ResourceScope
from ..streaming.ops import Source
from ..utils.marker import Marker

T = TypeVar("T")
FieldSelf = TypeVar("FieldSelf", bound="Field")
InputSelf = TypeVar("InputSelf", bound="Input")


class BoundField(Marker[T]):

    def __init__(
        self,
        name: str,
        *,
        dtype: Optional[Any] = None,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        group: Optional[str] = None,
        help: Optional[str] = None,
    ):
        Marker.__init__(self, dtype=dtype)
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
        config_provider: ProtocolConfigurationProvider = Provide[
            "protocol_config_provider"
        ],
    ) -> T:
        assert self.name is not None
        return config_provider.get_value(self.name, default=self.default)


class Field(Marker[T]):

    def __set_name__(self, owner: Any, name: str) -> None:
        self._bound_name = name
        super().__set_name__(owner, name)

    def __init__(
        self,
        *,
        dtype: Optional[Any] = None,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        group: Optional[str] = None,
        help: Optional[str] = None,
    ):
        Marker.__init__(self, dtype=dtype)

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
            dtype=self.dtype,
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
        dtype: Optional[Any] = None,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        help: Optional[str] = None,
    ):
        BoundField.__init__(
            self,
            name=name,
            dtype=dtype,
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
        dtype: Optional[Any] = None,
        default: Optional[T] = None,
        optional: Optional[bool] = None,
        label: Optional[str] = None,
        help: Optional[str] = None,
    ):
        Field.__init__(
            self,
            dtype=dtype,
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
            dtype=self.dtype,
            default=self.default,
            optional=self.optional,
            label=self.label,
            help=self.help,
        )


class Resource(Marker[T]):
    """Marker and descriptor for actor-scoped protocol resources."""

    def __set_name__(self, owner: Any, name: str) -> None:
        self._bound_name = name
        self.name = name
        super().__set_name__(owner, name)

    def __init__(
        self,
        *,
        builder: Callable[[Any], T],
        scope: ResourceScope = ResourceScope.PROCESS,
        dtype: Optional[Any] = None,
    ):
        Marker.__init__(self, dtype=dtype)
        if not callable(builder):
            raise TypeError("Resource builder must be a callable.")
        self.builder = builder
        self.scope = scope
        self.name: Optional[str] = None

    @overload
    def __get__(self: Self, instance: None, owner: Any) -> Self: ...

    @overload
    def __get__(self, instance: Any, owner: Any) -> T: ...

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self

        return self._get_resource(instance)

    @inject
    def _get_resource(
        self,
        instance: Any,
        provider: ResourceProvider = Provide["resource_provider"],
    ) -> T:
        assert self.name is not None

        if not isinstance(provider, ResourceProvider):
            raise NotImplementedError(f"Resources are not supported by this backend.")
            return self.builder(instance)

        return provider.get_resource(
            self.name,
            self.builder,
            instance,
            scope=self.scope,
            dtype=self.dtype,
        )

    def __set__(self, instance: Any, value: Any) -> None:
        raise AttributeError(
            f"Resource '{self.name}' is read-only and cannot be modified."
        )
