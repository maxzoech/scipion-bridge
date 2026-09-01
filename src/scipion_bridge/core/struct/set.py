"""Set container implementation.

A Set is a sequence container for Struct items, stored as flattened
N-dimensional arrays across the outer batch dimension.
"""

from __future__ import annotations
from typing import (
    Type,
    TypeVar,
    Dict,
    Union,
    Any,
    Optional,
)

from numpy.typing import ArrayLike

from .struct import Struct, Arg, Trait
from .schema import Entry, SchemaConvertible, Schema, _SchemaSetEntry
from .storage import ArrayStorage
from ..utils.marker import Marker

T = TypeVar("T", bound=Struct)


class BoundSetView(SchemaConvertible):
    """Read-only view returned when accessing a Set attribute on a Struct class."""

    def __init__(
        self,
        element_cls: Type[Struct],
        capacity_spec: Arg,
        owner_cls: Type[Trait],
    ) -> None:
        self._element_cls = element_cls
        self._capacity_spec = capacity_spec
        self._owner_cls = owner_cls

    @classmethod
    def default(cls) -> "BoundSetView":
        raise ValueError("Cannot instantiate default BoundSetView directly.")

    @classmethod
    def schema(cls) -> Schema:
        raise NotImplementedError

    @property
    def capacity(self) -> Optional[int]:
        if not self._capacity_spec.name:
            return self._capacity_spec.value

        target = getattr(self._owner_cls, self._capacity_spec.name, self._capacity_spec)
        return target.value if isinstance(target, Arg) else target

    def convert_to_entry(self) -> Entry:
        set_schema = self._element_cls.schema().to_set_schema(capacity=self.capacity)
        return _SchemaSetEntry(
            schema=set_schema,
            capacity=self.capacity,
        )


class Set(Marker[T], SchemaConvertible):

    _bridge_schema: Schema
    _capacity: Arg

    def __init__(
        self,
        capacity: Optional[Union[int, Arg]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._capacity = Arg.new(capacity)
        self._storage = ArrayStorage(schema=self._bridge_schema)

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None and owner is not None:
            assert self.dtype is not None
            return BoundSetView(
                element_cls=self.dtype,
                capacity_spec=self._capacity,
                owner_cls=owner,
            )
        return self

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls._dtype is None:
            raise TypeError(
                "Cannot convert unsubscripted Set to a schema. "
                "Please provide an element type (e.g., Set[Struct] or dtype=Struct)."
            )

        if not (isinstance(cls._dtype, type) and issubclass(cls._dtype, Struct)):
            raise TypeError(f"Element of a set has to be a Struct, got '{cls._dtype}'")

        cls._bridge_schema = cls._dtype._bridge_schema.to_set_schema()

    @classmethod
    def schema(cls) -> Schema:
        return cls._bridge_schema

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity.value

    def convert_to_entry(self) -> Entry:
        if self.dtype is None or self.schema is None:
            raise TypeError("Cannot convert unsubscripted Set to schema entry.")
        return _SchemaSetEntry(
            schema=self.schema(),
            capacity=self.capacity,
        )

    @classmethod
    def default(cls) -> "Set[T]":
        if cls._dtype is None:
            raise TypeError(
                "Cannot convert unsubscripted Set to a schema. "
                "Please provide an element type (e.g., Set[Struct] or dtype=Struct)."
            )
        
        return cls()


    def __setitem__(self, key, value):
        self._storage[key] = value


    def __getitem__(self, key: Union[int, slice, str]):
        return self._storage[key]
