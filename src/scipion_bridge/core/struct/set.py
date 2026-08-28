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

from .struct import Struct, Arg
from .storage import SchemaArrayStorage
from .schema import Entry, SchemaConvertible, Schema, _SchemaSetEntry
from ..utils.marker import Marker

T = TypeVar("T", bound=Struct)


class Set(Marker[T], SchemaArrayStorage, SchemaConvertible):

    _bridge_schema: Schema

    def __init__(
        self,
        capacity: Optional[Union[int, Arg]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._capacity = capacity

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
        return self._capacity.value if isinstance(self._capacity, Arg) else self._capacity

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
