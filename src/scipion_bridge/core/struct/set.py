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

from .struct import Struct, Dim
from .storage import SchemaArrayStorage
from .schema import Entry, SchemaConvertible, Schema, _SchemaSetEntry
from ..utils.marker import Marker

T = TypeVar("T", bound=Struct)


class Set(Marker[T], SchemaArrayStorage, SchemaConvertible):

    def __init__(
        self,
        *,
        capacity: Optional[Union[int, Dim]] = None,
        dtype: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> None:
        
        item = kwargs.pop("_item", None)
        super().__init__(dtype=dtype, **kwargs)

        if self.dtype is None:
            raise TypeError(
                "Cannot convert unsubscripted Set to a schema. "
                "Please provide an element type (e.g., Set[Struct] or dtype=Struct)."
            )

        if not (isinstance(self.dtype, type) and issubclass(self.dtype, Struct)):
            raise TypeError(f"Element of a set has to be of type Struct, got '{self.dtype}'")

        self._capacity = capacity

        context = { k: Dim.new(v, name=k) for k, v in kwargs.items() }
        self._item = item if item is not None else self.dtype(**context)

        assert isinstance(self._item, Struct)
        
        item_schema = self._item.schema
        assert item_schema is not None
        
        self.schema = item_schema.to_set_schema(capacity=self.capacity)

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity.resolve_value() if isinstance(self._capacity, Dim) else self._capacity

    def convert_to_entry(self) -> Entry:
        if self.dtype is None or self.schema is None:
            raise TypeError("Cannot convert unsubscripted Set to schema entry.")
        return _SchemaSetEntry(
            schema=self.schema,
            capacity=self.capacity,
        )

    def is_static(self) -> bool:
        assert self.schema is not None
        return self.schema.is_static

    def specialize(self, context: Optional[Dict[Any, Any]] = None) -> "Set[T]":
        if context is None:
            context = {}

        # Specialize the capacity Dim if it is dynamic.
        specialized_capacity = (
            self._capacity.infer(context) if isinstance(self._capacity, Dim)
            else context.get(self._capacity, self._capacity)
        )

        # Specialize the item with calling context
        specialized_item = self._item.specialize(context)

        return type(self)(
            capacity=specialized_capacity,
            dtype=self.dtype,
            _item=specialized_item,
            **self.options,
        )

    def validate(self, other: Any) -> None:
        if not isinstance(other, Set):
            raise TypeError(f"Expected Set instance, got '{type(other).__name__}'.")

    def default(self) -> "Set[T]":
        return self.specialize({})