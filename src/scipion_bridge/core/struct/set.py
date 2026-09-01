"""Set container implementation.

A Set is a sequence container for Struct items, stored as flattened
N-dimensional arrays across the outer batch dimension.
"""

from __future__ import annotations
from typing import (
    Self,
    Type,
    TypeVar,
    Dict,
    Union,
    Any,
    Optional,
    Tuple,
    overload,
)

from numpy.typing import ArrayLike

from .struct import Struct, Arg, Trait
from . import schema
from .schema import Entry, SchemaConvertible, Schema, _SchemaSetEntry
from .storage import _BaseStorage, ArrayStorage, ArrayStorageView
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

        storage = kwargs.get("_storage_view", ArrayStorage(schema=self._bridge_schema))
        assert isinstance(storage, _BaseStorage)

        self._capacity = Arg.new(capacity)
        self._storage = storage

    def __set_name__(self, owner: Type[Struct], name: str) -> None:
        # When we the Set is used as a descriptor, delete the storage
        # definition. This is a bit hacky but the only way to use Set[<Struct>]
        # both inside a struct and as a standalone class.
        del self._storage

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

    def _normalize_slice(self, index: slice) -> Tuple[int, int]:
        if index.step is not None and index.step != 1:
            raise ValueError("Slice step other than 1 is not supported.")

        if self.capacity is not None:
            start, stop, _ = index.indices(self.capacity)
            return start, stop

        # Capacity is unknown / dynamic
        if index.start is not None and index.start < 0:
            raise ValueError(
                f"Cannot resolve negative slice start '{index.start}' on a Set with unknown capacity."
            )
        start = index.start if index.start is not None else 0

        if index.stop is None:
            raise ValueError(
                f"Cannot resolve open-ended slice '{index}' on a Set with unknown capacity. A stop index must be specified."
            )
        if index.stop < 0:
            raise ValueError(
                f"Cannot resolve negative slice stop '{index.stop}' on a Set with unknown capacity."
            )
        stop = index.stop

        if start > stop:
            start = stop

        return start, stop

    def __setitem__(self, key, value):
        if isinstance(key, str):
            fields = self.schema().fields
            if key not in fields:
                raise ValueError

            entry = fields[key]
            if isinstance(entry, schema._ArrayEntryBase) and entry.is_static:
                self._storage.write_static_array(key, entry, value)
            else:
                raise NotImplementedError

    @overload
    def __getitem__(self, key: str) -> ArrayLike: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...
    
    def __getitem__(self, key) -> Union[ArrayLike, Set]:

        if isinstance(key, str):
            fields = self.schema().fields
            if key not in fields:
                raise ValueError

            entry = fields[key]
            if isinstance(entry, schema._ArrayEntryBase) and entry.is_static:
                return self._storage.read_static_array(key, entry)
            
        elif isinstance(key, slice):
            assert self.dtype is not None
            assert self.capacity is not None

            indices = self._normalize_slice(key)
            start, stop = indices
            new_size = stop - start

            if self.schema().is_static:
                base_start = self._storage.offset[0][0] if self._storage.offset else 0

                new_offset = (base_start + start, base_start + stop)
                base_offset = self._storage.offset or (new_offset,)
                
                subview = ArrayStorageView(
                    self.schema(),
                    parent=self._storage.parent or self._storage, 
                    root=self._storage.root,
                    offset=(new_offset, *base_offset[1:])
                )
            else:
                raise NotImplementedError("Sclicing ragged sets is not supported yet")

            return type(self)(capacity=new_size, _storage_view=subview)

        raise NotImplementedError
