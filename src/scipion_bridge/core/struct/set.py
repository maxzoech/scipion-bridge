"""Set container implementation.

A Set is a sequence container for Struct items, stored as flattened
N-dimensional arrays across the outer batch dimension.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

from .struct import Struct, Arg, Trait
from .schema import (
    Entry,
    KeyPath,
    SchemaConvertible,
    Schema,
    ArrayEntryBase,
    SchemaEntry,
    SchemaSetEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
    _SchemaSetEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaEntry,
)
from .storage import _BaseStorage, ArrayStorage, ArrayStorageView, _lookup_entry
from .utils.arrow_utils import RaggedArrayView
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
        raise NotImplementedError("Cannot get schema directly from BoundSetView.")

    @property
    def capacity(self) -> Optional[int]:
        if not self._capacity_spec.name:
            return self._capacity_spec.value

        target = getattr(self._owner_cls, self._capacity_spec.name, self._capacity_spec)
        return target.value if isinstance(target, Arg) else target

    def convert_to_entry(self) -> Entry:
        set_schema = self._element_cls.schema().to_set_schema(capacity=self.capacity)
        return SchemaSetEntry(
            schema=set_schema,
            capacity=self.capacity,
        )


class Set(Marker[T], SchemaConvertible):
    """Sequence container for Struct instances backed by Apache Arrow columnar storage."""

    _bridge_schema: Schema
    _capacity: Arg

    @overload
    def __init__(self, capacity: Sequence[T], **kwargs: Any) -> None: ...

    @overload
    def __init__(self, capacity: Optional[Union[int, Arg]] = None, **kwargs: Any) -> None: ...

    def __init__(
        self,
        capacity: Optional[Union[int, Arg, Sequence[Struct]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        items_to_populate: Optional[Sequence[Struct]] = None

        if capacity is not None and isinstance(capacity, Sequence) and not isinstance(capacity, (str, bytes)):
            items_to_populate = capacity
            derived_cap = len(items_to_populate)
            self._capacity = Arg.new(derived_cap)
            cap_val = derived_cap
        else:
            self._capacity = Arg.new(capacity)  # type: ignore
            cap_val = self._capacity.value

        storage = kwargs.get(
            "_storage_view",
            ArrayStorage(schema=self.schema(), capacity=cap_val),
        )
        if not isinstance(storage, _BaseStorage):
            raise TypeError(f"Expected _BaseStorage instance, got {type(storage).__name__}")
        self._storage = storage

        if items_to_populate is not None:
            for idx, item in enumerate(items_to_populate):
                self[idx] = item

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)
        del self._storage

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None and owner is not None:
            if self.dtype is None:
                raise TypeError("Cannot create BoundSetView without a defined dtype.")
            return BoundSetView(
                element_cls=self.dtype,
                capacity_spec=self._capacity,
                owner_cls=owner,
            )
        if isinstance(instance, Struct):

            _schema = self.schema()
            assert isinstance(_schema.dtype, type) and issubclass(_schema.dtype, Struct)
            
            subview = ArrayStorageView(
                self.schema(),
                parent=instance._storage.parent or instance._storage,
                path=(*instance._storage.path, self.name),
                offset=instance._storage.offset,
            )

            return Set[_schema.dtype](_storage_view=subview)

            raise NotImplementedError(
                f"Accessing nested Set{name_str} on a Struct instance is not supported yet."
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

    def __len__(self) -> int:
        if self.capacity is not None:
            return self.capacity
        for key, entry in self.schema().tree_iter():
            if key in self._storage:
                val = self._storage.read(key, entry)
                return len(val)
        return 0

    @classmethod
    def concat(cls, *sets: "Set[T]") -> "Set[T]":
        """Concatenate multiple Set instances into a single combined Set."""
        if not sets:
            raise ValueError("Need at least one Set to concatenate.")

        first = sets[0]
        element_cls = first.dtype
        if element_cls is None:
            raise TypeError("Cannot concatenate Sets with unspecified element type.")

        batches = [s.to_arrow() for s in sets]
        table = pa.Table.from_batches(batches)
        combined_batch = table.combine_chunks().to_batches()[0]
        return cls.from_arrow(element_cls, combined_batch)

    def to_arrow(self) -> pa.RecordBatch:
        """Convert underlying storage into an immutable Arrow RecordBatch."""
        if hasattr(self._storage, "to_record_batch"):
            return self._storage.to_record_batch()
        raise NotImplementedError("Storage does not support to_record_batch.")

    @classmethod
    def from_arrow(cls, element_cls: Type[T], batch: pa.RecordBatch) -> "Set[T]":
        """Construct a Set[T] wrapping an Arrow RecordBatch."""
        set_schema = element_cls.schema().to_set_schema(capacity=len(batch))
        storage = ArrayStorage.from_record_batch(batch, schema=set_schema)
        return cls[element_cls](capacity=len(batch), _storage_view=storage)

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity.value

    def convert_to_entry(self) -> Entry:
        if self.dtype is None:
            raise TypeError("Cannot convert unsubscripted Set to schema entry.")
        return SchemaSetEntry(
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

    def __setitem__(self, key: Union[int, str], value: Any) -> None:
        match key:
            case int(idx):
                if not isinstance(value, Struct):
                    raise ValueError(f"Expected value to be of subclass Struct, got {type(value).__name__}")

                subview_struct = self[idx]
                for field_path, entry in value.schema().tree_iter():

                    target_entry = _lookup_entry(self.schema(), field_path)
                    assert target_entry is not None

                    field_data = value._storage.read(field_path, entry)
                    subview_struct._storage.write(field_path, target_entry, field_data)

            case str(field_name):
                fields = self.schema().fields
                assert field_name in fields
                
                schema_entry = fields[field_name]
                if isinstance(schema_entry, ArrayEntryBase):
                    self._storage.write((field_name,), schema_entry, value)
                else:
                    raise NotImplementedError(
                        f"Writing to Set field '{field_name}' with schema entry type '{type(schema_entry).__name__}' is not supported yet."
                    )

            case _:
                raise TypeError(f"Invalid Set key type '{type(key).__name__}'. Expected int or str.")

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: str) -> Union[NDArray, RaggedArrayView, "Set[Any]"]: ...

    def __getitem__(
        self, key: Union[int, slice, str]
    ) -> Union[T, Self, NDArray, RaggedArrayView, "Set[Any]"]:
        match key:
            case int(idx):
                assert (isinstance(self.dtype, type) and issubclass(self.dtype, Struct))

                new_offset = self._storage.compute_index_offset(idx)
                subview = ArrayStorageView(
                    self.dtype.schema(),
                    parent=self._storage.parent or self._storage,
                    path=self._storage.path,
                    offset=new_offset,
                )

                return cast(T, self.dtype(_storage_view=subview))

            case slice() as sl:
                assert self.dtype is not None

                start, stop = self._normalize_slice(sl)
                new_size = stop - start

                new_offset = self._storage.compute_slice_offset(start, stop)
                subview = ArrayStorageView(
                    self.schema(),
                    parent=self._storage.parent or self._storage,
                    path=self._storage.path,
                    offset=new_offset,
                )
                return type(self)(capacity=new_size, _storage_view=subview)

            case str(field_name):
                fields = self.schema().fields
                if field_name not in fields:
                    raise ValueError(f"Field '{field_name}' not found in Set schema.")

                entry = fields[field_name]
                if isinstance(entry, ArrayEntryBase):
                    return self._storage.read((field_name,), entry)
                elif isinstance(entry, SchemaEntry):
                    assert entry.schema.dtype is not None
                    
                    new_path = (*self._storage.path, field_name)
                    sliced_view = ArrayStorageView(
                        entry.schema,
                        parent=(self._storage.parent or self._storage),
                        path=new_path,
                        offset=self._storage.offset,
                    )
                    sub_cls: Any = entry.schema.dtype
                    return Set[sub_cls](
                        capacity=self.capacity,
                        _storage_view=sliced_view,
                    )
                else:
                    raise NotImplementedError(
                        f"Reading Set field '{field_name}' with schema entry type '{type(entry).__name__}' is not supported yet."
                    )

            case _:
                raise TypeError(f"Invalid Set index type '{type(key).__name__}'. Expected int, slice, or str.")
