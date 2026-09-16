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
from typing_extensions import Self, TypeGuard
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.compute as pc

from .struct import Struct, Arg, Trait
from .schema import (
    Entry,
    SchemaConvertible,
    Schema,
    ArrayEntryBase,
    SchemaEntry,
    SchemaSetEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
)
from .key_path import KeyPath
from .storage import _BaseStorage, StagingEngine, StorageView
from .utils.arrow_utils import RaggedArrayView
from ..utils.marker import Marker

T = TypeVar("T", bound=Struct)


def _is_bool_sequence(
    key: Any,
) -> TypeGuard[Union[Sequence[bool], NDArray[np.bool_], pa.BooleanArray]]:
    if isinstance(key, np.ndarray):
        return key.ndim == 1 and (key.dtype == bool or np.issubdtype(key.dtype, np.bool_))
    if isinstance(key, (pa.Array, pa.ChunkedArray)) and pa.types.is_boolean(key.type):
        return True
    if isinstance(key, (list, tuple)):
        return len(key) > 0 and all(type(x) is bool or isinstance(x, (bool, np.bool_)) for x in key)
    return False


def _is_int_sequence(
    key: Any,
) -> TypeGuard[Union[Sequence[int], NDArray[np.integer], pa.Array]]:
    if isinstance(key, np.ndarray):
        return key.ndim == 1 and np.issubdtype(key.dtype, np.integer)
    if isinstance(key, (pa.Array, pa.ChunkedArray)) and pa.types.is_integer(key.type):
        return True
    if isinstance(key, (list, tuple)):
        return all(
            isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
            for x in key
        )
    return False


class Set(Marker[T], SchemaConvertible):
    """Sequence container for Struct instances backed by Apache Arrow columnar storage."""

    _bridge_schema: Schema

    # @overload
    # def __init__(self, capacity: Sequence[T], **kwargs: Any) -> None: ...

    # @overload
    # def __init__(self, capacity: Optional[Union[int, Arg]] = None, **kwargs: Any) -> None: ...

    def __init__(
        self,
        capacity: Optional[int] = None,
        storage: _BaseStorage = StagingEngine(),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._capacity = capacity
        self._storage = storage

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)

    # def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
    #     if instance is None and owner is not None:
    #         if self.dtype is None:
    #             raise TypeError("Cannot create BoundSetView without a defined dtype.")
    #         return BoundSetView(
    #             element_cls=self.dtype,
    #             capacity_spec=self._capacity,
    #             owner_cls=owner,
    #         )
    #     if isinstance(instance, Struct):
    #         if self.name is None:
    #             raise AttributeError("Descriptor name is not set.")

    #         _schema = self.schema()
    #         assert isinstance(_schema.dtype, type) and issubclass(_schema.dtype, Struct)

    #         new_path = (*instance._storage.path, self.name)
    #         new_offset = instance._storage.offset.descend()
    #         subview = ArrayStorageView(
    #             self.schema(),
    #             parent=instance._storage.parent or instance._storage,
    #             path=new_path,
    #             offset=new_offset,
    #         )

    #         elem_cls: Any = _schema.dtype
    #         assigned_cap = instance._storage._field_capacities.get(
    #             self.name, self.capacity
    #         )
    #         return cast(Any, Set)[elem_cls](capacity=assigned_cap, _storage_view=subview)
    #     return self

    # def __set__(self, instance: Any, value: Any) -> None:
    #     if not isinstance(instance, Struct):
    #         raise TypeError(f"Expected Struct instance, got '{type(instance).__name__}'.")
    #     if self.name is None:
    #         raise AttributeError("Set descriptor name is not set.")

    #     if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Set)):
    #         elem_cls = self.dtype
    #         if elem_cls is None:
    #             raise TypeError(
    #                 f"Cannot initialize Set field '{self.name}' with a sequence without a defined element type."
    #             )
    #         value = cast(Any, Set)[elem_cls](value)

    #     if not isinstance(value, Set):
    #         raise TypeError(
    #             f"Expected Set or Sequence of Structs for field '{self.name}', got '{type(value).__name__}'."
    #         )

    #     if self.dtype is not None and value.dtype is not None:
    #         if not (isinstance(value.dtype, type) and issubclass(value.dtype, self.dtype)):
    #             raise TypeError(
    #                 f"Cannot assign Set of '{value.dtype.__name__}' to field '{self.name}' "
    #                 f"expecting elements of type '{self.dtype.__name__}'."
    #             )

    #     if self.capacity is not None and value.capacity is not None:
    #         if value.capacity > self.capacity:
    #             raise ValueError(
    #                 f"Assigned Set length {value.capacity} exceeds field '{self.name}' capacity {self.capacity}."
    #             )

    #     if hasattr(instance, "_storage"):
    #         instance._storage._field_capacities[self.name] = value.capacity

    #     for key, entry in value.schema().tree_iter():
    #         target_entry = instance.schema().lookup_array((self.name, *key)) or entry
    #         data = value._storage.read(key, entry)
    #         instance._storage.write((self.name, *key), entry=target_entry, data=data)

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
    def item_type(cls) -> Optional[Type[Any]]:
        """Return the element Struct type of this Set class."""
        return cls._dtype

    @classmethod
    def schema(cls) -> Schema:
        return cls._bridge_schema

    def __len__(self) -> int:
        raise NotImplementedError
    
    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

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

    def __setitem__(self, key: Union[int, str], value: Any) -> None:

        match key:
            case int(index):
                if not isinstance(value, Struct):
                    raise ValueError(f"Expected value to be of subclass Struct, got {type(value).__name__}")

                for path, target_entry, source_entry in self.schema().tree_iter(value.schema()):
                    source_path = value._storage.root.extend(path)
                    data = value._storage.read(source_path, source_entry)

                    target_path = value._storage.root.narrow_index(index).extend(path)
                    value._storage.write(target_path, target_entry, data)
                    

            case str(field_name):
                fields = self.schema().fields
                assert field_name in fields
                
                schema_entry = fields[field_name]
                if isinstance(schema_entry, ArrayEntryBase):
                    raise NotImplementedError(
                        f"Writing to Set field '{field_name}' with schema entry type '{type(schema_entry).__name__}' is not supported yet."
                    )
                
                else:
                    raise NotImplementedError(
                        f"Writing to Set field '{field_name}' with schema entry type '{type(schema_entry).__name__}' is not supported yet."
                    )

            case _:
                raise TypeError(f"Invalid Set key type '{type(key).__name__}'. Expected int or str.")

    # @overload
    # def __getitem__(self, key: int) -> T: ...

    # @overload
    # def __getitem__(self, key: slice) -> Self: ...

    # @overload
    # def __getitem__(
    #     self, key: Union[Sequence[bool], NDArray[np.bool_], pa.BooleanArray]
    # ) -> Self: ...

    # @overload
    # def __getitem__(
    #     self, key: Union[Sequence[int], NDArray[np.integer], pa.IntegerArray]
    # ) -> Self: ...

    # @overload
    # def __getitem__(self, key: str) -> Union[NDArray, RaggedArrayView, "Set[Any]"]: ...

    def __getitem__(
        self,
        key: Union[
            int,
            slice,
            str,
            Sequence[bool],
            Sequence[int],
            NDArray[np.bool_],
            NDArray[np.integer],
            pa.Array,
        ],
    ) -> Union[T, Self, NDArray, RaggedArrayView, "Set[Any]"]:
        match key:
            case bool():
                raise TypeError(
                    "Cannot index Set with a single boolean. Use an integer, slice, str, or boolean mask."
                )

            case _ if _is_bool_sequence(key):
                if self.dtype is None:
                    raise TypeError("Cannot index Set with unspecified element type.")

                raise NotImplementedError("Boolean masking is not supported yet")

            case _ if _is_int_sequence(key):
                if self.dtype is None:
                    raise TypeError("Cannot index Set with unspecified element type.")

                raise NotImplementedError("Indexing with arrays is not supported yet")

            case int(idx):
                assert (isinstance(self.dtype, type) and issubclass(self.dtype, Struct))

                assert False

            case slice() as sl:
                assert self.dtype is not None

                assert False

            case str(field_name):
                fields = self.schema().fields
                if field_name not in fields:
                    raise ValueError(f"Field '{field_name}' not found in Set schema.")

                entry = fields[field_name]
                if isinstance(entry, ArrayEntryBase):
                    raise NotImplementedError("Reading an entry is not supported yet")
                    
                elif isinstance(entry, SchemaSetEntry):
                    assert entry.schema.dtype is not None
                    
                    new_path = (*self._storage.path, field_name)
                    new_offset = self._storage.offset.descend()
                    sliced_view = ArrayStorageView(
                        entry.schema,
                        parent=self._storage.root_storage,
                        path=new_path,
                        offset=new_offset,
                    )
                    sub_cls: Any = entry.schema.dtype
                    return Set[sub_cls](
                        capacity=entry.capacity,
                        _storage_view=sliced_view,
                    )
                
                elif isinstance(entry, SchemaEntry):
                    assert entry.schema.dtype is not None
                    
                    new_path = (*self._storage.path, field_name)
                    sliced_view = ArrayStorageView(
                        entry.schema,
                        parent=self._storage.root_storage,
                        path=new_path,
                        offset=self._storage.offset,
                    )
                    sub_cls = entry.schema.dtype
                    return Set[sub_cls](
                        capacity=self.capacity,
                        _storage_view=sliced_view,
                    )
                else:
                    raise NotImplementedError(
                        f"Reading Set field '{field_name}' with schema entry type '{type(entry).__name__}' is not supported yet."
                    )

            case _:
                raise TypeError(
                    f"Invalid Set index type '{type(key).__name__}'. Expected int, slice, str, or integer/boolean sequence."
                )
