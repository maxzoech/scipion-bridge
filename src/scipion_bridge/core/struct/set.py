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
from .storage import _BaseStorage, StorageView
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
        cap_val: Optional[int] = None

        if capacity is not None and isinstance(capacity, Sequence) and not isinstance(capacity, (str, bytes)):
            items_to_populate = capacity
            derived_cap = len(items_to_populate)
            self._capacity = Arg.new(derived_cap)
            cap_val = derived_cap
        else:
            self._capacity = Arg.new(capacity)  # type: ignore
            cap_val = self._capacity.value

        root_entry = SchemaSetEntry(schema=self.schema(), capacity=cap_val)
        storage = kwargs.get("_storage_view", None)
        if storage is not None and not isinstance(storage, _BaseStorage):
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
            if self.name is None:
                raise AttributeError("Descriptor name is not set.")

            _schema = self.schema()
            assert isinstance(_schema.dtype, type) and issubclass(_schema.dtype, Struct)

            new_path = (*instance._storage.path, self.name)
            new_offset = instance._storage.offset.descend()
            subview = ArrayStorageView(
                self.schema(),
                parent=instance._storage.parent or instance._storage,
                path=new_path,
                offset=new_offset,
            )

            elem_cls: Any = _schema.dtype
            assigned_cap = instance._storage._field_capacities.get(
                self.name, self.capacity
            )
            return cast(Any, Set)[elem_cls](capacity=assigned_cap, _storage_view=subview)
        return self

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(f"Expected Struct instance, got '{type(instance).__name__}'.")
        if self.name is None:
            raise AttributeError("Set descriptor name is not set.")

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Set)):
            elem_cls = self.dtype
            if elem_cls is None:
                raise TypeError(
                    f"Cannot initialize Set field '{self.name}' with a sequence without a defined element type."
                )
            value = cast(Any, Set)[elem_cls](value)

        if not isinstance(value, Set):
            raise TypeError(
                f"Expected Set or Sequence of Structs for field '{self.name}', got '{type(value).__name__}'."
            )

        if self.dtype is not None and value.dtype is not None:
            if not (isinstance(value.dtype, type) and issubclass(value.dtype, self.dtype)):
                raise TypeError(
                    f"Cannot assign Set of '{value.dtype.__name__}' to field '{self.name}' "
                    f"expecting elements of type '{self.dtype.__name__}'."
                )

        if self.capacity is not None and value.capacity is not None:
            if value.capacity > self.capacity:
                raise ValueError(
                    f"Assigned Set length {value.capacity} exceeds field '{self.name}' capacity {self.capacity}."
                )

        if hasattr(instance, "_storage"):
            instance._storage._field_capacities[self.name] = value.capacity

        for key, entry in value.schema().tree_iter():
            target_entry = instance.schema().lookup_array((self.name, *key)) or entry
            data = value._storage.read(key, entry)
            instance._storage.write((self.name, *key), entry=target_entry, data=data)

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
        if self.capacity is not None:
            return self.capacity

        raise TypeError(f"Set with dynamic capacity has no defined length.")

    def __bool__(self) -> bool:
        return True

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
        if table.num_rows == 0:
            combined_batch = batches[0]
        else:
            combined_batch = table.combine_chunks().to_batches()[0]
        return cls.from_arrow(element_cls, combined_batch)

    def to_arrow(self) -> pa.RecordBatch:
        """Convert underlying storage into an immutable Arrow RecordBatch."""
        if hasattr(self._storage, "to_record_batch"):
            return self._storage.to_record_batch()
        raise NotImplementedError("Storage does not support to_record_batch.")

    def freeze(self) -> Self:
        """Freeze underlying storage into immutable Arrow columnar format and return self."""
        self.to_arrow()
        return self

    def __reduce__(self) -> Tuple[Any, ...]:
        """Support pickle and Ray object store serialization via Apache Arrow RecordBatch."""
        if self.dtype is not None:
            return (Set.from_arrow, (self.dtype, self.to_arrow()))
        raise TypeError("Cannot serialize Set with unspecified element type.")

    @classmethod
    def from_arrow(
        cls,
        element_cls_or_batch: Union[Type[T], pa.RecordBatch],
        batch: Optional[pa.RecordBatch] = None,
    ) -> "Set[T]":
        """Construct a Set[T] wrapping an Arrow RecordBatch."""
        actual_batch: pa.RecordBatch
        if batch is None:
            if not isinstance(element_cls_or_batch, pa.RecordBatch):
                raise TypeError("Expected RecordBatch as argument.")
            actual_batch = element_cls_or_batch
            if cls._dtype is None:
                raise TypeError("Cannot determine element type from unsubscripted Set class.")
            element_cls = cast(Type[T], cls._dtype)
        else:
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError("Expected RecordBatch as second argument.")
            actual_batch = batch
            element_cls = cast(Type[T], element_cls_or_batch)

        set_schema = element_cls.schema().to_set_schema(capacity=len(actual_batch))
        root_entry = SchemaSetEntry(schema=set_schema, capacity=len(actual_batch))
        storage = ArrayStorage.from_record_batch(actual_batch, schema=set_schema, root_entry=root_entry)

        if cls._dtype == element_cls:
            target_cls = cls
        else:
            target_cls = cast(Any, Set)[element_cls]
        return target_cls(capacity=len(actual_batch), _storage_view=storage)

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

                    target_entry = self.schema().lookup_array(field_path)
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
    def __getitem__(
        self, key: Union[Sequence[bool], NDArray[np.bool_], pa.BooleanArray]
    ) -> Self: ...

    @overload
    def __getitem__(
        self, key: Union[Sequence[int], NDArray[np.integer], pa.IntegerArray]
    ) -> Self: ...

    @overload
    def __getitem__(self, key: str) -> Union[NDArray, RaggedArrayView, "Set[Any]"]: ...

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

                batch = self.to_arrow()
                pa_mask = pa.array(key, type=pa.bool_())
                if len(pa_mask) != len(batch):
                    raise IndexError(
                        f"Boolean mask length {len(pa_mask)} does not match Set length {len(batch)}."
                    )

                filtered_batch = pc.filter(batch, pa_mask)
                return self.from_arrow(self.dtype, filtered_batch)

            case _ if _is_int_sequence(key):
                if self.dtype is None:
                    raise TypeError("Cannot index Set with unspecified element type.")

                batch = self.to_arrow()
                set_len = len(batch)

                idx_arr = np.asarray(key, dtype=np.int64)
                if len(idx_arr) > 0:
                    norm_arr = np.where(idx_arr < 0, idx_arr + set_len, idx_arr)
                    if np.any((norm_arr < 0) | (norm_arr >= set_len)):
                        raise IndexError(
                            f"Index out of bounds for Set of length {set_len}."
                        )
                    pa_indices = pa.array(norm_arr, type=pa.int64())
                else:
                    pa_indices = pa.array([], type=pa.int64())

                taken_batch = pc.take(batch, pa_indices)
                return self.from_arrow(self.dtype, taken_batch)

            case int(idx):
                assert (isinstance(self.dtype, type) and issubclass(self.dtype, Struct))

                if self.capacity is None:
                    raise IndexError("Cannot index into a Set with dynamic capacity without a prior bound or slice.")

                norm_idx = self.capacity + idx if idx < 0 else idx
                if norm_idx < 0 or norm_idx >= self.capacity:
                    raise IndexError(f"Index {idx} out of range for Set with capacity {self.capacity}.")

                new_offset = self._storage.offset.push_index(norm_idx)
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

                new_offset = self._storage.offset.push_slice(start, stop)
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
