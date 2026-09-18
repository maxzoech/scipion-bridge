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
import awkward as ak

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
) -> TypeGuard[Union[Sequence[bool], NDArray[np.bool_]]]:
    if isinstance(key, np.ndarray):
        return key.ndim == 1 and (
            key.dtype == bool or np.issubdtype(key.dtype, np.bool_)
        )
    if isinstance(key, (list, tuple)):
        return len(key) > 0 and all(
            type(x) is bool or isinstance(x, (bool, np.bool_)) for x in key
        )
    return False


def _is_int_sequence(
    key: Any,
) -> TypeGuard[Union[Sequence[int], NDArray[np.integer]]]:
    if isinstance(key, np.ndarray):
        return key.ndim == 1 and np.issubdtype(key.dtype, np.integer)
    if isinstance(key, (list, tuple)):
        return all(
            isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
            for x in key
        )
    return False


class Set(Marker[T], SchemaConvertible):
    """Sequence container for Struct instances backed by Apache Arrow columnar storage."""

    _bridge_schema: Schema

    def __init__(
        self,
        items: Sequence[T] = (),
        capacity: Optional[Union[int, Arg]] = None,
        storage: Optional[_BaseStorage] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        items = list(items)
        self._storage = storage if storage is not None else StagingEngine()

        resolved_cap = capacity.value if isinstance(capacity, Arg) else capacity

        match (items, resolved_cap):
            case ([], None):
                # Mode 1: Empty set without defined capacity (dynamic)
                self._capacity = None

            case ([], int(cap)):
                # Mode 2: Empty set with explicit capacity
                self._capacity = cap

            case ([_, *_], None):
                # Mode 3: Set from elements (capacity set to number of elements)
                self._capacity = len(items)

            case ([_, *_], int(cap)):
                # Mode 4: Set from elements with extra capacity
                if cap < len(items):
                    raise ValueError(
                        f"Capacity {cap} is less than the number of items {len(items)}."
                    )
                self._capacity = cap

            case _:
                raise TypeError(
                    f"Invalid arguments for Set: items={type(items).__name__}, capacity={type(capacity).__name__}"
                )

        if items:
            self._populate_from_items(items)

    def _populate_from_items(self, items: Sequence[T]) -> None:
        """Write struct items into columnar storage."""
        target_schema = self.schema().to_set_schema(capacity=self._capacity)

        for path, entry in target_schema.tree_iter():
            col_chunks = [
                item.storage.read(item.storage.root.extend(path), entry)
                for item in items
            ]
            
            target_path = self._storage.root.extend(path)
            self._storage.write(target_path, entry, col_chunks)

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            return self

        if isinstance(instance, Struct):
            if self.name is None:
                raise AttributeError("Descriptor name is not set.")

            _schema = self.schema()
            assert isinstance(_schema.dtype, type) and issubclass(_schema.dtype, Struct)

            subview = instance.storage.append(self.name)
            st_len = subview.get_length()
            active_capacity = st_len if st_len is not None else self.capacity
            return type(self)(storage=subview, capacity=active_capacity)

        raise NotImplementedError

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(
                f"Expected Struct instance, got '{type(instance).__name__}'."
            )

        if not isinstance(value, Set):
            raise TypeError(
                f"Expected Struct instance, got '{type(instance).__name__}'."
            )

        if self.name is None:
            raise AttributeError("Set descriptor name is not set.")

        if self.name not in instance.schema().fields:
            raise AttributeError

        field_entry = instance.schema().fields[self.name]
        if not isinstance(field_entry, (SchemaEntry, SchemaSetEntry)):
            raise AttributeError

        for path, entry in field_entry.schema.tree_iter():

            source_path = value._storage.root.extend(path)
            data = value._storage.read(source_path, entry)

            target_path = instance.storage.root.append(self.name).extend(path)
            instance.storage.write(target_path, entry, data)

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
    def _active_span(self) -> Optional[int]:
        """Return the active indexable span: fixed capacity if set, otherwise storage data length."""
        if self._capacity is not None:
            return self._capacity

        return self._storage.get_length()

    def __bool__(self) -> bool:
        """A Set is truthy if and only if it contains populated data."""
        return len(self) > 0

    def __len__(self) -> int:
        """The actual length of the set queried directly from data."""
        st_len = self._storage.get_length()
        return st_len if st_len is not None else 0

    def __length_hint__(self) -> int:
        """The capacity of the set if fixed; returns 0 for unsized sets (PEP 424 fast estimate)."""
        return self._capacity if self._capacity is not None else 0

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

    @property
    def storage(self) -> _BaseStorage:
        return self._storage

    def convert_to_entry(self) -> Entry:
        if self.dtype is None:
            raise TypeError("Cannot convert unsubscripted Set to schema entry.")

        return SchemaSetEntry(
            schema=self.schema().to_set_schema(capacity=self.capacity),
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

    def __setitem__(
        self,
        key: Union[
            int,
            slice,
            str,
            Sequence[bool],
            Sequence[int],
            NDArray[np.bool_],
            NDArray[np.integer],
        ],
        value: Any,
    ) -> None:
        match key:
            case int(index):
                if not isinstance(value, Struct):
                    raise ValueError(
                        f"Expected value to be of subclass Struct, got {type(value).__name__}"
                    )

                span = self._active_span
                entry = self.entry
                assert isinstance(entry, SchemaEntry)
                for path, target_entry, source_entry in entry.schema.tree_iter(
                    value.schema()
                ):
                    source_path = value.storage.root.extend(path)
                    data = value.storage.read(source_path, source_entry)

                    target_path = self._storage.root.narrow_index(
                        index, length=span
                    ).extend(path)
                    self._storage.write(target_path, target_entry, data)

            case slice() as index:
                if not isinstance(value, Set):
                    raise ValueError(
                        f"Expected value to be of subclass Set, got {type(value).__name__}"
                    )

                span = self._active_span
                entry = self.entry
                assert isinstance(entry, SchemaEntry)
                for path, target_entry, source_entry in entry.schema.tree_iter(
                    value.schema()
                ):
                    source_path = value._storage.root.extend(path)
                    data = value._storage.read(source_path, source_entry)

                    target_path = self._storage.root.narrow_slice(
                        index, length=span
                    ).extend(path)
                    self._storage.write(target_path, target_entry, data)

            case _ if _is_bool_sequence(key):
                if not isinstance(value, Set):
                    raise ValueError(
                        f"Expected value to be of subclass Set, got {type(value).__name__}"
                    )

                span = self._active_span
                entry = self.entry
                assert isinstance(entry, SchemaEntry)
                for path, target_entry, source_entry in entry.schema.tree_iter(
                    value.schema()
                ):
                    source_path = value._storage.root.extend(path)
                    data = value._storage.read(source_path, source_entry)

                    target_path = self._storage.root.narrow_mask(
                        key, length=span
                    ).extend(path)
                    self._storage.write(target_path, target_entry, data)

            case _ if _is_int_sequence(key):
                if not isinstance(value, Set):
                    raise ValueError(
                        f"Expected value to be of subclass Set, got {type(value).__name__}"
                    )

                span = self._active_span
                entry = self.entry
                assert isinstance(entry, SchemaEntry)
                for path, target_entry, source_entry in entry.schema.tree_iter(
                    value.schema()
                ):
                    source_path = value._storage.root.extend(path)
                    data = value._storage.read(source_path, source_entry)

                    target_path = self._storage.root.narrow_indices(
                        key, length=span
                    ).extend(path)
                    self._storage.write(target_path, target_entry, data)

            case str(field_name):
                fields = self.schema().fields
                if field_name not in fields:
                    raise AttributeError

                schema_entry = fields[field_name]
                if isinstance(schema_entry, ArrayEntryBase):
                    if self.capacity is not None and len(value) > self.capacity:
                        raise ValueError(
                            f"Writing data of length {len(value)} exceeds capacity {self.capacity} for Set field '{field_name}'."
                        )

                    path = self._storage.root.append(field_name)
                    self._storage.write(path, schema_entry, value)

                elif isinstance(schema_entry, SchemaEntry):
                    if not isinstance(value, Set):
                        raise ValueError

                    assert isinstance(value, Set)

                    for (
                        path,
                        target_entry,
                        source_entry,
                    ) in schema_entry.schema.tree_iter(value.schema()):
                        source_path = value._storage.root.extend(path)
                        data = value._storage.read(source_path, source_entry)

                        target_path = self._storage.root.append(field_name).extend(path)
                        self._storage.write(target_path, target_entry, data)

                else:
                    raise NotImplementedError(
                        f"Writing to Set field '{field_name}' with schema entry type '{type(schema_entry).__name__}' is not supported yet."
                    )

            case _:
                raise TypeError(
                    f"Invalid Set key type '{type(key).__name__}'. Expected int, slice, str, or integer/boolean sequence."
                )

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: Union[Sequence[bool], NDArray[np.bool_]]) -> Self: ...

    @overload
    def __getitem__(self, key: Union[Sequence[int], NDArray[np.integer]]) -> Self: ...

    @overload
    def __getitem__(self, key: str) -> Union[NDArray, "Set[Any]"]: ...

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
        ],
    ) -> Union[T, Self, NDArray, "Set[Any]"]:
        match key:
            case bool():
                raise TypeError(
                    "Cannot index Set with a single boolean. Use an integer, slice, str, or boolean mask."
                )

            case _ if _is_bool_sequence(key):
                if self.dtype is None:
                    raise TypeError("Cannot index Set with unspecified element type.")

                span = self._active_span
                if span is None:
                    raise IndexError("Cannot index into a Set with dynamic capacity.")

                _dt = self.dtype
                assert isinstance(_dt, type) and issubclass(_dt, Struct)

                view = self._storage.narrow_mask(key, length=span)
                return Set[_dt](capacity=int(np.sum(key)), storage=view)

            case _ if _is_int_sequence(key):
                if self.dtype is None:
                    raise TypeError("Cannot index Set with unspecified element type.")

                span = self._active_span
                if span is None:
                    raise IndexError("Cannot index into a Set with dynamic capacity.")

                _dt = self.dtype
                assert isinstance(_dt, type) and issubclass(_dt, Struct)

                view = self._storage.narrow_indices(key, length=span)
                return Set[_dt](capacity=len(key), storage=view)

            case int(index):
                span = self._active_span
                if span is None:
                    raise IndexError("Cannot index into a Set with dynamic capacity.")

                _dt = self.dtype
                assert isinstance(_dt, type) and issubclass(_dt, Struct)

                view = self._storage.narrow_index(index, length=span)
                return cast(T, _dt(view))

            case slice() as index:
                if (index.step or 1) != 1:
                    raise ValueError("Only step=1 is supported.")

                _dt = self.dtype
                assert isinstance(_dt, type) and issubclass(_dt, Struct)

                span = self._active_span
                new_cap = (
                    len(range(*index.indices(span)))
                    if span is not None
                    else None
                )
                view = self._storage.narrow_slice(index, length=span)
                return Set[_dt](capacity=new_cap, storage=view)

            case str(field_name):
                fields = self.schema().fields
                if field_name not in fields:
                    raise ValueError(f"Field '{field_name}' not found in Set schema.")

                entry = fields[field_name]

                if isinstance(entry, ArrayEntryBase):
                    path = self._storage.root.append(field_name)
                    return self._storage.read(path, entry)

                elif isinstance(entry, SchemaEntry):
                    _dt = entry.schema.dtype
                    assert isinstance(_dt, type) and issubclass(_dt, Struct)

                    return Set[_dt](
                        capacity=self.capacity,
                        storage=self._storage.append(field_name),
                    )

                else:
                    raise NotImplementedError(
                        f"Reading Set field '{field_name}' with schema entry type '{type(entry).__name__}' is not supported yet."
                    )

            case _:
                raise TypeError(
                    f"Invalid Set index type '{type(key).__name__}'. Expected int, slice, str, or integer/boolean sequence."
                )


def concat(sets: Sequence[Set[T]]) -> Set[T]:
    """Concatenate multiple Sets with identical schemas along axis 0.

    Args:
        sets: A non-empty sequence of Set instances to concatenate.

    Returns:
        A new Set containing the concatenated data.
    """
    if not sets:
        raise ValueError("concat requires at least one Set to concatenate.")

    first_set, *other_sets = sets
    for s in sets:
        assert isinstance(s, Set)

    first_schema, *other_schemas = [s.schema() for s in sets]
    for other in other_schemas:
        if other != first_schema:
            raise ValueError(
                f"Cannot concatenate sets with mismatched schemas: {other} != {first_schema}",
            )

    if not other_sets:
        return first_set

    if all(s.capacity is not None for s in sets):
        total_capacity: Optional[int] = sum(cast(int, s.capacity) for s in sets)
    else:
        total_capacity = None

    target_entry = SchemaSetEntry(
        schema=first_set.schema().to_set_schema(capacity=total_capacity),
        capacity=total_capacity,
    )
    new_storage = first_set.storage.concat(
        [s.storage for s in other_sets],
        entry=target_entry,
    )

    assert first_set.dtype is not None
    return Set[first_set.dtype](
        capacity=total_capacity,
        storage=new_storage,
    )
