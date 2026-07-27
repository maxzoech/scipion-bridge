from __future__ import annotations
from typing import (
    Type,
    Generic,
    TypeVar,
    Dict,
    Union,
    Any,
    ForwardRef,
    Self,
    Tuple,
    Optional,
)

from ._type_checks import is_struct_type
from .entries import (
    Entry,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaSetEntry,
    _StructEntry,
    SchemaConvertible,
)
from .schema import Schema
from .struct import Struct
import zarr
from zarr.storage import MemoryStore

import numpy as np


def _convert_array_entry(entry: _ArrayEntry) -> Entry:
    if entry.is_static:
        assert entry.min_shape == entry.max_shape
        assert entry.min_shape is not None

        return _ArraySetEntry(
            dtype=entry.dtype, storage=entry.storage, shape=entry.min_shape
        )
    else:
        return _RaggedArraySetEntry(
            dtype=entry.dtype,
            storage=entry.storage,
            min_shape=entry.min_shape,
            max_shape=entry.max_shape,
            preferred_shape=entry.preferred_shape,
        )


def generate_set_schema(cls: Type):
    """Build a set-context schema from a Struct type.

    Converts ``_ArrayEntry`` fields to their set equivalents
    (``_ArraySetEntry`` / ``_RaggedArraySetEntry``) and recursively
    handles nested structs.
    """
    if not is_struct_type(cls):
        raise TypeError("Element of a set has to be of type Struct.")

    fields: Dict[str, Entry] = {}
    for k, v in cls.schema().fields.items():
        if isinstance(v, _StructEntry):
            set_schema = generate_set_schema(v.struct_cls)
            fields[k] = _StructEntry(struct_cls=v.struct_cls, schema=set_schema)
        elif isinstance(v, _ArrayEntry):
            fields[k] = _convert_array_entry(entry=v)
        elif isinstance(v, _SchemaSetEntry):
            fields[k] = v
        else:
            raise NotImplementedError(f"Unknown entry in schema: {v}")

    return Schema(fields)


T = TypeVar("T", bound=Struct)


class Set(Generic[T], SchemaConvertible):

    __runtime_args__ = None
    _generic_cache: Dict = {}

    def configure_array_storage(self) -> zarr.Group:
        return super().configure_array_storage(shape_prefix=(self._capacity,))

    def __init__(self, capacity: int):
        self._capacity = capacity

        super().__init__()

    @classmethod
    def to_schema_entry(cls) -> _SchemaSetEntry:
        return _SchemaSetEntry(
            schema=cls.schema(),
            capacity=cls.capacity(),
            item_type=cls.item_type()
        )

    @classmethod
    def _validate_as_field(cls, key_path: str) -> dict:
        from .schema import _validate_struct_datatypes

        wrapped_type = cls.item_type()
        is_serializable = _validate_struct_datatypes(wrapped_type, root=key_path)
        is_serializable[key_path] = all(is_serializable.values())
        return is_serializable

    @classmethod
    def __class_getitem__(cls, params):
        type_args = params if isinstance(params, tuple) else (params,)

        # Use Generic[T] behavior for TypeVars for type checkers
        if any(isinstance(t, TypeVar) for t in type_args):
            return super().__class_getitem__((params[0],))

        # TODO: Correctly handle forward-declared references
        if any(isinstance(t, (str, ForwardRef)) for t in type_args):
            return super().__class_getitem__((params[0],))

        cache_key = (cls, params)
        if cache_key in Set._generic_cache:
            return Set._generic_cache[cache_key]

        param_names = ",".join(getattr(t, "__name__", str(t)) for t in type_args)
        new_cls_name = f"{cls.__name__}[{param_names}]"

        new_cls = type(
            new_cls_name,
            (cls,),
            {
                "__module__": cls.__module__,
                "__runtime_args__": type_args,
                "__origin__": cls,
                "__args__": type_args,
            },
        )

        Set._generic_cache[cache_key] = new_cls
        return new_cls

    @classmethod
    def item_type(cls) -> Type:
        if not cls.__runtime_args__:
            raise TypeError(
                f"You must subscript {cls.__name__} (e.g., Set[CTF]) before calling schema()"
            )

        return cls.__runtime_args__[0]

    @classmethod
    def capacity(cls) -> Optional[int]:
        if not cls.__runtime_args__:
            raise TypeError(
                f"You must subscript {cls.__name__} (e.g., Set[CTF]) before calling schema()"
            )

        return cls.__runtime_args__[1] if len(cls.__runtime_args__) > 1 else None

    @classmethod
    def schema(cls):
        if not hasattr(cls, "_cached_schema"):
            item_type = cls.item_type()
            cls._cached_schema = generate_set_schema(item_type)
        return cls._cached_schema
    
    def _compute_slice_bounds(self, index: slice):
        start = index.start if index.start is not None else 0
        stop = index.stop if index.stop is not None else self._capacity

        if start < 0:
            start = self._capacity + start

        if stop < 0:
            stop = self._capacity + stop

        if index.step is not None and index.step != 1:
            raise NotImplementedError("Slicing with stride is not supported yet")
        assert stop > start

        return start, stop

    def _get_el(self, index: int) -> T:
        cls: Type[T] = type(self).item_type()
        new_el: T = cls()

        for path, _ in self.schema().tree_iter(root="root"):
            new_el._zarr_group[path] = self._zarr_group[path][index]

        return new_el

    def _get_slice(self, index: slice) -> Self:
        start, stop = self._compute_slice_bounds(index)

        cls = type(self)
        new_set = cls(capacity=stop - start)

        for path, _ in self.schema().tree_iter(root="root"):
            new_set._zarr_group[path][:] = self._zarr_group[path][start:stop]

        return new_set

    def _set_el(self, index: int, value: T) -> None:
        if not isinstance(value, self.item_type()):
            provided = (
                f"Set of '{type(value)}'"
                if isinstance(value, Set)
                else f"'{type(value).__name__}'"
            )
            raise TypeError(
                f"Cannot assign {provided} to element of Set of '{self.item_type().__name__}'"
            )

        for k in self.schema().fields:
            field_val = getattr(value, k)
            self._zarr_group[f"root.{k}"][index] = np.array(field_val)

    def _set_slice(self, index: slice, value: Set[T]) -> None:
        start, stop = self._compute_slice_bounds(index)
        if not (isinstance(value, Set) and value.item_type() == self.item_type()):
            provided = (
                f"Set of '{value.item_type()}'"
                if isinstance(value, Set)
                else f"'{type(value).__name__}'"
            )
            raise TypeError(
                f"Cannot assign {provided} to a slice of Set of '{self.item_type()}'"
            )

        slice_length = stop - start
        if slice_length != len(value):
            raise ValueError(
                f"Cannot assign a Set of capacity {len(value)} to a slice of length {slice_length}"
            )

        for path, _ in self.schema().tree_iter(root="root"):
            self._zarr_group[path][start:stop] = value._zarr_group[path][:]

    def __len__(self):
        return self._capacity

    def __iter__(self):
        for i in range(self._capacity):
            yield self[i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._get_slice(key)

        try:
            idx = int(key)

            if idx < 0:
                idx = self._capacity + idx
            if idx < 0 or idx >= self._capacity:
                raise IndexError(
                    f"Index {key} out of range for Set of capacity {self._capacity}"
                )

            return self._get_el(idx)

        except (TypeError, ValueError):
            raise TypeError(f"Indexing with {type(key).__name__} is not supported.")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            return self._set_slice(key, value)

        try:
            idx = int(key)

            if idx < 0:
                idx = self._capacity + idx
            if idx < 0 or idx >= self._capacity:
                raise IndexError(
                    f"Index {key} out of range for Set of capacity {self._capacity}"
                )

            self._set_el(idx, value)

        except (TypeError, ValueError):
            raise TypeError(f"Indexing with {type(key).__name__} is not supported.")
