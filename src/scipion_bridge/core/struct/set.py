from typing import Type, Generic, TypeVar, Dict, Union, Any, ForwardRef

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
import zarr
from zarr.storage import MemoryStore

import numpy as np

def _convert_array_entry(entry: _ArrayEntry) -> Entry:
    if entry.is_static:
        assert entry.min_shape == entry.max_shape
        assert entry.min_shape is not None

        return _ArraySetEntry(
            dtype=entry.dtype,
            storage=entry.storage,
            shape=entry.min_shape
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


T = TypeVar("T")



class Set(Generic[T], SchemaConvertible):

    __runtime_args__ = None
    _generic_cache: Dict = {}

    def configure_array_storage(self, capacity: int) -> zarr.Group:

        store = MemoryStore()
        root = zarr.group(store=store)

        for name, entry in self.schema().iter_leaves():
            # Leaf entries: _ArrayEntry, _ArraySetEntry, _RaggedArraySetEntry
            if isinstance(entry, _ArrayEntry):
                assert entry.max_shape == entry.min_shape

                root.create_array(
                    name,
                    shape=entry.max_shape,
                    dtype=entry.dtype,
                )
            elif isinstance(entry, _ArraySetEntry) and entry.is_static == True:
                root.create_array(
                    name,
                    shape=[capacity, *entry.max_shape],
                    dtype=entry.dtype,
                )
            else:
                raise NotImplementedError(f"Cannot create zarr group for entry {entry} at {name}")

        return root

    def __init__(self, capacity: int):
        super().__init__(capacity)

    @classmethod
    def to_schema_entry(cls) -> _SchemaSetEntry:
        return _SchemaSetEntry(schema=cls.schema())

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
            return super().__class_getitem__(params)

        # TODO: Correctly handle forward-declared references
        if any(isinstance(t, (str, ForwardRef)) for t in type_args):
            return super().__class_getitem__(params)

        cache_key = (cls, params)
        if cache_key in Set._generic_cache:
            return Set._generic_cache[cache_key]

        param_names = ",".join(getattr(t, '__name__', str(t)) for t in type_args)
        new_cls_name = f"{cls.__name__}[{param_names}]"

        new_cls = type(new_cls_name, (cls,), {
            "__module__": cls.__module__,
            "__runtime_args__": type_args,
            "__origin__": cls,
            "__args__": type_args,
        })

        Set._generic_cache[cache_key] = new_cls
        return new_cls

    @classmethod
    def item_type(cls) -> Type:
        if not cls.__runtime_args__:
            raise TypeError(f"You must subscript {cls.__name__} (e.g., Set[CTF]) before calling schema()")

        return cls.__runtime_args__[0]

    @classmethod
    def schema(cls):
        if not hasattr(cls, '_cached_schema'):
            item_type = cls.item_type()
            cls._cached_schema = generate_set_schema(item_type)
        return cls._cached_schema


    def _get_element(self, index: int, prefix: str = "root"):
        data_dict = {}
        for k, entry in self.schema().fields.items():
            if isinstance(entry, _ArraySetEntry):
                storage_key = f"{prefix}.{k}"
                data_dict[k] = self._zarr_group[storage_key][index]
            else:
                raise NotImplementedError(f"Indexing into {entry} is not supported yet")
        return self.item_type()(**data_dict)

    def _set_element(self, index: int, value: T, prefix: str = "root"):
        for k, entry in self.schema().fields.items():
            if isinstance(entry, _ArraySetEntry):
                storage_key = f"{prefix}.{k}"
                self._zarr_group[storage_key][index] = np.array(value[k]) # type: ignore
            else:
                raise NotImplementedError(f"Indexing into {entry} is not supported yet")

    def __getitem__(self, key):
        try:
            return self._get_element(int(key))
        except ValueError:
            pass

    def __setitem__(self, key, value):
        try:
            self._set_element(int(key), value)
            return
        except ValueError:
            pass
