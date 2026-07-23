

from functools import cache

from typing import Type, Generic, TypeVar, get_args, Dict
from .struct import Struct, _is_struct_type
from .schema import Schema, Entry, _ArraySetEntry, _RaggedArraySetEntry, _ArrayEntry


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

    


def generate_set_schema(cls: Type[Struct]):
    
    if not issubclass(cls, Struct):
        raise TypeError("Element of a set has to be of type Struct.")

    fields: Dict[str, Entry] = {}
    for k, v in cls.schema().fields.items():
        if _is_struct_type(v):
            raise NotImplementedError("Sub-structs are not yet supported")
        elif isinstance(v, _ArrayEntry):
            fields[k] = _convert_array_entry(entry=v)
        else:
            raise NotImplementedError(f"Unkown entry in schema: {v}")

    return Schema(fields)

T = TypeVar("T", bound=Struct)

class Set(Generic[T]):

    __runtime_args__ = None
    _generic_cache: Dict = {}

    @classmethod
    @cache # Caches the class creation so Set[CTF] is only built once
    def __class_getitem__(cls, params):
        cache_key = (cls, params)
        if cache_key in Set._generic_cache:
            return Set._generic_cache[cache_key]

        type_args = params if isinstance(params, tuple) else (params,)
        param_names = ",".join(getattr(t, '__name__', str(t)) for t in type_args)
        new_cls_name = f"{cls.__name__}[{param_names}]"

        new_cls = type(new_cls_name, (cls,), {
            "__runtime_args__": type_args,
            
            # Optional: Duck-type as a standard GenericAlias so standard library
            # tools like `typing.get_args(Set[CTF])` still work beautifully at runtime.
            "__origin__": cls,
            "__args__": type_args,
        })

        Set._generic_cache[cache_key] = new_cls
        return new_cls

    @classmethod
    @cache
    def schema(cls):
        if not cls.__runtime_args__:
            raise TypeError(f"You must subscript {cls.__name__} (e.g., Set[CTF]) before calling schema()")
        
        item_type = cls.__runtime_args__[0]

        return generate_set_schema(item_type)