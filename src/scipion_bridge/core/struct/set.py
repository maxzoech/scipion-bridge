from typing import Type, Generic, TypeVar, Dict

from ._type_checks import is_struct_type
from .entries import (
    Entry,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaSetEntry,
    _StructEntry,
)
from .schema import Schema


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

class Set(Generic[T]):

    __runtime_args__ = None
    _generic_cache: Dict = {}

    @classmethod
    def __class_getitem__(cls, params):
        cache_key = (cls, params)
        if cache_key in Set._generic_cache:
            return Set._generic_cache[cache_key]

        type_args = params if isinstance(params, tuple) else (params,)
        param_names = ",".join(getattr(t, '__name__', str(t)) for t in type_args)
        new_cls_name = f"{cls.__name__}[{param_names}]"

        new_cls = type(new_cls_name, (cls,), {
            "__runtime_args__": type_args,

            # Duck-type as a standard GenericAlias so standard library
            # tools like `typing.get_args(Set[CTF])` still work at runtime.
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
