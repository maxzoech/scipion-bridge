import numpy as np
import abc
from dataclasses import dataclass
from enum import Enum
import typing
from typing import Type, Any, Dict, Optional, TypeVar, Generic, Tuple, Set

from ..utils.type_annotation import has_untyped_class_definitions
from ..utils.format import format_list

T = TypeVar("T")


def _is_struct_type(dtype: Any) -> bool:
    from .struct import Struct
    return isinstance(dtype, type) and issubclass(dtype, Struct)


class Entry:
    pass  # Marker Type


class Array(Generic[T]):
    pass


class _ArrayLocation(Enum):
    AUTOMATIC = "auto"
    # HOST_MEMORY = "host_memory"


@dataclass
class _ArrayEntry(Entry):
    dtype: np.dtype
    storage: _ArrayLocation
    min_shape: Optional[Tuple[int]]
    max_shape: Optional[Tuple[int]]
    preferred_shape: Optional[Tuple[int]]

    @property
    def is_static(self) -> bool:
        return self.min_shape is not None and self.min_shape == self.max_shape


@dataclass
class Schema:
    fields: Dict[str, Entry]

    def entries(self) -> Set[str]:
        return set(self.fields.keys())

    @property
    def is_static(self) -> bool:
        for entry in self.fields.values():
            if _is_struct_type(entry):
                if not create_schema(entry).is_static: # type: ignore
                    return False
            elif isinstance(entry, _ArrayEntry):
                if not entry.is_static:
                    return False
            else:
                return False
        return True


    def print_tree(self, typename: Optional[str] = None) -> None:  # pragma: no cover
        """
        Prints a Schema object in a hierarchical tree format.
        """
        print(typename if typename is not None else "/")
        
        def _print_node(node: dict, prefix: str = ""):
            items = list(node.items())
            for i, (key, value) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "

                if _is_struct_type(value):
                    schema = create_schema(value)

                    print(f"{prefix}{connector}{key}")
                    extension = "    " if is_last else "│   "
                    _print_node(schema.fields, prefix + extension)
                    
                elif isinstance(value, _ArrayEntry):
                    # Cleanly format the ArrayEntry properties
                    dtype_str = value.dtype.name if hasattr(value.dtype, 'name') else str(value.dtype)
                    loc_str = value.storage.value

                    array_info = [f"storage: {loc_str}"]
                    if value.min_shape is not None:
                        array_info += [f"min: {value.min_shape}"]

                    if value.max_shape is not None:
                        array_info += [f"max: {value.max_shape}"]

                    if value.is_static == True:
                        array_info += [f"sized"]

                    array_info_str = ", ".join(array_info)
                    
                    print(f"{prefix}{connector}{key}: Array[{dtype_str}]({array_info_str})")
                    
                else:
                    # Fallback for unexpected types
                    print(f"{prefix}{connector}{key}: {value}")

        _print_node(self.fields)


def _supports_array_storage(dtype: Type):
    if _is_struct_type(dtype):
        return _validate_struct_datatypes(dtype, root=dtype.__qualname__)
    
    origin = typing.get_origin(dtype)
    if dtype == Array or origin is Array:
        args = typing.get_args(dtype)
        if args:
            return _supports_array_storage(args[0])
        return True

    try:
        return not np.dtype(dtype).hasobject
    except TypeError:
        return False


def _validate_struct_datatypes(cls: Type[Any], *, root: Optional[str] = None):

    is_serializable = {}
    
    attributes = {k: v for k, v in typing.get_type_hints(cls).items() if not k.startswith("_")}
    for k, v in attributes.items():
        key_path = k if root is None else f"{root}.{k}"
        
        origin = typing.get_origin(v)
        if v == Array or origin is Array:
            args = typing.get_args(v)
            elem_type = args[0] if args else float
            is_serializable[key_path] = _supports_array_storage(elem_type)
        elif _is_struct_type(v):
            nested_fields = _validate_struct_datatypes(v, root=key_path)
            is_serializable.update(nested_fields)
        else:
            is_serializable[key_path] = _supports_array_storage(v)
            
    return is_serializable


def create_schema(cls: Type) -> Schema:
    # Test if the module has attributes without type annotation
    if has_untyped_class_definitions(cls):
        raise TypeError(
            f"The struct {cls.__qualname__} declares attributes without type annotations."
        )

    # Verify that types can be serialized with Zarr
    is_serializable = _validate_struct_datatypes(cls)
    if not all(is_serializable.values()):
        incompatible_attrs = [k for k, v in is_serializable.items() if v == False]

        attr_str = "attribute" if len(incompatible_attrs) == 1 else "attributes"
        incompatible_list = format_list(incompatible_attrs)

        raise TypeError(
            f"The {attr_str} '{incompatible_list}' cannot be declared in struct '{cls.__qualname__}' because it does not support array serialization."
        )

    def _convert(dtype: Type):
        origin = typing.get_origin(dtype)
        if dtype == Array or origin is Array:
            args = typing.get_args(dtype)
            elem_type = args[0] if args else float
            return _ArrayEntry(
                np.dtype(elem_type),
                _ArrayLocation.AUTOMATIC,
                min_shape=None,
                max_shape=None,
                preferred_shape=None,
            )

        if _is_struct_type(dtype):
            return dtype

        return _ArrayEntry(
            np.dtype(dtype),
            _ArrayLocation.AUTOMATIC,
            min_shape=(1,),
            max_shape=(1,),
            preferred_shape=None,
        )

    attributes = {k: v for k, v in typing.get_type_hints(cls).items() if not k.startswith("_")}
    return Schema(
        fields={ k: _convert(v) for k, v in attributes.items() }
    )