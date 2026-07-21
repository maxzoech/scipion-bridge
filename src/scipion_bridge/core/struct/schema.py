import numpy as np
import zarr

from dataclasses import dataclass
from functools import wraps
from enum import Enum

import typing
from typing import Type, Any, Dict, Optional, TypeVar, Generic, Tuple

from ..utils.type_annotation import has_untyped_class_definitions
from ..utils.format import format_list

T = TypeVar("T")

class ArrayLocation(Enum):
    AUTOMATIC = "auto"
    HOST_MEMORY = "host_memory"

class Entry:
    pass # Marker Type

@dataclass
class ArrayEntry(Entry):
    dtype: np.dtype
    storage: ArrayLocation
    min_shape: Optional[Tuple[int]]
    max_shape: Optional[Tuple[int]]
    preferred_shape: Optional[Tuple[int]]


@dataclass
class Schema:
    tree: Dict[str, Entry]


def _supports_array_storage(dtype: Type):
    if isinstance(dtype, type) and issubclass(dtype, Struct):
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


def _validate_struct_datatypes(cls: Type["Struct"], *, root: Optional[str] = None):
    is_serializable = {}
    
    for k, v in cls._attributes().items():
        key_path = k if root is None else f"{root}.{k}"
        
        origin = typing.get_origin(v)
        if v == Array or origin is Array:
            args = typing.get_args(v)
            elem_type = args[0] if args else Any
            is_serializable[key_path] = _supports_array_storage(elem_type) # type: ignore
        elif isinstance(v, type) and issubclass(v, Struct):
            nested_fields = _validate_struct_datatypes(v, root=key_path)
            is_serializable.update(nested_fields)
        else:
            is_serializable[key_path] = _supports_array_storage(v)
            
    return is_serializable

class Array(Generic[T]):
    pass

class Struct:
    
    _scipion_bridge_schema: Schema

    @classmethod
    def _attributes(cls):
        return { k: v for k, v in typing.get_type_hints(cls).items() if k != "_scipion_bridge_schema" }

    @classmethod
    def _generate_schema(cls):
        def _convert(dtype: Type):
            origin = typing.get_origin(dtype)
            if dtype == Array or origin is Array:
                args = typing.get_args(dtype)
                elem_type = args[0] if args else float
                return ArrayEntry(
                    np.dtype(elem_type),
                    ArrayLocation.AUTOMATIC,
                    min_shape=None,
                    max_shape=None,
                    preferred_shape=None,
                )

            if isinstance(dtype, type) and issubclass(dtype, Struct):
                return dtype._scipion_bridge_schema.tree

            return ArrayEntry(
                np.dtype(dtype),
                ArrayLocation.AUTOMATIC,
                min_shape=(1,),
                max_shape=(1,),
                preferred_shape=None,
            )

        return Schema(
            tree={ k: _convert(v) for k, v in cls._attributes().items() }
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

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

        cls._scipion_bridge_schema = cls._generate_schema()


    def __init__(self):
        print("Init the zarr storage here...")

    
    @classmethod
    def print_schema(cls) -> None:
        """
        Prints a Schema object in a hierarchical tree format.
        """
        print(cls.__qualname__)
        
        def _print_node(node: dict, prefix: str = ""):
            items = list(node.items())
            for i, (key, value) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "
                
                if isinstance(value, dict):
                    # Print the parent node (e.g., 'ctf')
                    print(f"{prefix}{connector}{key}")
                    # Extend the prefix for the children
                    extension = "    " if is_last else "│   "
                    _print_node(value, prefix + extension)
                    
                elif isinstance(value, ArrayEntry):
                    # Cleanly format the ArrayEntry properties
                    dtype_str = value.dtype.name if hasattr(value.dtype, 'name') else str(value.dtype)
                    loc_str = value.storage.value

                    array_info = [f"storage: {loc_str}"]
                    if value.min_shape is not None:
                        array_info += [f"min: {value.min_shape}"]

                    if value.max_shape is not None:
                        array_info += [f"max: {value.max_shape}"]

                    array_info_str = ", ".join(array_info)
                    
                    print(f"{prefix}{connector}{key}: Array[{dtype_str}]({array_info_str})")
                    
                else:
                    # Fallback for unexpected types
                    print(f"{prefix}{connector}{key}: {value}")

        _print_node(cls._scipion_bridge_schema.tree)