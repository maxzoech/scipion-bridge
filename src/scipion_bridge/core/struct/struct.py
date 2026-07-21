import numpy as np
import zarr

from dataclasses import dataclass
from functools import wraps
from enum import Enum

import typing
from typing import Type, Any, Dict, Optional

from ..utils.type_annotation import has_untyped_class_definitions
from ..utils.format import format_list


class ArrayLocation(Enum):
    AUTOMATIC = "auto"
    HOST_MEMORY = "host_memory"

@dataclass
class Schema:

    @dataclass
    class Array:
        dtype: np.dtype
        location: ArrayLocation

    tree: Dict[str, Any]


def _supports_array_storage(dtype: Type):
    if issubclass(dtype, Struct):
        return _validate_struct_datatypes(dtype, root=dtype.__qualname__)

    return not np.dtype(dtype).hasobject


def _validate_struct_datatypes(cls: Type["Struct"], *, root: Optional[str] = None):
    is_serializable = {}
    
    for k, v in cls._attributes().items():
        key_path = k if root is None else f"{root}.{k}"
        
        if issubclass(v, Struct):
            nested_fields = _validate_struct_datatypes(v, root=key_path)
            is_serializable.update(nested_fields)
        else:
            is_serializable[key_path] = _supports_array_storage(v)
            
    return is_serializable

class Struct:
    
    _scipion_bridge_schema: Schema

    @classmethod
    def _attributes(cls):
        return { k: v for k, v in typing.get_type_hints(cls).items() if k != "_scipion_bridge_schema" }

    @classmethod
    def _generate_schema(cls):
        def _convert(dtype: Type):
            if issubclass(dtype, Struct):
                return dtype._scipion_bridge_schema.tree

            return Schema.Array(np.dtype(dtype), ArrayLocation.AUTOMATIC)

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