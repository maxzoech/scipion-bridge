import numpy as np
import zarr

import abc
from dataclasses import dataclass
from functools import wraps, cache
from enum import Enum

import typing
from typing import Type, Any, Dict, Optional, TypeVar, Generic, Tuple, Set, get_type_hints

from ..utils.type_annotation import has_untyped_class_definitions
from ..utils.format import format_list

T = TypeVar("T")

class Entry:
    pass # Marker Type

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



@dataclass
class Schema:
    tree: Dict[str, Entry]

    def entries(self) -> Set[str]:
        return set(self.tree.keys())

    def print_tree(self, typename: Optional[str] = None) -> None:
        """
        Prints a Schema object in a hierarchical tree format.
        """
        print(typename if typename is not None else "/")
        
        def _print_node(node: dict, prefix: str = ""):
            items = list(node.items())
            for i, (key, value) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "

                if isinstance(value, type) and issubclass(value, Struct):
                    schema = create_schema(value)

                    print(f"{prefix}{connector}{key}")
                    extension = "    " if is_last else "│   "
                    _print_node(schema.tree, prefix + extension)
                    
                elif isinstance(value, _ArrayEntry):
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
                    print(f"{prefix}{connector}{key}: {value} (fallback)")

        _print_node(self.tree)


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
    
    attributes = typing.get_type_hints(cls)
    for k, v in attributes.items():
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

        if issubclass(dtype, Struct):
            return dtype

        return _ArrayEntry(
            np.dtype(dtype),
            _ArrayLocation.AUTOMATIC,
            min_shape=(1,),
            max_shape=(1,),
            preferred_shape=None,
        )

    attributes = typing.get_type_hints(cls)
    return Schema(
        tree={ k: _convert(v) for k, v in attributes.items() }
    )


class Struct:

    @classmethod
    @cache
    def schema(cls) -> Schema:
        return create_schema(cls)
   
    def __init__(self) -> None:
        
        self._zarr_group = zarr.group(overwrite=True)
        for key, value in self.schema().tree.items():
            if isinstance(value, _ArrayEntry):
                initial_shape = (
                    value.preferred_shape or
                    value.min_shape or 
                    (1,) # Fallback for None, allowing it to be appended to later
                )

                self._zarr_group.create_array(
                    name=key,
                    shape=initial_shape,
                    dtype=value.dtype,
                )


    def __setattr__(self, name, value):
        schema = type(self).schema()        
        if name not in schema.entries() or isinstance(value, Struct):
            super().__setattr__(name, value)
        else:

            input_has_shape = hasattr(value, "shape") or hasattr(value, "__len__")

            entry = schema.tree[name]
            value = np.array(value).astype(entry.dtype)
            orig_shape = value.shape

            value = np.reshape(value, [-1])
            is_scalar = not input_has_shape and value.size == 1

            buffer = self._zarr_group[name]
            buffer.resize(value.shape)

            buffer.attrs['orig_shape'] = orig_shape
            buffer.attrs['is_scalar'] = is_scalar

            buffer[:] = value

    def __getattribute__(self, name):
        schema = type(self).schema()
        attrs = set(schema.entries())

        if name not in attrs:
            return super().__getattribute__(name)
        elif isinstance(schema.tree[name], type) and issubclass(schema.tree[name], Struct):
            return super().__getattribute__(name)
        else:
            buffer = self._zarr_group[name]
            orig_shape = buffer.attrs['orig_shape']
            is_scalar = buffer.attrs['is_scalar']

            value = np.array(buffer).reshape(orig_shape)
            if is_scalar:
                value = value.item()

            return value
    
    @classmethod
    def print_schema(cls) -> None:
        schema = create_schema(cls)
        schema.print_tree(cls.__qualname__)

    def print_storage(self) -> None:
        print(self._zarr_group.tree())