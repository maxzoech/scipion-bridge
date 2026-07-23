import numpy as np
import abc
from dataclasses import dataclass
from enum import Enum
import typing
from typing import Type, Any, Dict, Optional, TypeVar, Generic, Tuple, Set

from ..utils.type_annotation import has_untyped_class_definitions
from ..utils.format import format_list

T = TypeVar("T")
class Array(Generic[T]):
    pass

class Entry(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def is_static(self) -> bool:
        ...


class _ArrayLocation(Enum):
    AUTOMATIC = "auto"


@typing.runtime_checkable
class PrintableEntry(typing.Protocol):
    def format_entry(self, name: str) -> str:
        ...


class ArrayEntryBase(Entry):
    dtype: np.dtype
    storage: _ArrayLocation

    @property
    @abc.abstractmethod
    def entry_name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def min_shape(self) -> Optional[Tuple[int, ...]]:
        ...

    @property
    @abc.abstractmethod
    def max_shape(self) -> Optional[Tuple[int, ...]]:
        ...

    @property
    def is_static(self) -> bool:
        return self.min_shape is not None and self.min_shape == self.max_shape

    def format_entry(self, name: str) -> str:
        dtype_str = self.dtype.name if hasattr(self.dtype, 'name') else str(self.dtype)
        loc_str = self.storage.value
        array_info = [f"storage: {loc_str}"]
        if self.min_shape is not None:
            array_info.append(f"min: {self.min_shape}")
        if self.max_shape is not None:
            array_info.append(f"max: {self.max_shape}")
        array_info_str = ", ".join(array_info)
        return f"{name}: {self.entry_name}[{dtype_str}]({array_info_str})"


@dataclass
class _ArrayEntry(ArrayEntryBase):
    dtype: np.dtype
    storage: _ArrayLocation
    preferred_shape: Optional[Tuple[int, ...]] = None
    min_shape: Optional[Tuple[int, ...]] = None
    max_shape: Optional[Tuple[int, ...]] = None

    @property
    def entry_name(self):
        return "Array"

@dataclass
class _ArraySetEntry(ArrayEntryBase):
    dtype: np.dtype
    storage: _ArrayLocation
    shape: Tuple[int, ...]

    @property
    def is_static(self) -> bool:
        return True

    @property
    def min_shape(self) -> Tuple[int, ...]:
        return self.shape
    
    @property
    def max_shape(self) -> Tuple[int, ...]:
        return self.shape

    @property
    def preferred_shape(self) -> Tuple[int, ...]:
        return self.shape

    @property
    def entry_name(self):
        return "ArraySet"

    def format_entry(self, name: str) -> str:
        dtype_str = self.dtype.name if hasattr(self.dtype, 'name') else str(self.dtype)
        loc_str = self.storage.value
        return f"{name}: ArraySet[{dtype_str}](storage: {loc_str}, shape: {self.shape})"

@dataclass
class _RaggedArraySetEntry(ArrayEntryBase):
    dtype: np.dtype
    storage: _ArrayLocation
    preferred_shape: Optional[Tuple[int, ...]] = None
    min_shape: Optional[Tuple[int, ...]] = None
    max_shape: Optional[Tuple[int, ...]] = None

    @property
    def entry_name(self):
        return "RaggedArraySet"

@dataclass
class Schema:
    fields: Dict[str, Entry]

    def entries(self) -> Set[str]:
        return set(self.fields.keys())

    @property
    def is_static(self) -> bool:
        for entry in self.fields.values():
            from .struct import _is_struct_type
            if _is_struct_type(entry):
                return create_schema(entry).is_static # type: ignore
            elif isinstance(entry, Entry):
                return entry.is_static
            else:
                return False
        return True


    def print_tree(self, typename: Optional[str] = None) -> None:  # pragma: no cover
        """
        Prints a Schema object in a hierarchical tree format.
        """

        header = typename if typename is not None else "/"
        if self.is_static:
            header += " (static size)"

        print(header)
        
        def _print_node(node: dict, prefix: str = ""):
            from .struct import _is_struct_type

            items = list(node.items())
            for i, (key, value) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "

                if _is_struct_type(value):
                    print(f"{prefix}{connector}{value.__name__} (opaque struct)")
                elif isinstance(value, Schema):

                    extension = "    " if is_last else "│   "
                    print(f"{prefix}{connector}{key}")
                    _print_node(value.fields, prefix + extension)

                elif isinstance(value, ArrayEntryBase):
                    print(f"{prefix}{connector}{value.format_entry(key)}")
                    
                else:
                    # Fallback for unexpected types
                    print(f"{prefix}{connector}{key}: {value}")

        _print_node(self.fields)


def _supports_array_storage(dtype: Type):
    from .struct import _is_struct_type

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
    from .struct import _is_struct_type
    from .set import Set

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
        elif issubclass(v, Set):
            wrapped_type = v.item_type()
            is_serializable = _validate_struct_datatypes(wrapped_type, root=key_path)
            
            is_serializable[key_path] = all(is_serializable.values())
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
        from .struct import _is_struct_type
        from .set import Set as BridgeSet

        if isinstance(dtype, type) and issubclass(dtype, BridgeSet):
            schema = dtype.schema()
            return schema
        
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