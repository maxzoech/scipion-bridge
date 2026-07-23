"""Schema definition and construction.

A Schema is a tree of Entry objects that describes the storage layout
derived from a Struct class definition.  ``create_schema`` is the public
factory that builds a Schema from a Struct type.
"""

import numpy as np
import typing
from dataclasses import dataclass
from typing import Type, Any, Dict, Optional, Set, TypeVar, Generic, Tuple

from ..utils.type_annotation import has_untyped_class_definitions
from ..utils.format import format_list
from ._type_checks import is_struct_type
from .entries import (
    Entry,
    _ArrayEntry,
    _ArrayLocation,
    _SchemaSetEntry,
    _StructEntry,
)


T = TypeVar("T")

class Array(Generic[T]):
    """Type annotation marker for variable-shape array fields."""
    pass


@dataclass
class Schema:
    """A tree of :class:`Entry` objects describing the storage layout of a Struct."""

    fields: Dict[str, Entry]

    def entries(self) -> Set[str]:
        return set(self.fields.keys())

    @property
    def is_static(self) -> bool:
        """True when every field in the schema has a fixed shape."""
        return all(entry.is_static for entry in self.fields.values())

    def print_tree(self, typename: Optional[str] = None) -> None:  # pragma: no cover
        """Print the schema in a hierarchical tree format."""
        header = typename if typename is not None else "/"
        if self.is_static:
            header += " (static size)"

        print(header)

        def _print_node(schema: "Schema", prefix: str = ""):
            items = list(schema.fields.items())
            for i, (key, entry) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "

                print(f"{prefix}{connector}{entry.format_entry(key)}")

                if entry.children is not None:
                    extension = "    " if is_last else "│   "
                    _print_node(entry.children, prefix + extension)

        _print_node(self)


def _supports_array_storage(dtype: Type):
    """Return whether *dtype* can be stored in an array backend."""
    if is_struct_type(dtype):
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
    """Recursively validate that all fields in *cls* support array storage."""
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
        elif is_struct_type(v):
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
    """Build a :class:`Schema` from a Struct class definition.

    Validates that all fields have type annotations and that their types
    support array serialization before constructing the schema tree.
    """
    # Reject classes with untyped attributes (e.g. ``x = 10``)
    if has_untyped_class_definitions(cls):
        raise TypeError(
            f"The struct {cls.__qualname__} declares attributes without type annotations."
        )

    # Verify that all types can be serialized
    is_serializable = _validate_struct_datatypes(cls)
    if not all(is_serializable.values()):
        incompatible_attrs = [k for k, v in is_serializable.items() if v == False]

        attr_str = "attribute" if len(incompatible_attrs) == 1 else "attributes"
        incompatible_list = format_list(incompatible_attrs)

        raise TypeError(
            f"The {attr_str} '{incompatible_list}' cannot be declared in struct "
            f"'{cls.__qualname__}' because it does not support array serialization."
        )

    def _convert(dtype: Type) -> Entry:
        from .set import Set as BridgeSet

        if isinstance(dtype, type) and issubclass(dtype, BridgeSet):
            schema = dtype.schema()
            return _SchemaSetEntry(schema)

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

        if is_struct_type(dtype):
            return _StructEntry(
                struct_cls=dtype,
                schema=create_schema(dtype),
            )

        return _ArrayEntry(
            np.dtype(dtype),
            _ArrayLocation.AUTOMATIC,
            min_shape=(1,),
            max_shape=(1,),
            preferred_shape=None,
        )

    attributes = {k: v for k, v in typing.get_type_hints(cls).items() if not k.startswith("_")}
    return Schema(
        fields={k: _convert(v) for k, v in attributes.items()}
    )