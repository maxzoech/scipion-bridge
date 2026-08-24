import abc
from dataclasses import dataclass

import numpy as np

from typing import Iterator, Optional, Dict, Tuple, Type, Union

class SchemaConvertable(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def convert_to_entry(self) -> "Entry":
        ...

    @abc.abstractmethod
    def is_static(self) -> bool:
        ...

class Entry(metaclass=abc.ABCMeta):
    """Abstract base for all schema field entries."""

    @property
    @abc.abstractmethod
    def is_static(self) -> bool:
        """True when the entry's shape is fully known at schema-creation time."""
        ...

    @abc.abstractmethod
    def format_entry(self, name: str) -> str:
        """Return a human-readable label for *name* used by ``print_tree``."""
        ...

    @property
    def children(self) -> Optional["Schema"]:
        """Return the nested schema if this entry contains children, else None."""
        return None


class _ArrayEntryBase(Entry):
    """Shared behaviour for all array-backed entry types."""

    dtype: np.dtype
    shape: Tuple[Union[int, None], ...]

    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[Union[int, None], ...],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype
        self.shape = tuple(shape)

    @property
    @abc.abstractmethod
    def entry_name(self) -> str:
        ...

    @property
    def is_static(self) -> bool:
        """Static if all dimensions are defined integers (no None / dynamic dims)."""
        return all(isinstance(dim, int) and dim >= 0 for dim in self.shape)

    def format_entry(self, name: str) -> str:
        dtype_str = self.dtype.name if hasattr(self.dtype, "name") else str(self.dtype)
        shape_str = list(self.shape)

        return f"{name}: {self.entry_name}[{dtype_str}], shape: {shape_str})"

class _ArrayEntry(_ArrayEntryBase):
    """An array field whose shape may or may not be fully static."""

    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[Union[int, None], ...],
    ) -> None:
        super().__init__(dtype=dtype, shape=shape)

    @property
    def entry_name(self) -> str:
        return "Array"

class _ArraySetEntry(_ArrayEntryBase):
    """A fixed-shape array field inside a Set context."""

    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[int, ...],
    ) -> None:
        # Enforce that ArraySet only receives fully concrete integer dimensions
        if any(dim is None or dim < 0 for dim in shape):
            raise ValueError(f"ArraySet shape must be fully static, got: {shape}")

        super().__init__(dtype=dtype, shape=shape)

    @property
    def is_static(self) -> bool:
        return True

    @property
    def entry_name(self) -> str:
        return "ArraySet"

class _RaggedArraySetEntry(_ArrayEntryBase):
    """A variable-shape array field inside a Set."""

    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[Union[int, None], ...],
    ) -> None:
        super().__init__(dtype=dtype, shape=shape)

    @property
    def is_static(self) -> bool:
        return False

    @property
    def entry_name(self) -> str:
        return "RaggedArraySet"

class _SchemaEntry(Entry):
    """Wraps a nested struct type and its schema for record instantiation."""

    def __init__(self, schema: "Schema") -> None:
        super().__init__()
        self.schema = schema

    @property
    def is_static(self) -> bool:
        return self.schema.is_static

    @property
    def children(self) -> "Schema":
        return self.schema

    def format_entry(self, name: str) -> str:
        return f"{name} (struct)"
    

class _SchemaSetEntry(_SchemaEntry):
    """Wraps a Set[Foo] container entry capable of instantiating Foo elements."""

    def __init__(
        self,
        schema: "Schema",
        capacity: Optional[int] = None,
    ) -> None:
        super().__init__(schema=schema)
        self.capacity = capacity

    @property
    def is_static(self) -> bool:
        return False #self.schema.is_static and self.capacity is not None

    def format_entry(self, name: str) -> str:
        size_str = self.capacity if self.capacity is not None else "dynamic"
        cls_name = self.schema.dtype.__name__ if self.schema.dtype else "struct"
        return f"{name}: Set[{cls_name}](size: {size_str})"


@dataclass
class Schema:
    """A tree of :class:`Entry` objects describing the storage layout of a Struct."""

    dtype: Optional[Type]
    fields: Dict[str, Entry]

    @property
    def is_static(self) -> bool:
        """True when every field in the schema has a fixed shape."""
        return all(entry.is_static for entry in self.fields.values())

    def tree_iter(self, root: str = "") -> Iterator[Tuple[str, Entry]]:
        """Yield (path, entry) for all leaf entries in the schema."""
        for field_name, entry in self.fields.items():
            path = f"{root}.{field_name}" if root else field_name
            if entry.children is not None:
                yield from entry.children.tree_iter(root=path)
            else:
                yield path, entry

#     def iter_leaves(self, prefix: str = "") -> Iterator[Tuple[str, Entry]]:
#         """Yield (path, entry) for all leaf entries in the schema."""
#         yield from self.tree_iter(root=prefix)

#     def map_leaves(self, func: Callable[[str, Entry], Any], prefix: str = "") -> Dict[str, Any]:
#         """Apply func to all leaf entries, returning a dictionary mapping path -> result."""
#         return {
#             path: func(path, entry)
#             for path, entry in self.iter_leaves(prefix=prefix)
#         }

    def print_tree(self, typename: Optional[str] = None) -> None:  # pragma: no cover
        """Print the schema in a hierarchical tree format."""
        header = typename if typename is not None else "/"
        if self.is_static:
            header += " (static size)"

        print(header)

        def _print_node(schema: "Schema", prefix: str = ""):
            items = list(schema.fields.items())
            for i, (field_name, entry) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "

                print(f"{prefix}{connector}{entry.format_entry(field_name)}")

                if entry.children is not None:
                    extension = "    " if is_last else "│   "
                    _print_node(entry.children, prefix + extension)

        _print_node(self)


# # ---------------------------------------------------------------------------
# # Validation helpers
# # ---------------------------------------------------------------------------

# def _supports_array_storage(cls: Type):
#     """Return whether *cls* can be stored in an array backend."""
#     if isinstance(cls, type) and issubclass(cls, SchemaConvertible):
#         return _validate_struct_datatypes(cls, root=cls.__qualname__)

#     try:
#         return not np.cls(cls).hasobject
#     except TypeError:
#         return False


# def _validate_struct_datatypes(cls: Type[Any], *, root: Optional[str] = None):
#     """Recursively validate that all fields in *cls* support array storage."""
#     is_serializable = {}

#     attributes = {k: v for k, v in typing.get_type_hints(cls).items() if not k.startswith("_")}
#     for k, v in attributes.items():
#         field_name_path = k if root is None else f"{root}.{k}"

#         if is_array_marker(v):
#             elem_type = v.cls()
#             is_serializable[field_name_path] = _supports_array_storage(elem_type)
#         elif isinstance(v, type) and issubclass(v, SchemaConvertible):
#             nested = v._validate_as_field(field_name_path)
#             is_serializable.update(nested)
#         else:
#             is_serializable[field_name_path] = _supports_array_storage(v)

#     return is_serializable


# # ---------------------------------------------------------------------------
# # Schema construction
# # ---------------------------------------------------------------------------

# def create_schema(cls: Type) -> Schema:
#     """Build a :class:`Schema` from a Struct class definition.

#     Validates that all fields have type annotations and that their types
#     support array serialization before constructing the schema tree.
#     """
#     # Reject classes with untyped attributes (e.g. ``x = 10``)
#     if has_untyped_class_definitions(cls):
#         raise TypeError(
#             f"The struct {cls.__qualname__} declares attributes without type annotations."
#         )

#     # Verify that all types can be serialized
#     is_serializable = _validate_struct_datatypes(cls)
#     if not all(is_serializable.values()):
#         incompatible_attrs = [k for k, v in is_serializable.items() if v == False]

#         attr_str = "attribute" if len(incompatible_attrs) == 1 else "attributes"
#         incompatible_list = format_list(incompatible_attrs)

#         raise TypeError(
#             f"The {attr_str} '{incompatible_list}' cannot be declared in struct "
#             f"'{cls.__qualname__}' because it does not support array serialization."
#         )

#     def _convert(field: Type) -> Entry:
#         # SchemaConvertible types (Struct, Set) know how to produce their own entry
#         if isinstance(field, type) and issubclass(field, SchemaConvertible):
#             return field.to_schema_entry()

#         # origin = typing.get_origin(cls)
#         if is_array_marker(field):
#             v: Any = field
#             elem_type = v.cls()

#             if (
#                 all(isinstance(x, int) for x in v.shape()) and
#                 len(v.shape()) > 0
#             ):
#                 static_shape = v.shape()
#             else:
#                 static_shape = None

#             return _ArrayEntry(
#                 np.cls(elem_type),
#                 _ArrayLocation.AUTOMATIC,
#                 min_shape=static_shape,
#                 max_shape=static_shape,
#                 preferred_shape=None,
#             )

#         return _ArrayEntry(
#             np.cls(field),
#             _ArrayLocation.AUTOMATIC,
#             min_shape=(1,),
#             max_shape=(1,),
#             preferred_shape=None,
#         )

#     attributes = {k: v for k, v in typing.get_type_hints(cls).items() if not k.startswith("_")}
#     return Schema(
#         fields={k: _convert(v) for k, v in attributes.items()}
#     )
