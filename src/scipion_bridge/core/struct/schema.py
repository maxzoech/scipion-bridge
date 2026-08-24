import numpy as np

from typing import (
    Self,
    Type,
    Any,
    Tuple,
    Dict,
    Optional,
    Union,
    Set,
    TypeVar,
    Generic,
    Tuple,
    Iterator,
    Callable,
    ForwardRef,
    get_origin,
)

from ..utils.format import format_list
from ..utils.marker import Marker

from .entries import (
    Entry,
    SchemaConvertible,
    _ArrayEntry,
    _ArrayLocation,
)

from ._type_checks import is_array_marker

T = TypeVar("T")


class Array(Marker[T]):

    def __init__(
        self,
        dtype: Optional[np.dtype] = None,
        *,
        shape: Tuple[Union[int, None], ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype)

        self.shape = shape
        for k, v in kwargs.items():
            setattr(self, k, v)


def _is_array_type(cls: Type) -> bool:
    try:
        dt = np.dtype(cls)
        return dt.kind != "O"
    except (TypeError, ValueError):
        return False


class Schema:

    _schema_fields: dict[str, Array]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls_name = cls.__name__
        annotations = getattr(cls, "__annotations__", {})
        fields: dict[str, Array] = {}

        for field_name, field_type in annotations.items():
            origin_cls = get_origin(field_type)
            default_val = getattr(cls, field_name, None)

            if _is_array_type(field_type):
                fields[field_name] = Array(
                    dtype=np.dtype(field_type),
                    shape=(),
                    _scalar_type=field_type,
                )

            elif origin_cls is Array:
                if not default_val:
                    raise ValueError(
                        f"Field '{field_name}' in '{cls_name}' is typed as '{field_type}', "
                        f"but is missing a default Array specification. "
                        f"Expected: {field_name}: {field_type} = Array(shape=(...))"
                    )

                fields[field_name] = default_val

            else:
                raise TypeError(
                    f"Invalid type annotation '{cls!r}' for field '{field_name}' in Schema '{cls}'. "
                    f"Expected a primitive numeric/scalar type (e.g., float, int, bool) or an Array type (e.g., Array[float]), "
                    f"but got an unsupported or non-convertible type."
                )

        cls._schema_fields = fields

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


# @dataclass
# class Schema:
#     """A tree of :class:`Entry` objects describing the storage layout of a Struct."""

#     fields: Dict[str, Entry]

#     def entries(self) -> Set[str]:
#         return set(self.fields.field_names())

#     @property
#     def is_static(self) -> bool:
#         """True when every field in the schema has a fixed shape."""
#         return all(entry.is_static for entry in self.fields.values())

#     def tree_iter(self, root: str = "") -> Iterator[Tuple[str, Entry]]:
#         """Yield (path, entry) for all leaf entries in the schema."""
#         for field_name, entry in self.fields.items():
#             path = f"{root}.{field_name}" if root else field_name
#             if entry.children is not None:
#                 yield from entry.children.tree_iter(root=path)
#             else:
#                 yield path, entry

#     def iter_leaves(self, prefix: str = "") -> Iterator[Tuple[str, Entry]]:
#         """Yield (path, entry) for all leaf entries in the schema."""
#         yield from self.tree_iter(root=prefix)

#     def map_leaves(self, func: Callable[[str, Entry], Any], prefix: str = "") -> Dict[str, Any]:
#         """Apply func to all leaf entries, returning a dictionary mapping path -> result."""
#         return {
#             path: func(path, entry)
#             for path, entry in self.iter_leaves(prefix=prefix)
#         }

#     def print_tree(self, typename: Optional[str] = None) -> None:  # pragma: no cover
#         """Print the schema in a hierarchical tree format."""
#         header = typename if typename is not None else "/"
#         if self.is_static:
#             header += " (static size)"

#         print(header)

#         def _print_node(schema: "Schema", prefix: str = ""):
#             items = list(schema.fields.items())
#             for i, (field_name, entry) in enumerate(items):
#                 is_last = (i == len(items) - 1)
#                 connector = "└── " if is_last else "├── "

#                 print(f"{prefix}{connector}{entry.format_entry(field_name)}")

#                 if entry.children is not None:
#                     extension = "    " if is_last else "│   "
#                     _print_node(entry.children, prefix + extension)

#         _print_node(self)


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
