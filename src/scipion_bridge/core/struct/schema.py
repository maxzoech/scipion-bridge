import abc
from dataclasses import dataclass

import numpy as np

from typing import Iterator, Optional, Dict, Tuple, Type, Union, Any


class SchemaConvertible(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def default(cls) -> "SchemaConvertible":
        """Create a default, unspecialized instance from the type."""
        ...

    @abc.abstractmethod
    def specialize(
        self, context: Optional[Dict[Any, Any]] = None
    ) -> "SchemaConvertible":
        """Specialize this specification with dimension bindings from context."""
        ...

    @abc.abstractmethod
    def validate(self, other: Any) -> None:
        """Validate that another instance matches this specification's type and structure."""
        ...

    @abc.abstractmethod
    def convert_to_entry(self) -> "Entry":
        """Convert this instance into a schema Entry tree representation."""
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

    @abc.abstractmethod
    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        """Transform this entry into its Set-vectorized entry representation."""
        ...


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
    def entry_name(self) -> str: ...

    @property
    def is_static(self) -> bool:
        """Static if all dimensions are defined integers >= 0."""
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

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        if self.is_static:
            return _ArraySetEntry(
                dtype=self.dtype,
                shape=self.shape,
                capacity=capacity,
            )
        else:
            return _RaggedArraySetEntry(
                dtype=self.dtype,
                shape=self.shape,
                capacity=capacity,
            )


class _ArraySetEntry(_ArrayEntryBase):
    """A fixed-shape array field inside a Set context."""

    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[Union[int, None], ...],
        capacity: Optional[int] = None,
    ) -> None:
        # Enforce that ArraySet only receives fully concrete integer dimensions
        if any(dim is None for dim in shape):
            raise ValueError(f"ArraySet shape must be fully static, got: {shape}")

        super().__init__(dtype=dtype, shape=shape)
        self.capacity = capacity

    @property
    def is_static(self) -> bool:
        return True

    @property
    def entry_name(self) -> str:
        return "ArraySet"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return _ArraySetEntry(
            dtype=self.dtype, shape=self.shape, capacity=capacity or self.capacity
        )


class _RaggedArraySetEntry(_ArrayEntryBase):
    """A variable-shape array field inside a Set."""

    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[Union[int, None], ...],
        capacity: Optional[int] = None,
    ) -> None:
        super().__init__(dtype=dtype, shape=shape)
        self.capacity = capacity

    @property
    def is_static(self) -> bool:
        return False

    @property
    def entry_name(self) -> str:
        return "RaggedArraySet"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return _RaggedArraySetEntry(
            dtype=self.dtype,
            shape=self.shape,
            capacity=(capacity or self.capacity),
        )


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

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return _SchemaEntry(schema=self.schema.to_set_schema(capacity=capacity))


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
        return self.schema.is_static and self.capacity is not None

    def format_entry(self, name: str) -> str:
        size_str = self.capacity if self.capacity is not None else "dynamic"
        cls_name = self.schema.dtype.__name__ if self.schema.dtype else "struct"
        return f"{name}: Set[{cls_name}](size: {size_str})"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return _SchemaSetEntry(
            schema=self.schema.to_set_schema(capacity=capacity),
            capacity=self.capacity,
        )


@dataclass
class Schema:
    """A tree of :class:`Entry` objects describing the storage layout of a Struct."""

    dtype: Optional[Type]
    fields: Dict[str, Entry]

    def to_set_schema(self, capacity: Optional[int] = None) -> "Schema":
        """Transform this schema into its Set-vectorized representation."""
        return Schema(
            dtype=self.dtype,
            fields={
                name: entry.to_set_entry(capacity=capacity)
                for name, entry in self.fields.items()
            },
        )

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
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "

                print(f"{prefix}{connector}{entry.format_entry(field_name)}")

                if entry.children is not None:
                    extension = "    " if is_last else "│   "
                    _print_node(entry.children, prefix + extension)

        _print_node(self)
