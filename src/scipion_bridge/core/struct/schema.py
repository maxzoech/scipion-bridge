"""Schema definitions for Struct and Set data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Type, Union, TypeAlias

import numpy as np

KeyPath: TypeAlias = Tuple[str, ...]


class SchemaConvertible(metaclass=abc.ABCMeta):
    """Abstract base for classes or objects convertible to a schema representation."""

    @classmethod
    @abc.abstractmethod
    def schema(cls) -> "Schema":
        ...

    @classmethod
    @abc.abstractmethod
    def default(cls) -> "SchemaConvertible":
        """Create a default, unspecialized instance from the type."""
        ...

    @abc.abstractmethod
    def convert_to_entry(self) -> "Entry":
        """Convert this instance into a schema Entry tree representation."""
        ...

    @classmethod
    def print_schema(cls) -> None:
        cls.schema().print_tree()

    def __set_name__(self, owner: type, name: str) -> None:
        pass


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


@dataclass
class ArrayEntryBase(Entry):
    """Shared behaviour for all array-backed entry types."""

    dtype: np.dtype
    shape: Tuple[Optional[int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, np.dtype):
            object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "shape", tuple(self.shape))

    @property
    @abc.abstractmethod
    def entry_name(self) -> str:
        ...

    @property
    def is_static(self) -> bool:
        """Static if all dimensions are defined integers >= 0."""
        return all(isinstance(dim, int) and dim >= 0 for dim in self.shape)

    def format_entry(self, name: str) -> str:
        dtype_str = self.dtype.name if hasattr(self.dtype, "name") else str(self.dtype)
        return f"{name}: {self.entry_name}[{dtype_str}], shape: {list(self.shape)})"


@dataclass
class ArrayEntry(ArrayEntryBase):
    """An array field whose shape may or may not be fully static."""

    @property
    def entry_name(self) -> str:
        return "Array"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        entry_cls = ArraySetEntry if self.is_static else RaggedArraySetEntry
        return entry_cls(
            dtype=self.dtype,
            shape=self.shape,
            capacity=capacity,
        )


@dataclass
class ArraySetEntry(ArrayEntryBase):
    """A fixed-shape array field inside a Set context."""

    capacity: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(dim is None for dim in self.shape):
            raise ValueError(f"ArraySet shape must be fully static, got: {self.shape}")

    @property
    def is_static(self) -> bool:
        return True

    @property
    def entry_name(self) -> str:
        return "ArraySet"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return ArraySetEntry(
            dtype=self.dtype,
            shape=self.shape,
            capacity=capacity if capacity is not None else self.capacity,
        )


@dataclass
class RaggedArraySetEntry(ArrayEntryBase):
    """A variable-shape array field inside a Set."""

    capacity: Optional[int] = None

    @property
    def is_static(self) -> bool:
        return False

    @property
    def entry_name(self) -> str:
        return "RaggedArraySet"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return RaggedArraySetEntry(
            dtype=self.dtype,
            shape=self.shape,
            capacity=capacity if capacity is not None else self.capacity,
        )


@dataclass
class SchemaEntry(Entry):
    """Wraps a nested struct type and its schema for record instantiation."""

    schema: "Schema"

    @property
    def is_static(self) -> bool:
        return self.schema.is_static

    @property
    def children(self) -> Optional["Schema"]:
        return self.schema

    def format_entry(self, name: str) -> str:
        return f"{name} (struct)"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return SchemaEntry(schema=self.schema.to_set_schema(capacity=capacity))


@dataclass
class SchemaSetEntry(SchemaEntry):
    """Wraps a Set[Foo] container entry capable of instantiating Foo elements."""

    capacity: Optional[int] = None

    @property
    def is_static(self) -> bool:
        return self.schema.is_static and self.capacity is not None

    def format_entry(self, name: str) -> str:
        size_str = self.capacity if self.capacity is not None else "dynamic"
        cls_name = self.schema.dtype.__name__ if self.schema.dtype else "struct"
        return f"{name}: Set[{cls_name}](size: {size_str})"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return SchemaSetEntry(
            schema=self.schema.to_set_schema(capacity=capacity),
            capacity=capacity if capacity is not None else self.capacity,
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

    def tree_iter(self, root: KeyPath = ()) -> Iterator[Tuple[KeyPath, ArrayEntryBase]]:
        """Yield (path_tuple, entry) for all leaf array entries in the schema."""
        for field_name, entry in self.fields.items():
            path = (*root, field_name)
            if entry.children is not None:
                yield from entry.children.tree_iter(root=path)
            elif isinstance(entry, ArrayEntryBase):
                yield path, entry

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


# Backward compatibility aliases
_ArrayEntryBase = ArrayEntryBase
_ArrayEntry = ArrayEntry
_ArraySetEntry = ArraySetEntry
_RaggedArraySetEntry = RaggedArraySetEntry
_SchemaEntry = SchemaEntry
_SchemaSetEntry = SchemaSetEntry
