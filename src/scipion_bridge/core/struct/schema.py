"""Schema definitions for Struct and Set data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cache
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    TypeAlias,
    overload,
)
from functools import cache, cached_property
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    TypeAlias,
    overload,
)

import numpy as np

from .key_path import KeyPath


class SchemaConvertible(metaclass=abc.ABCMeta):
    """Abstract base for classes or objects convertible to a schema representation."""

    @classmethod
    @abc.abstractmethod
    def schema(cls) -> "Schema": ...

    @classmethod
    @abc.abstractmethod
    def default(cls) -> "SchemaConvertible":
        """Create a default, unspecialized instance from the type."""
        ...

    @property
    @cache
    def entry(self) -> Entry:
        return self.convert_to_entry()

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

    @cached_property
    def children(self) -> Optional["Schema"]:
        """Return the nested schema if this entry contains children, else None."""
        return None

    capacity: Optional[int] = None

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
    def entry_name(self) -> str: ...

    @property
    def is_static(self) -> bool:
        """Static if all dimensions are defined integers >= 0."""
        return all(isinstance(dim, int) and dim >= 0 for dim in self.shape)

    def format_entry(self, name: str) -> str:
        dtype_str = self.dtype.name if hasattr(self.dtype, "name") else str(self.dtype)
        return f"{name}: {self.entry_name}[{dtype_str}], shape: {list(self.shape)}"


class SetEntryBase(Entry):

    capacity: Optional[int] = None


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
class ArraySetEntry(ArrayEntryBase, SetEntryBase):
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
class RaggedArraySetEntry(ArrayEntryBase, SetEntryBase):
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

    @cached_property
    def children(self) -> Optional["Schema"]:
        return self.schema

    def format_entry(self, name: str) -> str:
        return f"{name} (struct)"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return SchemaEntry(schema=self.schema.to_set_schema(capacity=capacity))


@dataclass
class SchemaSetEntry(SchemaEntry, SetEntryBase):
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
            schema=self.schema,
            capacity=self.capacity,
        )


@dataclass
class CollectionEntry(Entry):
    """Wraps a statically-sized Collection container entry."""

    element_entry: Entry
    size: int

    @property
    def is_static(self) -> bool:
        return self.element_entry.is_static

    @cached_property
    def children(self) -> Optional["Schema"]:
        return Schema(
            dtype=None,
            fields={str(i): self.element_entry for i in range(self.size)},
        )

    def format_entry(self, name: str) -> str:
        match self.element_entry:
            case SchemaEntry(schema=Schema(dtype=type() as dtype)):
                elem_name = dtype.__name__
            case Entry() as entry:
                elem_name = entry.__class__.__name__

        return f"{name}: Collection[{elem_name}](size: {self.size})"

    def to_set_entry(self, capacity: Optional[int] = None) -> "Entry":
        return CollectionEntry(
            element_entry=self.element_entry.to_set_entry(capacity=capacity),
            size=self.size,
        )


@dataclass
class Schema:
    """A tree of :class:`Entry` objects describing the storage layout of a Struct."""

    dtype: Optional[Type]
    fields: Dict[str, Entry]
    capacity: Optional[int] = None

    def to_set_schema(self, capacity: Optional[int] = None) -> "Schema":
        """Transform this schema into its Set-vectorized representation."""
        return Schema(
            dtype=self.dtype,
            fields={
                name: entry.to_set_entry(capacity=capacity)
                for name, entry in self.fields.items()
            },
            capacity=capacity,
        )

    @property
    def is_static(self) -> bool:
        """True when every field in the schema has a fixed shape."""
        return all(entry.is_static for entry in self.fields.values())

    def tree_iter(
        self,
        *others: "Schema",
        root: KeyPath = KeyPath(root=()),
    ) -> Iterator[Tuple[Any, ...]]:
        """Yield (path, *entries) for all leaf array entries across this and optional other schemas.

        Strictly validates that all schemas have matching field keys and compatible hierarchy
        at every level.

        Args:
            *others: Additional Schema instances to traverse in parallel.
            root: Base KeyPath prefix for relative path accumulation.

        Raises:
            ValueError: If field names do not match across schemas at any level.
            TypeError: If a field is a nested branch in one schema but a leaf in another,
                       or if a leaf entry is not an ArrayEntryBase.
        """
        for field_name, entries in _common_entries(
            self.fields, *(other.fields for other in others)
        ):
            path = root.append(field_name)
            has_children = [e.children is not None for e in entries]

            if any(has_children):
                if not all(has_children):
                    raise TypeError(
                        f"Structural mismatch at field '{field_name}': "
                        f"some schemas define a nested branch while others define a leaf."
                    )

                child_schema, *other_children = (e.children for e in entries)
                assert child_schema is not None

                yield from child_schema.tree_iter(*other_children, root=path)

            elif all(isinstance(e, ArrayEntryBase) for e in entries):
                yield (path, *entries)

            else:
                raise TypeError(
                    f"Structural mismatch at field '{field_name}': "
                    f"expected ArrayEntryBase leaf nodes, got {[type(e).__name__ for e in entries]}."
                )

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

                if isinstance(entry, CollectionEntry):
                    if entry.children is not None and entry.size > 0:
                        extension = "    " if is_last else "│   "
                        if entry.size == 1:
                            print(
                                f"{prefix}{extension}└── {entry.element_entry.format_entry('0')}"
                            )
                            if entry.element_entry.children is not None:
                                _print_node(
                                    entry.element_entry.children,
                                    prefix + extension + "    ",
                                )
                        else:
                            print(
                                f"{prefix}{extension}├── {entry.element_entry.format_entry('0')}"
                            )
                            if entry.element_entry.children is not None:
                                _print_node(
                                    entry.element_entry.children,
                                    prefix + extension + "│   ",
                                )
                            if entry.size > 2:
                                print(f"{prefix}{extension}├── ...")
                            last_idx = entry.size - 1
                            print(
                                f"{prefix}{extension}└── {last_idx} (struct, collapsed)"
                            )
                elif entry.children is not None:
                    extension = "    " if is_last else "│   "
                    _print_node(entry.children, prefix + extension)

        _print_node(self)


def _common_entries(*dicts: Dict[str, Any]) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    if not dicts:
        return

    primary_dict, *other_dicts = dicts
    primary_keys = set(primary_dict.keys())

    for other_dict in other_dicts:
        keys = other_dict.keys()

        if primary_keys != keys:
            missing = primary_keys - keys
            extra = keys - primary_keys
            raise ValueError(
                f"Schema field mismatch: missing fields {missing}, extra fields {extra}"
            )

    for k in primary_dict:
        yield (k, tuple(d[k] for d in dicts))
