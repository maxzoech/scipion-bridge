"""Schema entry type definitions.

All Entry subclasses live here. Each entry describes what kind of data a
schema field holds (scalar array, ragged array, nested struct, etc.).

Entry is the abstract base: every concrete entry must implement
``is_static`` and ``format_entry``, and may optionally override
``children`` to expose a nested Schema for tree traversal.
"""

from __future__ import annotations

import abc
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Schema


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------

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
    def children(self) -> Optional[Schema]:
        """Return the nested schema if this entry contains children, else None."""
        return None


class SchemaConvertible(metaclass=abc.ABCMeta):
    """Base class for types that can be converted into schema entries.

    Both ``Struct`` and ``Set`` inherit from this class, providing a uniform
    interface that ``create_schema`` dispatches on polymorphically.

    Subclasses must implement:

    - ``to_schema_entry()``: Convert this type into a schema ``Entry``.
    - ``_validate_as_field(key_path)``: Validate that the type's fields
      support array serialization, returning a ``dict[str, bool]``.
    """

    @classmethod
    @abc.abstractmethod
    def to_schema_entry(cls) -> Entry:
        """Convert this type into a schema Entry for use in a parent schema."""
        ...

    @classmethod
    @abc.abstractmethod
    def _validate_as_field(cls, key_path: str) -> dict:
        """Validate that this type's fields support array serialization.

        Returns a dict mapping field paths to booleans (True if serializable).
        Called by schema validation when this type appears as a field
        annotation in a Struct.
        """
        ...


class _ArrayLocation(Enum):
    AUTOMATIC = "auto"


# ---------------------------------------------------------------------------
# Array entries
# ---------------------------------------------------------------------------

class _ArrayEntryBase(Entry):
    """Shared behaviour for all array-backed entry types."""

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
class _ArrayEntry(_ArrayEntryBase):
    """An array field whose shape may or may not be known."""

    dtype: np.dtype
    storage: _ArrayLocation
    preferred_shape: Optional[Tuple[int, ...]] = None
    min_shape: Optional[Tuple[int, ...]] = None
    max_shape: Optional[Tuple[int, ...]] = None

    @property
    def entry_name(self):
        return "Array"


@dataclass
class _ArraySetEntry(_ArrayEntryBase):
    """A fixed-shape array field inside a Set context."""

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
class _RaggedArraySetEntry(_ArrayEntryBase):
    """A variable-shape array field inside a Set context."""

    dtype: np.dtype
    storage: _ArrayLocation
    preferred_shape: Optional[Tuple[int, ...]] = None
    min_shape: Optional[Tuple[int, ...]] = None
    max_shape: Optional[Tuple[int, ...]] = None

    @property
    def entry_name(self):
        return "RaggedArraySet"


# ---------------------------------------------------------------------------
# Composite / structural entries
# ---------------------------------------------------------------------------

@dataclass
class _StructEntry(Entry):
    """Wraps a nested struct type and its eagerly-resolved schema."""

    struct_cls: type
    schema: Schema

    @property
    def is_static(self) -> bool:
        return self.schema.is_static

    def format_entry(self, name: str) -> str:
        return name

    @property
    def children(self) -> Schema:
        return self.schema


@dataclass
class _SchemaSetEntry(Entry):
    """Wraps a schema produced by a ``Set[X]`` field annotation."""

    schema: Schema

    @property
    def is_static(self):
        return self.schema.is_static

    def format_entry(self, name: str) -> str:
        return f"{name} (schema set)"

    @property
    def children(self) -> Schema:
        return self.schema
