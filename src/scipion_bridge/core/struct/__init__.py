"""Core Struct and Set abstractions and Arrow storage engine."""

from .collection import Collection
from .exceptions import UninitializedFieldError
from .identity import ids_rng
from .key_path import IndexType, KeyPath
from .schema import (
    ArrayEntry,
    ArrayEntryBase,
    ArraySetEntry,
    CollectionEntry,
    Entry,
    RaggedArraySetEntry,
    Schema,
    SchemaConvertible,
    SchemaEntry,
    SchemaSetEntry,
    SetEntryBase,
)
from .set import Set, concat
from .storage import StorageView
from .struct import Arg, Array, Dim, Struct

__all__ = [
    # Types
    "Struct",
    "Set",
    "Collection",
    "concat",
    "ids_rng",
    "Array",
    "Arg",
    "Dim",
    "KeyPath",
    "IndexType",
    "UninitializedFieldError",
    # Schemas and storage, for sinks and backends
    "Schema",
    "SchemaConvertible",
    "Entry",
    "ArrayEntryBase",
    "ArrayEntry",
    "SetEntryBase",
    "ArraySetEntry",
    "RaggedArraySetEntry",
    "SchemaEntry",
    "SchemaSetEntry",
    "CollectionEntry",
    "StorageView",
]
