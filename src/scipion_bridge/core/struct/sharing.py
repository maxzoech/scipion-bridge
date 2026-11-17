"""Sharing the columns of a Set or Struct with a new one (internal).

``share_columns`` gives the storage of a new object of another (or the same)
type the columns of an existing one as read-only views, with their
references and element ids, without reading or copying rows. ``cast`` and
``tag`` are built on it; see ``cast.py`` for the rules.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import awkward as ak
import numpy as np

from . import identity, references
from .key_path import KeyPath
from .references import Stored
from .schema import (
    ArrayEntryBase,
    CollectionEntry,
    Entry,
    Schema,
    SchemaEntry,
    SchemaSetEntry,
)
from .storage import FieldKey, RootEngine


def share_columns(value: Stored, target: type, engine: RootEngine) -> None:
    """Make ``engine``, the storage of a new ``target``, share the columns of
    ``value``: its initialized leaves of the target's fields, their
    references and the element ids.

    Raises:
        TypeError: If a field of both types differs in kind, dtype or shape,
            or the types share no fields.
    """
    names = f"{type(value).__name__} to {target.__name__}"
    source_schema, target_schema = value.schema(), target.schema()
    _check_structure(source_schema, target_schema, (), names)
    source_leaves = {
        path.path: (path, entry) for path, entry in source_schema.tree_iter()
    }
    target_leaves = list(target_schema.tree_iter())
    if not any(path.path in source_leaves for path, _ in target_leaves):
        raise TypeError(f"Cannot cast {names}: they share no fields.")

    storage = value.storage
    root_engine = storage.root_storage
    referenced = (
        references._reference_states(value, root_engine)
        if isinstance(root_engine, RootEngine) and root_engine.references()
        else {}
    )
    for path, entry in target_leaves:
        if path.path not in source_leaves:
            continue
        source_path, source_entry = source_leaves[path.path]
        source_key = storage.root.extend(source_path)
        if source_key not in storage:
            continue
        key = engine.root.extend(path)
        if path.path in referenced:
            reference, state = referenced[path.path]
            rows = (
                _read_only(storage.read(source_key, source_entry))
                if state.state == "cached"
                else None
            )
            engine.set_reference(_field_key(key), reference, rows=rows)
        else:
            engine.write(key, entry, _read_only(storage.read(source_key, source_entry)))
    identity.share_ids(value, target_schema, engine)


def _check_structure(
    source: Schema,
    target: Schema,
    where: Tuple[str, ...],
    names: str,
) -> None:
    """Raise if a field of both schemas differs in kind, dtype or shape."""
    for name in [name for name in target.fields if name in source.fields]:
        path = (*where, name)
        a, b = source.fields[name], target.fields[name]
        mismatch = _mismatch(a, b)
        if mismatch is not None:
            raise TypeError(
                f"Cannot cast {names}: field '{'.'.join(path)}' is {mismatch}. Casting shares columns "
                "and never converts values.",
            )
        if a.children is not None and b.children is not None:
            _check_structure(a.children, b.children, path, names)


def _mismatch(a: Entry, b: Entry) -> Optional[str]:
    """How the entries of a field differ, or None if the field can be shared."""
    kind_a, kind_b = _kind(a), _kind(b)
    if kind_a != kind_b:
        return f"a {kind_a} in one type and a {kind_b} in the other"
    match (a, b):
        case (CollectionEntry(size=size_a), CollectionEntry(size=size_b)) if (
            size_a != size_b
        ):
            return (
                f"a Collection of size {size_a} in one type and {size_b} in the other"
            )
        case (ArrayEntryBase(), ArrayEntryBase()) if (a.dtype, tuple(a.shape)) != (
            b.dtype,
            tuple(b.shape),
        ):
            return (
                f"{_describe_leaf(a)} in one type and {_describe_leaf(b)} in the other"
            )
        case _:
            return None


def _kind(entry: Entry) -> str:
    match entry:
        case SchemaSetEntry():
            return "nested Set"
        case SchemaEntry():
            return "nested Struct"
        case CollectionEntry():
            return "Collection"
        case ArrayEntryBase():
            return "leaf"
        case _:
            return type(entry).__name__


def _describe_leaf(entry: ArrayEntryBase) -> str:
    return f"{np.dtype(entry.dtype).name} of shape {list(entry.shape)}"


def _field_key(key: KeyPath) -> FieldKey:
    return RootEngine._decompose(key)[0]


def _read_only(column: Any) -> Any:
    """A read-only view of a NumPy column; Awkward Arrays are immutable."""
    match column:
        case np.ndarray():
            view = column.view()
            view.flags.writeable = False
            return view
        case ak.Array():
            return column
        case _:
            raise TypeError(f"Unexpected column type {type(column).__name__}.")
