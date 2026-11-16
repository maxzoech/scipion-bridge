"""Changing the type of a Set or Struct without copying its columns (internal).

``cast(value, T)`` returns a ``T`` whose leaves share the columns of the
leaves of ``value`` with the same field path: no rows are read and no bytes
move, and references to rows of input files stay references. Fields of ``T``
that ``value`` lacks stay uninitialized; fields ``T`` lacks are dropped.
Matching is structural, by field path, not by class hierarchy; a field both
types have must have the same kind, dtype and shape, since values are never
converted.

The result holds read-only views of the buffers of ``value``. Writes through
its storage copy a buffer first (copy-on-write) or replace the column, so
writing to the result never changes ``value``; writing in place into a buffer
read from it raises. Writes to ``value`` after the cast can show through the
views of its owned buffers.

The streaming operator ``Op.astype`` is built on it. It is not part of the
public API of the type system.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Type, TypeVar, Union, cast as typing_cast

import awkward as ak
import numpy as np

from . import references
from .collection import Collection
from .key_path import STRUCT_ROOT, KeyPath
from .references import Stored
from .schema import (
    ArrayEntryBase,
    CollectionEntry,
    Entry,
    Schema,
    SchemaEntry,
    SchemaSetEntry,
)
from .set import Set
from .storage import FieldKey, RootEngine
from .struct import Struct

S = TypeVar("S", bound=Union[Set, Struct])


def check_target(target: Any) -> None:
    """Raise if ``target`` is not a type ``cast`` can produce.

    Raises:
        NotImplementedError: For a ``Collection`` type.
        TypeError: For anything but a subscripted Set type or a Struct class.
    """
    match target:
        case type() if issubclass(target, Collection):
            raise NotImplementedError(
                f"Casting to {target.__name__} is not implemented; cast its items.",
            )
        case type() if issubclass(target, Set) and target._dtype is not None:
            pass
        case type() if issubclass(target, Struct):
            pass
        case _:
            raise TypeError(
                f"Can only cast to a Set type (e.g. Set[Particle]) or a Struct "
                f"class, got {target!r}.",
            )


def cast(value: Stored, target: Type[S]) -> S:
    """A ``target`` sharing the columns of ``value``, without reading or copying rows.

    Args:
        value: A Set (or a view of one) or a Struct (or an element of a Set).
        target: A Set type for a Set, a Struct class for a Struct.

    Raises:
        NotImplementedError: For a Collection, as the value or the target.
        TypeError: If ``value`` cannot be cast to ``target``: a Set to a
            Struct or the other way round, a view of the members of every
            element of a Set, types without a common field, or a field whose
            kind, dtype or shape differs.
    """
    check_target(target)
    match value:
        case Collection():
            raise NotImplementedError(
                f"Casting a {type(value).__name__} is not implemented; cast its items.",
            )
        case Set(is_descriptor=False) if issubclass(target, Set):
            if value.storage.root.spans_several_axes():
                raise TypeError(
                    "A view of the members of every element of a Set has no "
                    "rows of its own to cast; index the outer Set first, e.g. "
                    "groups[i].members.",
                )
            engine = RootEngine()
            _share_leaves(value, target, engine)
            return typing_cast(S, target(capacity=value.capacity, storage=engine))
        case Struct(is_descriptor=False) if issubclass(target, Struct):
            engine = RootEngine(root=STRUCT_ROOT)
            _share_leaves(value, target, engine)
            return typing_cast(S, target(storage=engine))
        case _:
            raise TypeError(
                f"Cannot cast {type(value).__name__} to {target.__name__}: a Set "
                "casts to a Set type, a Struct to a Struct class.",
            )


def _share_leaves(value: Stored, target: type, engine: RootEngine) -> None:
    """Make the leaves of ``engine`` share the initialized leaves of ``value``."""
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
