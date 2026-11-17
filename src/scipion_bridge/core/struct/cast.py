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

from typing import Any, Type, TypeVar, Union, cast as typing_cast

from .collection import Collection
from .key_path import STRUCT_ROOT
from .references import Stored
from .set import Set
from .sharing import share_columns
from .storage import RootEngine
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
            share_columns(value, target, engine)
            return typing_cast(S, target(capacity=value.capacity, storage=engine))
        case Struct(is_descriptor=False) if issubclass(target, Struct):
            engine = RootEngine(root=STRUCT_ROOT)
            share_columns(value, target, engine)
            return typing_cast(S, target(storage=engine))
        case _:
            raise TypeError(
                f"Cannot cast {type(value).__name__} to {target.__name__}: a Set "
                "casts to a Set type, a Struct to a Struct class.",
            )
