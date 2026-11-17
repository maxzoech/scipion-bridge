"""Element ids of Sets and Structs (internal; see ``id_column.py``).

Every element of a Set, and every Struct, can carry an id, which lineage
uses to trace it through a pipeline. The ids are a column of the storage
that is not part of the schema: a ``RootEngine`` keeps an ``IdColumn`` per
Set (or Struct) it holds, keyed by the field key of its root. ``()`` is the
root of a Set or a standalone Struct, ``("particles",)`` the Set of a Struct
field, and ``("3", "particles")`` that field in slot 3 of a Collection.

Ids follow the rows, not the values: views select them, ``concat`` joins
them, writing a row gives it the ids of what was written, and writing values
into columns leaves them unchanged. Sets and Structs without ids behave as
before; ids appear where resolvers set them and where ``tag`` derives them.

Ids are stored for Sets with one axis only: the root of a Set, and the Set of
a Struct field or of a Collection slot. The members of nested Sets below two
axes (``Set[Class2D].particles``) have a ragged column per outer element,
which segments cannot describe yet; writing elements with ids there raises
``NotImplementedError``, so that ids are never lost silently.
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .id_column import IdColumn, generators, tag_column
from .key_path import KeyPath
from .references import Stored
from .schema import Schema
from .storage import FieldKey, RootEngine, _BaseStorage

NESTED_IDS_MESSAGE = (
    "Element ids of nested Sets below two axes are not supported yet: a Set "
    "cannot hold Structs whose nested Sets carry element ids, e.g. a "
    "Set[Class2D] of particles a pipeline read. Use a Collection instead, "
    "e.g. B.Collection[Class2D] (see Known Issues)."
)

# The id columns of an object and of the Sets nested in it, by their field
# key relative to the object's root.
HiddenIds = Dict[FieldKey, IdColumn]


def element_ids(obj: Stored) -> Optional[IdColumn]:
    """The ids of the elements of ``obj`` (one for a Struct), or ``None``.

    The elements of a view are selected from the ids of its Set. Elements
    written after the ids have no id.

    Raises:
        NotImplementedError: For a view of the members of every element of
            a Set (``groups["members"]``).
    """
    engine = _engine_of(obj.storage)
    root = obj.storage.root
    if engine is None or not engine._ids:
        return None
    if root.spans_several_axes():
        raise NotImplementedError(
            "A view of the members of every element of a Set has no single "
            "column of ids; index the outer Set first.",
        )
    column = engine._ids.get(_key(root))
    if column is None:
        return None
    index = RootEngine._compute_index(root)
    match index:
        case ():
            return column.padded(_rows(obj))
        case (selector,):
            return column.narrow(selector).padded(_rows(obj))
        case _:
            return None


def hidden_ids(obj: Stored) -> HiddenIds:
    """The id columns of ``obj`` and of the Sets nested in it.

    Nested columns exist only below a root without a selection: a Struct or
    a Collection slot, never the rows of a Set.
    """
    engine = _engine_of(obj.storage)
    if engine is None or not engine._ids:
        return {}
    root = obj.storage.root
    prefix = _key(root)
    found: HiddenIds = {}
    own = element_ids(obj)
    if own is not None:
        found[()] = own
    if RootEngine._compute_index(root) == ():
        found.update(
            {
                key[len(prefix) :]: column
                for key, column in engine._ids.items()
                if len(key) > len(prefix) and key[: len(prefix)] == prefix
            },
        )
    return found


def has_ids(obj: Stored) -> bool:
    """Whether some element of ``obj``, or of a Set nested in it, has an id."""
    return any(column.any_assigned for column in hidden_ids(obj).values())


def set_ids(obj: Stored, column: Optional[IdColumn]) -> None:
    """Give the elements of ``obj``, which owns its storage, the ids ``column``.

    ``None`` removes the ids.
    """
    engine = obj.storage
    assert isinstance(engine, RootEngine), "ids are set on objects owning their storage"
    key = _key(engine.root)
    if column is None:
        engine._ids.pop(key, None)
    else:
        engine._ids[key] = column


def restore_ids(obj: Stored, hidden: HiddenIds) -> None:
    """Restore the id columns ``hidden_ids`` returned, e.g. after unpickling."""
    engine = _engine_of(obj.storage)
    assert engine is not None, "ids are restored on objects with a root engine"
    prefix = _key(obj.storage.root)
    for key, column in hidden.items():
        engine._ids[prefix + key] = column


def share_ids(value: Stored, target_schema: Schema, engine: RootEngine) -> None:
    """Give ``engine``, the new storage of a cast of ``value``, the ids of
    ``value``; nested Sets the target schema lacks are dropped."""
    for key, column in hidden_ids(value).items():
        if key == () or _has_path(target_schema, key):
            engine._ids[key] = column


def check_placement(value: Stored, target_root: KeyPath) -> None:
    """Raise before ``value`` is written to ``target_root`` if element ids
    would land below two axes.

    Raises:
        NotImplementedError: If ``value`` is a Set with ids written into the
            elements of an outer Set, or a Struct with ids in its nested Sets
            written into a row of a Set.
    """
    hidden = hidden_ids(value)
    own = hidden.get(())
    nested = [column for key, column in hidden.items() if key != ()]
    axes = len(target_root.axis_indices())
    rows_selected = RootEngine._compute_index(target_root) != ()
    if (axes > 1 and own is not None and own.any_assigned) or (
        rows_selected and any(column.any_assigned for column in nested)
    ):
        raise NotImplementedError(NESTED_IDS_MESSAGE)


def write_ids(value: Stored, target: _BaseStorage, target_root: KeyPath) -> None:
    """Give the elements just written from ``value`` to ``target_root`` the
    ids of ``value``, as a copy of its rows keeps their identity.

    Elements of ``value`` without ids leave the written elements without
    ids. Callers run ``check_placement`` before writing the columns.
    """
    check_placement(value, target_root)
    hidden = hidden_ids(value)
    index = RootEngine._compute_index(target_root)
    engine = _engine_of(target)
    if engine is None:
        # Only a RootEngine stores ids, as only it stores references.
        return
    key = _key(target_root)
    if len(target_root.axis_indices()) <= 1:
        _assign(engine, key, index, hidden.get(()), _rows(value))
    if index == ():
        for nested_key, column in hidden.items():
            if nested_key != ():
                engine._ids[key + nested_key] = column


def populate_ids(target: Stored, items: Sequence[Stored]) -> None:
    """Give a Set built from ``items`` (Structs) their ids.

    Raises:
        NotImplementedError: If an item carries ids in its nested Sets,
            which would land below two axes.
    """
    hidden = [hidden_ids(item) for item in items]
    if any(
        key != () and column.any_assigned
        for found in hidden
        for key, column in found.items()
    ):
        raise NotImplementedError(NESTED_IDS_MESSAGE)
    columns = [found.get(()) for found in hidden]
    if all(column is None for column in columns):
        return
    set_ids(
        target,
        IdColumn.concat(
            [
                IdColumn.unassigned(1) if column is None else column
                for column in columns
            ],
        ),
    )


def concat_ids(sets: Sequence[Stored], result: Stored) -> None:
    """Give the concatenation ``result`` of ``sets`` the ids of their elements."""
    columns = [element_ids(s) for s in sets]
    if all(column is None for column in columns):
        return
    set_ids(
        result,
        IdColumn.concat(
            [
                IdColumn.unassigned(_rows(s)) if column is None else column
                for s, column in zip(sets, columns)
            ],
        ),
    )


def tag(obj: Stored, name: str, discriminator: Any) -> None:
    """Derive new ids for the elements of ``obj``, which owns its storage.

    Elements without ids keep none (see ``tag_column``).
    """
    column = element_ids(obj)
    if column is not None:
        set_ids(obj, tag_column(column, name, discriminator))


def ids_rng(obj: Stored, salt: Union[int, str, bytes]) -> List[np.random.Generator]:
    """One NumPy random generator per element of ``obj``, seeded from its id.

    Seeding random augmentations from element ids makes them reproducible:
    when a stage runs again, e.g. after a resume, every element draws the
    same values, so its id and its values agree. Pair it with ``tag`` to
    give views of an element ids of their own::

        views = particles[np.repeat(np.arange(len(particles)), 6)]
        views = views.tag("view", np.tile(np.arange(6), len(particles)))
        for view, rng in zip(views, B.ids_rng(views, salt="augment")):
            view.pixels = augment(view.pixels, rng)

    See "Randomness and resume" in the user guide.

    Args:
        obj: A Set, or a Struct (one generator).
        salt: Tells generators of the same elements for different purposes
            apart.

    Raises:
        ValueError: If some element has no id.
    """
    column = element_ids(obj)
    if column is None or not column.assigned:
        raise ValueError(
            "Random generators are seeded from element ids, and some elements "
            "have none: ids come from the inputs a resolver read, and rows a "
            "stage creates get theirs when the stage emits them. Seed from "
            "the elements they derive from, before creating new rows.",
        )
    return generators(column, salt)


def _assign(
    engine: RootEngine,
    key: FieldKey,
    index: tuple,
    column: Optional[IdColumn],
    rows: int,
) -> None:
    """Set the ids of the elements ``index`` of the Set at ``key``."""
    existing = engine._ids.get(key)
    if index == ():
        updated = column
    elif column is None and existing is None:
        updated = None
    else:
        (selector,) = index
        values = IdColumn.unassigned(rows) if column is None else column
        updated = (existing or IdColumn()).assign(selector, values)
    if updated is None:
        engine._ids.pop(key, None)
    else:
        engine._ids[key] = updated


def _has_path(schema: Schema, key: FieldKey) -> bool:
    """Whether ``schema`` has the nested field ``key``."""
    current: Optional[Schema] = schema
    for name in key:
        if current is None or name not in current.fields:
            return False
        current = current.fields[name].children
    return True


def _rows(obj: Stored) -> int:
    """The elements of a Set or a Collection, 1 for a Struct."""
    return len(obj) if isinstance(obj, Sized) else 1


def _key(root: KeyPath) -> FieldKey:
    return RootEngine._decompose(root)[0]


def _engine_of(storage: _BaseStorage) -> Optional[RootEngine]:
    engine = storage.root_storage
    return engine if isinstance(engine, RootEngine) else None
