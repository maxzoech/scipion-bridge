"""Internal access to the reference columns of Sets (see ``source.py``).

The library's own code uses these functions to create references (resolvers
reading files), read them ahead (loaders), rely on them (spilling) and show
them (reports). They are not part of the public API of the type system: a
Set with references behaves exactly like one owning the same values.

Every decision about references is made by one ``ReferencePolicy``: whether
loaders read rows ahead, and whether pickling keeps the bytes of a reference
or only the reference. The default reproduces the behaviour of the library;
a cost model can replace it (``set_policy``) without changing the callers.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Protocol, Tuple

import numpy as np

from .key_path import KeyPath
from .schema import ArrayEntryBase, Schema
from .source import Reference
from .storage import FieldKey, RootEngine, _BaseStorage

State = Literal["reference", "cached", "owned"]


class Stored(Protocol):
    """A Set or Struct: an object with a schema and a storage."""

    @property
    def storage(self) -> _BaseStorage: ...

    @classmethod
    def schema(cls) -> Schema: ...


@dataclass(frozen=True)
class ColumnState:
    """The provenance of a leaf column of a Set, as seen by reports.

    Attributes:
        path: The field path of the leaf, relative to the Set.
        state: ``reference`` (rows of a source, not read here), ``cached``
            (rows of a source, read) or ``owned`` (computed).
        rows: The rows of the leaf.
        nbytes: The bytes of the rows, read or not.
        source: A description of the source of a reference.
        domain: Where the source can be read.
    """

    path: FieldKey
    state: State
    rows: int
    nbytes: int
    source: str = ""
    domain: str = ""


class PicklingMode(enum.Enum):
    """What pickling a Set is for.

    ``TRANSFER``: to another process that uses it now (Ray's object store).
    ``SPILL``: to disk, until a stage takes it; the rows of references can be
    read again from their sources instead.
    """

    TRANSFER = "transfer"
    SPILL = "spill"


_mode: contextvars.ContextVar[PicklingMode] = contextvars.ContextVar(
    "scipion_bridge_pickling_mode",
    default=PicklingMode.TRANSFER,
)


@contextlib.contextmanager
def pickling(mode: PicklingMode) -> Iterator[None]:
    """Pickle Sets for ``mode`` inside the block."""
    token = _mode.set(mode)
    try:
        yield
    finally:
        _mode.reset(token)


def current_mode() -> PicklingMode:
    return _mode.get()


class ReferencePolicy:
    """The decisions about references; the defaults of the library.

    Every method gets the state of one column. A cost model replaces the
    policy with ``set_policy``.
    """

    def prefetch(self, state: ColumnState) -> bool:
        """Whether a loader reads the rows before it pushes an item."""
        return True

    def keep_bytes_when_spilling(self, state: ColumnState) -> bool:
        """Whether a spilled item keeps the read rows of a reference."""
        return False

    def keep_bytes_in_transfer(self, state: ColumnState) -> bool:
        """Whether an item sent to another process keeps the read rows."""
        return True

    def portable(self, state: ColumnState) -> bool:
        """Whether another process can read the source of a reference."""
        return True


_policy: ReferencePolicy = ReferencePolicy()


def policy() -> ReferencePolicy:
    return _policy


def set_policy(new: ReferencePolicy) -> None:
    global _policy
    _policy = new


# -- Creating and reading references -------------------------------------------
def set_reference(obj: Stored, path: FieldKey, reference: Reference) -> None:
    """Make the leaf ``path`` of ``obj`` hold rows of a source, unread.

    ``obj`` must own its storage (not be a view of another Set), and the
    reference must have as many rows as ``obj``.
    """
    engine = _engine_of(obj)
    key = obj.storage.root.extend(KeyPath(root=tuple((n, slice(None)) for n in path)))
    field_key = RootEngine._decompose(key)[0]
    length = obj.storage.get_length()
    if length is not None and length != len(reference):
        raise ValueError(
            f"A reference of {len(reference)} rows cannot hold field {path} "
            f"of {length} rows.",
        )
    engine.set_reference(field_key, reference)


def column_states(obj: Stored) -> Dict[FieldKey, ColumnState]:
    """The state of every initialized leaf of ``obj``, without reading any."""
    storage = obj.storage
    engine = _root_engine(obj)
    referenced = {} if engine is None else _reference_states(obj, engine)
    states: Dict[FieldKey, ColumnState] = {}
    for path, entry in obj.schema().tree_iter():
        key = storage.root.extend(path)
        if not isinstance(entry, ArrayEntryBase) or key not in storage:
            continue
        if path.path in referenced:
            states[path.path] = referenced[path.path][1]
            continue
        value = storage.read(key, entry)
        states[path.path] = ColumnState(
            path=path.path,
            state="owned",
            rows=len(value),
            nbytes=int(value.nbytes),
        )
    return states


def prefetch(obj: Stored) -> None:
    """Read the rows of the references of ``obj`` the policy reads ahead."""
    for path, state in column_states(obj).items():
        if state.state == "reference" and _policy.prefetch(state):
            obj.storage.read(
                obj.storage.root.extend(_key_path(path)),
                _entry_of(obj, path),
            )


def verify(obj: Stored) -> None:
    """Make every cached reference of ``obj`` whose rows changed an owned column.

    Writes in place into read rows are not seen by the storage. Before a
    reference is relied on instead of its bytes, its rows are compared with
    the digests taken when they were read.
    """
    engine = _root_engine(obj)
    if engine is None:
        return
    for field_key, reference in engine.references().items():
        if engine.is_cached(field_key) and not reference.matches(
            np.asarray(engine._data[field_key]),
        ):
            engine.own(field_key)


# -- Pickling ------------------------------------------------------------------
# A reference as pickled: the path of its leaf, the reference narrowed to the
# pickled rows, and whether the pickled batch holds its rows.
PickledReference = Tuple[FieldKey, Reference, bool]


def pickled_references(obj: Stored) -> Tuple[List[PickledReference], frozenset]:
    """The references to pickle with ``obj``, and the leaves whose bytes to omit.

    Decided by the current ``PicklingMode`` and the policy. Rows that are
    not read yet are never read here if the reference is portable.
    """
    engine = _root_engine(obj)
    if engine is None or not engine.references():
        # Most Sets have no references: pickling costs nothing more.
        return [], frozenset()
    mode = current_mode()
    if mode is PicklingMode.SPILL:
        verify(obj)
    pickled: List[PickledReference] = []
    omitted = set()
    for path, (reference, state) in _reference_states(obj, engine).items():
        keep = _keeps_bytes(mode, state)
        if not keep:
            omitted.add(path)
        pickled.append((path, reference, keep))
    return pickled, frozenset(omitted)


def _reference_states(
    obj: Stored,
    engine: RootEngine,
) -> Dict[FieldKey, Tuple[Reference, ColumnState]]:
    """The reference columns of ``obj``, narrowed to its rows, by path.

    Only the references of the engine are visited, never its other columns.
    Columns ``obj`` selects in a way a reference cannot follow (a single
    element) are left out: they are pickled as bytes.
    """
    prefix = RootEngine._decompose(obj.storage.root)[0]
    found: Dict[FieldKey, Tuple[Reference, ColumnState]] = {}
    for field_key in engine.references():
        if field_key[: len(prefix)] != prefix:
            continue
        path = field_key[len(prefix) :]
        reference = engine.reference_at(obj.storage.root.extend(_key_path(path)))
        if reference is None:
            continue
        found[path] = (
            reference,
            ColumnState(
                path=path,
                state="cached" if engine.is_cached(field_key) else "reference",
                rows=len(reference),
                nbytes=reference.nbytes,
                source=reference.source.describe(),
                domain=reference.source.domain,
            ),
        )
    return found


def _keeps_bytes(mode: PicklingMode, state: ColumnState) -> bool:
    """Whether pickling keeps the rows of a reference column."""
    if state.state == "reference":
        # Unread rows travel as the reference, unless the receiver cannot
        # read the source.
        return not _policy.portable(state)
    if mode is PicklingMode.SPILL:
        return _policy.keep_bytes_when_spilling(state) or not _policy.portable(state)
    return _policy.keep_bytes_in_transfer(state) or not _policy.portable(state)


def restore_references(obj: Stored, pickled: List[PickledReference]) -> None:
    """Restore the references of an unpickled ``obj``."""
    engine = _engine_of(obj)
    for path, reference, has_rows in pickled:
        key = obj.storage.root.extend(_key_path(path))
        field_key = RootEngine._decompose(key)[0]
        rows = engine._data[field_key] if has_rows else None
        engine.set_reference(field_key, reference, rows=rows)


# -- Helpers -------------------------------------------------------------------
def _key_path(path: FieldKey) -> KeyPath:
    return KeyPath(root=tuple((name, slice(None)) for name in path))


def _entry_of(obj: Stored, path: FieldKey) -> Any:
    return next(e for p, e in obj.schema().tree_iter() if p.path == path)


def _root_engine(obj: Stored) -> Optional[RootEngine]:
    engine = obj.storage.root_storage
    return engine if isinstance(engine, RootEngine) else None


def _engine_of(obj: Stored) -> RootEngine:
    engine = obj.storage
    assert isinstance(engine, RootEngine), "references need a Set owning its storage"
    return engine
