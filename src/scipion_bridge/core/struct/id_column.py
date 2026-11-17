"""Columns of element ids, stored as segments (internal; see ``identity.py``).

The elements of a Set have ids, which lineage uses to trace them through a
pipeline. They are stored like references: as segments of consecutive
elements, so a chunk of particles read from one stack has one segment however
many particles it holds. A segment is one of:

- **positional**: element ``k`` is ``(namespace, positions[k])``, e.g. the rows
  of a stack (a ``range``) or Scipion item ids (an ``int64`` array);
- **mixed**: element ``k`` is ``(namespaces[indices[k]], positions[k])``:
  positional ids of many namespaces in one segment;
- **explicit**: element ``k`` has the 16-byte id ``explicit[k]``, e.g. after
  tagging with hashed values;
- **unassigned**: the elements have no id yet, e.g. rows a stage created.

Views narrow a column, ``concat`` joins columns (merging adjacent runs), and
writes of rows assign ids to positions, all without materializing the ids.

Runs of positional ids shorter than ``SHORT_RUN`` are joined into mixed
segments. Rows of small stacks (a few particles each) and the members of a
class, selected from many chunks, would otherwise give a segment per element;
a mixed segment stores them in three arrays instead, so the number of
segments stays a fraction of the number of elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..utils.hashing import ID_BYTES, item_id

# The positions of a positional segment.
Positions = Union[range, NDArray[np.int64]]
# Runs of positional ids shorter than this are joined into mixed segments.
SHORT_RUN = 32
# A selection of elements: an index, a slice, positions or a mask.
Selector = Union[int, slice, NDArray[np.integer], NDArray[np.bool_]]


def element_id(namespace: bytes, position: int) -> bytes:
    """The 16-byte id of element ``position`` of ``namespace``."""
    return item_id(namespace, position.to_bytes(8, "little", signed=True))


@dataclass(frozen=True, eq=False)
class IdSegment:
    """The ids of consecutive elements of a column.

    Attributes:
        positions: The positions of a positional or mixed segment;
            ``range(n)`` for the other kinds, which only count their
            elements.
        namespace: The namespace of a positional segment, else ``None``.
        explicit: The ``(n, 16)`` ids of an explicit segment, else ``None``.
        namespaces: The ``(k, 16)`` namespaces of a mixed segment, else
            ``None``.
        indices: The namespace of every element of a mixed segment, as an
            index into ``namespaces``.
    """

    positions: Positions
    namespace: Optional[bytes] = None
    explicit: Optional[NDArray[np.uint8]] = None
    namespaces: Optional[NDArray[np.uint8]] = None
    indices: Optional[NDArray[np.int32]] = None

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def assigned(self) -> bool:
        """Whether the elements have ids."""
        return (
            self.namespace is not None
            or self.explicit is not None
            or self.namespaces is not None
        )

    @property
    def mixed(self) -> bool:
        return self.namespaces is not None

    def namespace_of(self) -> NDArray[np.uint8]:
        """The ``(n, 16)`` namespace of every element of a positional or
        mixed segment."""
        if self.namespaces is not None and self.indices is not None:
            return self.namespaces[self.indices]
        else:
            assert self.namespace is not None
            row = np.frombuffer(self.namespace, dtype=np.uint8)
            return np.broadcast_to(row, (len(self), ID_BYTES))

    def take(self, selector: Union[slice, NDArray[np.int64]]) -> IdSegment:
        """The elements ``selector`` of this segment."""
        match selector:
            case slice():
                positions: Positions = self.positions[selector]
            case np.ndarray():
                positions = np.asarray(self.positions, dtype=np.int64)[selector]
        explicit = None if self.explicit is None else self.explicit[selector]
        if self.namespace is None and self.namespaces is None:
            positions = range(len(positions))
        if self.namespaces is not None and self.indices is not None:
            namespaces, indices = _pruned(self.namespaces, self.indices[selector])
            return IdSegment(positions, namespaces=namespaces, indices=indices)
        else:
            return IdSegment(positions, self.namespace, explicit)

    def ids(self) -> NDArray[np.uint8]:
        """The ``(n, 16)`` ids of the elements.

        Raises:
            ValueError: If the elements have no ids.
        """
        if self.explicit is not None:
            return self.explicit
        elif self.namespace is not None or self.namespaces is not None:
            return np.array(
                [
                    np.frombuffer(element_id(row.tobytes(), int(p)), dtype=np.uint8)
                    for row, p in zip(self.namespace_of(), self.positions)
                ],
                dtype=np.uint8,
            ).reshape(len(self), ID_BYTES)
        else:
            raise ValueError(f"{len(self)} elements have no ids.")


@dataclass(frozen=True, eq=False)
class IdColumn:
    """The ids of the elements of a Set, in order, as segments."""

    segments: Tuple[IdSegment, ...] = ()

    @staticmethod
    def of(namespace: bytes, positions: Positions) -> IdColumn:
        """Elements ``positions`` of ``namespace``."""
        if isinstance(positions, np.ndarray):
            positions = positions.astype(np.int64, copy=False)
        return IdColumn((IdSegment(positions, namespace),))

    @staticmethod
    def unassigned(n: int) -> IdColumn:
        """``n`` elements without ids."""
        return IdColumn((IdSegment(range(n)),) if n > 0 else ())

    @staticmethod
    def explicit_ids(ids: NDArray[np.uint8]) -> IdColumn:
        """Elements with the ``(n, 16)`` ids ``ids``."""
        assert ids.ndim == 2 and ids.shape[1] == ID_BYTES
        return IdColumn((IdSegment(range(len(ids)), None, ids),) if len(ids) else ())

    def __len__(self) -> int:
        return sum(len(segment) for segment in self.segments)

    @property
    def assigned(self) -> bool:
        """Whether every element has an id."""
        return all(segment.assigned for segment in self.segments)

    @property
    def any_assigned(self) -> bool:
        """Whether some element has an id."""
        return any(segment.assigned for segment in self.segments)

    def narrow(self, selector: Selector) -> IdColumn:
        """The elements ``selector`` (an index, a slice, positions or a mask).

        Positions beyond the column are elements without ids, e.g. rows of a
        Set written after its ids.
        """
        positions = _positions(selector, len(self))
        if len(positions) and int(positions.max()) >= len(self):
            return self.padded(int(positions.max()) + 1)._gather(positions)
        else:
            return self._gather(positions)

    def padded(self, n: int) -> IdColumn:
        """This column with elements without ids appended up to ``n``."""
        if len(self) >= n:
            return self
        else:
            return IdColumn.concat([self, IdColumn.unassigned(n - len(self))])

    def assign(self, selector: Selector, values: IdColumn) -> IdColumn:
        """This column with the ids of ``values`` at ``selector``.

        The column grows with elements without ids to cover ``selector``.

        Raises:
            ValueError: If ``selector`` selects another number of elements
                than ``values`` holds.
        """
        base = self.padded(_extent(selector, len(self)))
        positions = _positions(selector, len(base))
        if len(positions) != len(values):
            raise ValueError(
                f"Cannot assign the ids of {len(values)} elements to "
                f"{len(positions)} elements.",
            )
        gather = np.arange(len(base), dtype=np.int64)
        gather[positions] = len(base) + np.arange(len(values), dtype=np.int64)
        return IdColumn.concat([base, values])._gather(gather)

    @staticmethod
    def concat(columns: Sequence[IdColumn]) -> IdColumn:
        """The elements of ``columns``, in order; adjacent runs are joined."""
        return IdColumn(_merged([s for column in columns for s in column.segments]))

    def filled(self, namespace: bytes) -> IdColumn:
        """This column with every element without an id given
        ``(namespace, its position in the column)``."""
        segments: List[IdSegment] = []
        offset = 0
        for segment in self.segments:
            if segment.assigned:
                segments.append(segment)
            else:
                positions = range(offset, offset + len(segment))
                segments.append(IdSegment(positions, namespace))
            offset += len(segment)
        return IdColumn(_merged(segments))

    def digest_parts(self) -> List[bytes]:
        """An encoding of the ids, equal for columns of equal ids, however
        they are split into segments (O(elements), for small columns)."""
        n = len(self)
        kinds = np.zeros(n, dtype=np.uint8)
        ids = np.zeros((n, ID_BYTES), dtype=np.uint8)
        positions = np.zeros(n, dtype=np.int64)
        offset = 0
        for segment in self.segments:
            span = slice(offset, offset + len(segment))
            if segment.explicit is not None:
                kinds[span] = 2
                ids[span] = segment.explicit
            elif segment.assigned:
                kinds[span] = 1
                ids[span] = segment.namespace_of()
                positions[span] = segment.positions
            offset += len(segment)
        return [kinds.tobytes(), ids.tobytes(), positions.tobytes()]

    def ids(self) -> NDArray[np.uint8]:
        """The ``(n, 16)`` ids of the elements.

        Raises:
            ValueError: If some elements have no ids.
        """
        parts = [segment.ids() for segment in self.segments]
        return np.concatenate([np.empty((0, ID_BYTES), dtype=np.uint8), *parts])

    def elements(self) -> List[Any]:
        """Every element as ``(namespace, position)``, its explicit id, or
        ``None`` without an id; for tests and reports."""
        result: List[Any] = []
        for segment in self.segments:
            if segment.explicit is not None:
                result.extend(bytes(row) for row in segment.explicit)
            elif segment.assigned:
                result.extend(
                    (row.tobytes(), int(p))
                    for row, p in zip(segment.namespace_of(), segment.positions)
                )
            else:
                result.extend([None] * len(segment))
        return result

    def _gather(self, positions: NDArray[np.int64]) -> IdColumn:
        """The elements at ``positions``, which lie inside the column."""
        if len(positions) == 0:
            return IdColumn()
        starts = np.cumsum([0, *(len(segment) for segment in self.segments)])
        owners = np.searchsorted(starts, positions, side="right") - 1
        runs = np.split(
            np.arange(len(positions)),
            np.flatnonzero(np.diff(owners)) + 1,
        )
        parts = [
            self.segments[int(owners[run[0]])].take(
                _compact(positions[run] - starts[int(owners[run[0]])]),
            )
            for run in runs
        ]
        return IdColumn(_merged(parts))


def tag_column(column: IdColumn, name: str, discriminator: Any) -> IdColumn:
    """The ids of new elements derived from ``column``'s by ``name``.

    An integer discriminator (one per element, or one for all) derives a
    namespace per value, so the result stays segments: the ``k``-th view of
    the elements of a stack is ``(item_id(stack, name, k), row)``. Any other
    discriminator (e.g. a seed, augmentation parameters or bytes) is hashed
    with the id of every element into explicit ids. Elements without ids
    keep none: they are new elements already.

    Raises:
        ValueError: If ``discriminator`` has another number of values than
            ``column`` has elements.
    """
    values = np.asarray(discriminator)
    if values.ndim == 0:
        values = np.broadcast_to(values, (len(column),))
    if len(values) != len(column):
        raise ValueError(
            f"A tag of {len(column)} elements needs as many discriminators, "
            f"got {len(values)}.",
        )
    label = name.encode()
    tagged: List[IdSegment] = []
    offset = 0
    for segment in column.segments:
        segment_values = values[offset : offset + len(segment)]
        offset += len(segment)
        if not segment.assigned:
            tagged.append(segment)
        elif segment.namespace is not None and _is_integer(segment_values):
            tagged.extend(_tag_runs(segment, label, segment_values))
        elif segment.mixed and _is_integer(segment_values):
            tagged.append(_tag_mixed(segment, label, segment_values))
        else:
            tagged.append(
                IdSegment(
                    range(len(segment)),
                    None,
                    _hashed(segment.ids(), label, segment_values),
                ),
            )
    return IdColumn(_merged(tagged))


def generators(column: IdColumn, salt: Union[int, str, bytes]) -> List[Any]:
    """One NumPy random generator per element, seeded from its id and ``salt``.

    Raises:
        ValueError: If some elements have no ids.
    """
    salt_bytes = _value_bytes(np.asarray(salt))
    return [
        np.random.default_rng(
            int.from_bytes(item_id(row.tobytes(), salt_bytes), "little"),
        )
        for row in column.ids()
    ]


def _tag_runs(
    segment: IdSegment,
    label: bytes,
    values: NDArray[Any],
) -> List[IdSegment]:
    """The tagged segments of a positional segment: one per run of values."""
    assert segment.namespace is not None
    namespace = segment.namespace
    runs = np.split(np.arange(len(values)), np.flatnonzero(np.diff(values)) + 1)
    return [
        IdSegment(
            segment.take(_compact(run.astype(np.int64))).positions,
            item_id(
                namespace, label, int(values[run[0]]).to_bytes(8, "little", signed=True)
            ),
        )
        for run in runs
        if len(run)
    ]


def _tag_mixed(segment: IdSegment, label: bytes, values: NDArray[Any]) -> IdSegment:
    """The tagged mixed segment: a namespace per namespace and value, derived
    as for positional segments."""
    assert segment.namespaces is not None and segment.indices is not None
    pairs, inverse = np.unique(
        np.stack([segment.indices.astype(np.int64), values.astype(np.int64)]),
        axis=1,
        return_inverse=True,
    )
    namespaces = np.array(
        [
            np.frombuffer(
                item_id(
                    segment.namespaces[index].tobytes(),
                    label,
                    int(value).to_bytes(8, "little", signed=True),
                ),
                dtype=np.uint8,
            )
            for index, value in pairs.T
        ],
        dtype=np.uint8,
    ).reshape(-1, ID_BYTES)
    return IdSegment(
        np.asarray(segment.positions, dtype=np.int64),
        namespaces=namespaces,
        indices=np.asarray(inverse, dtype=np.int32).ravel(),
    )


def _hashed(
    ids: NDArray[np.uint8], label: bytes, values: NDArray[Any]
) -> NDArray[np.uint8]:
    """The ids derived from ``ids`` by hashing every value with its id."""
    return np.array(
        [
            np.frombuffer(
                item_id(row.tobytes(), label, _value_bytes(value)), dtype=np.uint8
            )
            for row, value in zip(ids, values)
        ],
        dtype=np.uint8,
    ).reshape(len(ids), ID_BYTES)


def _value_bytes(value: Any) -> bytes:
    """A canonical encoding of a discriminator or salt."""
    array = np.asarray(value)
    match array.dtype.kind:
        case "U":
            return b"str" + str(array.item()).encode()
        case "S":
            return b"bytes" + bytes(array.item())
        case "O":
            item = array.item()
            return (
                b"bytes" + item
                if isinstance(item, bytes)
                else b"str" + str(item).encode()
            )
        case _:
            return array.dtype.str.encode() + np.ascontiguousarray(array).tobytes()


def _is_integer(values: NDArray[Any]) -> bool:
    return np.issubdtype(values.dtype, np.integer)


def _positions(selector: Selector, n: int) -> NDArray[np.int64]:
    """The absolute positions ``selector`` selects among ``n`` elements."""
    match selector:
        case bool() | np.bool_():
            raise TypeError(f"Cannot select elements with {selector!r}.")
        case int() | np.integer():
            index = int(selector)
            return np.array([index + n if index < 0 else index], dtype=np.int64)
        case slice():
            return np.arange(*selector.indices(max(n, _slice_stop(selector, n))))
        case np.ndarray() if selector.dtype == np.bool_:
            return np.flatnonzero(selector).astype(np.int64)
        case np.ndarray():
            return selector.astype(np.int64, copy=False)
        case _:
            raise TypeError(f"Cannot select elements with {selector!r}.")


def _slice_stop(selector: slice, n: int) -> int:
    return n if selector.stop is None else selector.stop


def _extent(selector: Selector, n: int) -> int:
    """The length a column needs to hold every element of ``selector``."""
    match selector:
        case slice():
            return max(n, _slice_stop(selector, n))
        case _:
            positions = _positions(selector, n)
            return max(n, int(positions.max()) + 1) if len(positions) else n


def _compact(positions: NDArray[np.int64]) -> Union[slice, NDArray[np.int64]]:
    """``positions`` as a slice if they are one ascending run, else as is."""
    if len(positions) and np.all(np.diff(positions) == 1):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    else:
        return positions


def _merged(segments: Sequence[IdSegment]) -> Tuple[IdSegment, ...]:
    """``segments`` without empty ones, adjacent runs of a kind joined, and
    short positional runs joined into mixed segments."""
    result: List[IdSegment] = []
    for segment in (s for s in segments if len(s)):
        joined = _joined(result[-1], segment) if result else None
        if joined is None:
            result.append(segment)
        else:
            result[-1] = joined
    return _compacted(result)


def _compacted(segments: Sequence[IdSegment]) -> Tuple[IdSegment, ...]:
    """``segments`` with every stretch of several short positional runs and
    mixed segments joined into one mixed segment."""
    result: List[IdSegment] = []
    stretch: List[IdSegment] = []
    for segment in [*segments, None]:
        if segment is not None and _is_fragment(segment):
            stretch.append(segment)
            continue
        if len(stretch) > 1:
            result.append(_mixed_of(stretch))
        else:
            result.extend(stretch)
        stretch = []
        if segment is not None:
            result.append(segment)
    return tuple(result)


def _is_fragment(segment: IdSegment) -> bool:
    return segment.mixed or (segment.namespace is not None and len(segment) < SHORT_RUN)


def _mixed_of(segments: Sequence[IdSegment]) -> IdSegment:
    """One mixed segment holding the positional and mixed ``segments``."""
    tables: List[NDArray[np.uint8]] = []
    indices: List[NDArray[np.int64]] = []
    offset = 0
    for segment in segments:
        if segment.namespaces is not None and segment.indices is not None:
            table = segment.namespaces
            local = segment.indices.astype(np.int64)
        else:
            assert segment.namespace is not None
            table = np.frombuffer(segment.namespace, dtype=np.uint8)[None]
            local = np.zeros(len(segment), dtype=np.int64)
        tables.append(table)
        indices.append(local + offset)
        offset += len(table)
    rows = np.ascontiguousarray(np.concatenate(tables)).view(f"V{ID_BYTES}").ravel()
    unique, inverse = np.unique(rows, return_inverse=True)
    namespaces, merged = _pruned(
        np.frombuffer(unique.tobytes(), dtype=np.uint8).reshape(-1, ID_BYTES),
        inverse.ravel()[np.concatenate(indices)],
    )
    return IdSegment(
        np.concatenate([np.asarray(s.positions, dtype=np.int64) for s in segments]),
        namespaces=namespaces,
        indices=merged,
    )


def _pruned(
    namespaces: NDArray[np.uint8],
    indices: NDArray[np.integer],
) -> Tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """The namespaces ``indices`` use, and the indices into them.

    A view of a few elements of a mixed segment would otherwise keep the
    namespaces of all of its elements.
    """
    used, remapped = np.unique(indices, return_inverse=True)
    if len(used) == len(namespaces):
        return namespaces, np.asarray(indices, dtype=np.int32)
    return namespaces[used], remapped.ravel().astype(np.int32)


def _joined(first: IdSegment, second: IdSegment) -> Optional[IdSegment]:
    """One segment holding ``first`` then ``second``, if they are of a kind."""
    if first.explicit is not None and second.explicit is not None:
        explicit = np.concatenate([first.explicit, second.explicit])
        return IdSegment(range(len(explicit)), None, explicit)
    if not first.assigned and not second.assigned:
        return IdSegment(range(len(first) + len(second)))
    if (
        first.namespace is None
        or first.namespace != second.namespace
        or first.explicit is not None
        or second.explicit is not None
        or first.mixed
        or second.mixed
    ):
        return None
    a, b = first.positions, second.positions
    if (
        isinstance(a, range)
        and isinstance(b, range)
        and a.step == b.step == 1
        and a.stop == b.start
    ):
        return IdSegment(range(a.start, b.stop), first.namespace)
    return IdSegment(
        np.concatenate([np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)]),
        first.namespace,
    )
