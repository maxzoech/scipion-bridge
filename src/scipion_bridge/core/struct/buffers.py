"""Stateless helpers that build, reshape and fill the column buffers of storages.

A column is a NumPy array whenever all of its rows are present and have the
same shape. Only a column whose rows differ in shape, or which has missing
rows, is an Awkward Array. This holds for static fields (whose element shape is
known from the schema) and for dynamic fields (with ``None`` dimensions) alike;
a dynamic field only *allows* rows of different shapes. While a dynamic column
is written element by element, it is kept as a list of per-row buffers ("rows")
and stacked into a single column when it is read as a whole.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union, overload

import awkward as ak
import numpy as np

from .key_path import IndexType, KeyPath
from .schema import Entry, SetEntryBase
from .utils.arrow_utils import is_regular_awkward

Column = Union[np.ndarray, ak.Array]


def as_regular_numpy(data: Any, dtype: Any) -> Optional[np.ndarray]:
    """``data`` as a NumPy array of ``dtype``, or None if it has no regular shape.

    Ragged nested sequences and values that are not numeric (strings aside)
    cannot be represented by a regular NumPy array; they are stored in Awkward
    Arrays instead.
    """
    try:
        arr = np.asarray(data, dtype=dtype)
    except (ValueError, TypeError):
        return None

    return None if arr.dtype == object else arr


def check_castable(source: np.dtype, target: Any, key: KeyPath) -> None:
    """Raise if values of dtype ``source`` cannot be stored in a field of dtype ``target``."""
    if source == object or np.can_cast(source, target, casting="same_kind"):
        return

    raise TypeError(
        f"Cannot cast data of dtype '{source}' to field '{key}' dtype '{target}'.",
    )


@overload
def pad_to_capacity(
    column: np.ndarray,
    capacity: Optional[int],
    dtype: Any,
) -> np.ndarray: ...


@overload
def pad_to_capacity(
    column: ak.Array,
    capacity: Optional[int],
    dtype: Any,
) -> ak.Array: ...


def pad_to_capacity(column: Column, capacity: Optional[int], dtype: Any) -> Column:
    """Pad ``column`` along axis 0 to ``capacity`` rows.

    NumPy columns are padded with zeros, Awkward columns with missing rows.
    """
    if capacity is None or capacity <= len(column):
        return column

    match column:
        case np.ndarray():
            padded = np.zeros((capacity, *column.shape[1:]), dtype=dtype)
            padded[: len(column)] = column
            return padded
        case _:
            return ak.pad_none(column, capacity, axis=0)


def has_numpy_layout(column: ak.Array) -> bool:
    """True if ``column`` is a NumPy array behind regular dimensions only.

    Such an array converts to NumPy without copying or inspecting the data.
    """
    return _is_numpy_layout(column.layout)


def _is_numpy_layout(layout: ak.contents.Content) -> bool:
    match layout:
        case ak.contents.NumpyArray():
            return True
        case ak.contents.RegularArray():
            return _is_numpy_layout(layout.content)
        case _:
            return False


def regularized(column: ak.Array) -> Column:
    """``column`` as a NumPy array if all its rows are present and of one shape.

    Used where a column is built from parts (rows, sequences); a column read
    back from storage is never checked again.
    """
    match column:
        case _ if has_numpy_layout(column):
            return ak.to_numpy(column)
        case _ if _is_complete(column) and is_regular_awkward(column):
            return ak.to_numpy(column)
        case _:
            return column


def _is_complete(column: ak.Array) -> bool:
    """True if no value of ``column`` is missing, at any depth."""
    return not any(ak.any(ak.is_none(column, axis=axis)) for axis in range(column.ndim))


def concat_columns(columns: Sequence[Column]) -> Column:
    """Concatenate columns along axis 0.

    NumPy columns whose rows have one shape are joined with NumPy; any other
    mix (rows of different shapes, Awkward columns) gives an Awkward Array.
    Empty columns do not take part, so they never force the switch to Awkward.
    """
    filled = [column for column in columns if len(column) > 0] or list(columns[:1])
    match filled:
        case [np.ndarray() as first, *rest] if all(
            isinstance(column, np.ndarray) and column.shape[1:] == first.shape[1:]
            for column in rest
        ):
            return np.concatenate(filled, axis=0)
        case _:
            return ak.concatenate(filled, axis=0)


def stack_rows(rows: List[Any]) -> Column:
    """Stack per-row buffers into a single column; None rows become missing rows.

    Rows of one shape give a NumPy array; rows of different shapes, or missing
    rows, give an Awkward Array.
    """
    present = [row for row in rows if row is not None]

    match present:
        case []:
            return ak.Array([None] * len(rows))

        case [first, *_] if not isinstance(first, (np.ndarray, ak.Array, Sequence)):
            return regularized(ak.Array(rows))

        case _ if len(present) == len(rows) and _have_equal_shapes(present):
            return np.stack(rows, axis=0)

        case _ if len(present) == len(rows):
            return regularized(_stack_ragged(present, rows))

        case _:
            return _stack_ragged(present, rows)


def _have_equal_shapes(rows: List[Any]) -> bool:
    """True if all rows are NumPy arrays of the same shape."""
    return all(isinstance(row, np.ndarray) for row in rows) and all(
        row.shape == rows[0].shape for row in rows
    )


def _stack_ragged(present: List[Any], rows: List[Any]) -> ak.Array:
    """Concatenate rows of different lengths into an Awkward Array."""
    flat = ak.concatenate(present, axis=0)
    lengths = [0 if row is None else len(row) for row in rows]
    unflat = ak.unflatten(flat, lengths, axis=0)

    if len(present) == len(rows):
        return unflat

    return ak.mask(unflat, [row is not None for row in rows])


def stack_sequence(rows: Sequence[Column]) -> Optional[Column]:
    """Stack a sequence of array rows, or None if some row is not an array.

    Rows of one shape give a NumPy array, otherwise an Awkward Array.
    """
    match rows:
        case _ if _have_equal_shapes(list(rows)):
            return np.stack(rows, axis=0)
        case _ if all(isinstance(row, (np.ndarray, ak.Array)) for row in rows):
            return regularized(
                ak.unflatten(
                    ak.concatenate(rows, axis=0),
                    [len(row) for row in rows],
                    axis=0,
                ),
            )
        case _:
            return None


def split_rows(column: Column) -> List[Any]:
    """Split a column into a list of per-row buffers.

    Regular columns become writable NumPy row views (one copy at most, if the
    column is read-only); irregular or masked columns become Awkward rows,
    keeping missing rows as None.
    """
    match column:
        case np.ndarray():
            rows = column if column.flags.writeable else column.copy()
            return list(rows)
        case ak.Array() if not ak.any(
            ak.is_none(column, axis=0),
        ) and is_regular_awkward(column):
            rows = ak.to_numpy(column)
            return list(rows if rows.flags.writeable else rows.copy())
        case ak.Array():
            return [row for row in column]
        case _:
            raise TypeError(
                f"Cannot split column of type '{type(column).__name__}' into rows.",
            )


def coerce_static_shape(
    arr: np.ndarray,
    entry_shape: Tuple[Optional[int], ...],
    key: KeyPath,
    *,
    is_set: bool = False,
) -> np.ndarray:
    """Validate and adjust the shape of ``arr`` to the element shape of a static field."""
    match (arr.shape, entry_shape, is_set):
        case ((), (1,), False):
            return arr.reshape(1)

        case ((), (1,), True):
            return arr.reshape(1, 1)

        case ((), (), _):
            return arr.reshape(())

        case ((_,), (1,), True):
            return arr.reshape(-1, 1)

        case ((1,), (1,), False):
            return arr

        case (shape, target, _) if (
            len(shape) >= len(target) and shape[len(shape) - len(target) :] == target
        ):
            return arr

        case _:
            raise ValueError(
                f"Shape mismatch for static field '{key}': expected {entry_shape} for element.",
            )


def broadcastable(target: Tuple[int, ...], source: Tuple[int, ...]) -> bool:
    """True if arrays of the two shapes broadcast against each other."""
    try:
        np.broadcast_shapes(target, source)
    except ValueError:
        return False
    else:
        return True


def expanded_shape(
    shape: Tuple[int, ...],
    effective_idx: Tuple[IndexType, ...],
) -> Tuple[int, ...]:
    """Return the shape needed to hold ``effective_idx`` (``shape`` if it fits)."""
    new_shape = list(shape)
    for i, idx in enumerate(effective_idx):
        match idx:
            case int(n) if n >= new_shape[i]:
                new_shape[i] = n + 1
            case slice() as s if s.stop is not None and s.stop > new_shape[i]:
                new_shape[i] = s.stop
            case _:
                pass
    return tuple(new_shape)


def expand_to_fit(
    buffer: np.ndarray,
    effective_idx: Tuple[IndexType, ...],
) -> np.ndarray:
    """Return ``buffer``, or a zero-padded copy large enough for ``effective_idx``."""
    new_shape = expanded_shape(buffer.shape, effective_idx)
    if new_shape == buffer.shape:
        return buffer

    new_buffer = np.zeros(new_shape, dtype=buffer.dtype)
    new_buffer[tuple(slice(0, s) for s in buffer.shape)] = buffer
    return new_buffer


def infer_outer_dims(
    entry: Entry,
    effective_idx: Tuple[IndexType, ...],
    min_length: int = 0,
) -> Tuple[int, ...]:
    """Outer dimensions of a new column that has to hold ``effective_idx``.

    The first dimension is at least the capacity of a Set field and
    ``min_length`` (the length of the sibling columns).
    """
    capacity = entry.capacity if isinstance(entry, SetEntryBase) else None
    floors = [max(capacity or 0, min_length)] + [0] * (len(effective_idx) - 1)
    return tuple(
        max(floor, _index_extent(idx)) for floor, idx in zip(floors, effective_idx)
    )


def _index_extent(idx: IndexType) -> int:
    """Minimal length of a dimension that ``idx`` indexes into."""
    match idx:
        case int(n):
            return n + 1
        case slice() as s if s.stop is not None:
            return s.stop
        case np.ndarray() as arr:
            return len(arr)
        case _:
            return 0


def nested_rows(dims: Sequence[int]) -> List[Any]:
    """Nested lists of None rows with the given outer dimensions."""
    match dims:
        case []:
            return []
        case [n]:
            return [None] * n
        case [n, *inner]:
            return [nested_rows(inner) for _ in range(n)]
        case _:
            raise TypeError(f"Invalid dimensions '{dims}'.")


def assign_rows(
    rows: List[Any], effective_idx: Tuple[IndexType, ...], data: Any
) -> None:
    """Write ``data`` into the nested row lists at ``effective_idx``.

    Slice and index-array writes require one data element per selected row;
    the row lists never change their length here.
    """
    match effective_idx:
        case (int() as i,):
            rows[i] = data

        case (int() as i, *rest):
            assign_rows(rows[i], tuple(rest), data)

        case (slice() | np.ndarray() as idx, *rest):
            positions = _positions(idx, len(rows))
            if len(positions) != len(data):
                raise ValueError(
                    f"Cannot assign {len(data)} elements to {len(positions)} rows.",
                )
            for position, element in zip(positions, data):
                _assign_row(rows, position, tuple(rest), element)

        case _:
            raise TypeError(
                f"Unsupported index '{effective_idx}' for a dynamic column.",
            )


def _positions(idx: Union[slice, np.ndarray], length: int) -> Sequence[int]:
    match idx:
        case slice():
            return range(*idx.indices(length))
        case _:
            return [int(i) for i in idx]


def _assign_row(
    rows: List[Any],
    position: int,
    rest: Tuple[IndexType, ...],
    element: Any,
) -> None:
    match rest:
        case ():
            rows[position] = element
        case _:
            assign_rows(rows[position], rest, element)
