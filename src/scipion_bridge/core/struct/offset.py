"""Offset value class for multi-dimensional storage indexing and slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple, Union, TypeAlias

IndexType: TypeAlias = Union[slice, int]


@dataclass(frozen=True, slots=True)
class Offset(Sequence[IndexType]):
    """Immutable multi-dimensional index coordinates in storage.

    Each element represents one dimension:
      - int   -> element indexing (single element selected along that dimension)
      - slice -> range slicing (sub-range or unbounded selection)

    An empty tuple () represents the entire unbounded storage.
    """

    dims: Tuple[IndexType, ...] = ()

    def __init__(self, dims: Union[Offset, Tuple[IndexType, ...], Sequence[IndexType]] = ()) -> None:
        if isinstance(dims, Offset):
            object.__setattr__(self, "dims", dims.dims)
        else:
            object.__setattr__(self, "dims", tuple(dims))

    @classmethod
    def empty(cls) -> Offset:
        """Create an empty, unbounded Offset."""
        return cls(())

    @classmethod
    def from_index(cls, idx: int) -> Offset:
        """Create an Offset indexing a single item on dimension 0."""
        return cls((idx,))

    @classmethod
    def from_slice(cls, start: Optional[int] = None, stop: Optional[int] = None) -> Offset:
        """Create an Offset slicing a range on dimension 0."""
        return cls((slice(start, stop),))

    # ── Queries ───────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """True if no dimensions are indexed."""
        return len(self.dims) == 0

    @property
    def is_unbounded(self) -> bool:
        """True if empty or all dimensions are unbounded slices (e.g. slice(None))."""
        if not self.dims:
            return True
        return all(
            isinstance(d, slice) and d.start is None and d.stop is None and d.step is None
            for d in self.dims
        )

    @property
    def is_element_index(self) -> bool:
        """True if all dimensions are integer indices (selecting a single leaf element)."""
        return len(self.dims) > 0 and all(isinstance(d, int) for d in self.dims)

    def required_len(self, dim_idx: int) -> Optional[int]:
        """Return the minimum required capacity (idx + 1 or slice stop) along dim_idx, or None."""
        if dim_idx >= len(self.dims):
            return None
        match self.dims[dim_idx]:
            case int(idx):
                return idx + 1
            case slice(stop=int(stop)):
                return stop
            case _:
                return None

    @property
    def first(self) -> IndexType:
        """The outermost dimension index."""
        if not self.dims:
            raise IndexError("Empty offset has no first dimension.")
        return self.dims[0]

    @property
    def tail(self) -> Offset:
        """All dimensions except the outermost."""
        return Offset(self.dims[1:])

    # ── Operations ────────────────────────────────────────────

    def push_index(self, idx: int) -> Offset:
        """Select an integer index along the current active dimension."""
        if self.is_empty:
            return Offset((idx,))

        *prefix, last = self.dims
        match last:
            case slice() as base:
                base_start = base.start or 0
                return Offset((*prefix, base_start + idx))
            case _:
                raise ValueError(
                    f"Cannot index dimension '{last}' that is already an indexed integer."
                )

    def push_slice(self, start: int, stop: int) -> Offset:
        """Narrow the active dimension to a sub-slice [start:stop]."""
        if self.is_empty:
            return Offset((slice(start, stop),))

        *prefix, last = self.dims
        match last:
            case slice() as base:
                base_start = base.start or 0
                return Offset((*prefix, slice(base_start + start, base_start + stop)))
            case _:
                raise ValueError(
                    f"Cannot slice dimension '{last}' that is already an indexed integer."
                )

    def descend(self) -> Offset:
        """Descend into a child Set dimension by appending an unbounded slice."""
        return Offset((*self.dims, slice(None, None, None)))

    # ── Sequence Protocol & Interop ───────────────────────────

    def to_tuple(self) -> Tuple[IndexType, ...]:
        """Return the underlying dimensions as a raw tuple."""
        return self.dims

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self) -> Iterator[IndexType]:
        return iter(self.dims)

    def __getitem__(self, item: Any) -> Any:
        res = self.dims[item]
        if isinstance(res, tuple):
            return Offset(res)
        return res

    def __bool__(self) -> bool:
        return len(self.dims) > 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Offset):
            return self.dims == other.dims
        if isinstance(other, tuple):
            return self.dims == other
        return False

    def __repr__(self) -> str:
        return f"Offset({self.dims!r})"

