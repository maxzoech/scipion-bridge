"""Offset value class for multi-dimensional storage indexing and slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple, Union, TypeAlias

from functools import cache

IndexType: TypeAlias = Union[slice, int]

def _add_bound(base: int | None, delta: int | None) -> int | None:
    if delta is None:
        return base
    if base is None:
        return delta
    if (base >= 0) != (delta >= 0):
        raise ValueError(f"Cannot compose mixed-sign bounds ({base}, {delta}) without sequence length.")
    return base + delta

class KeyPath(Sequence[Tuple[str, IndexType]]):
    """Represents an immutable navigation path to nested data structures or array-backed storage.

    A `KeyPath` models hierarchical field traversal combined with positional slicing
    or scalar indexing. It allows programmatic querying, projection, and manipulation
    of nested records, structs, tabular columns, or multidimensional arrays.

    Each path segment is stored as a `(name, index)` tuple where:
        - `name`: The name in the corresponding schema.
        - `index`: An `IndexType` indicating the selection across that dimension.
          This is typically initialized as an unbounded slice (`slice(None)`),
          which can subsequently be narrowed to a subslice or a concrete integer index.

    The path functions as an immutable sequence of components, supporting operations
    like chaining, relative slice composition, and offset projection without requiring
    eager resolution of the underlying container length when bounds are statically determinable.

    Attributes:
        components (tuple[tuple[str, IndexType], ...]): The underlying sequence
            of component identifier and index pairs.
        path (tuple[str, ...]): The ordered attribute/field names traversed by the path.
        indices (tuple[IndexType, ...]): The active slices or scalar indices for
            each step of the path.

    Examples:
        Constructing and narrowing an attribute path:

        >>> path = KeyPath().append("users").append("orders")
        >>> path.path
        ('root', 'users', 'orders')

        Narrowing ranges along the path:

        >>> narrowed = path.narrow_slice(slice(0, 10)).narrow_slice(slice(2, 5))
        >>> narrowed.indices[-1]
        slice(2, 5, 1)

        Pinning the tail component to a scalar record index:

        >>> scalar_item = narrowed.narrow_index(1)
        >>> scalar_item.indices[-1]
        3
    """

    def __init__(self, root: Sequence[Tuple[str, IndexType]] = (("root", slice(None)),)) -> None:
        super().__init__()

        self.components = root


    @property
    def path(self) -> Tuple[str, ...]:
        return tuple([n for (n, _) in self.components])

    @property
    def indices(self) -> Tuple[IndexType, ...]:
        return tuple([i for (_, i) in self.components])


    def append(self, name: str) -> KeyPath:
        """Appends an attribute or field name to the key path.

        Extends the path to target a child attribute on an object, a struct field,
        or a set column. The new component is initialized with an unbounded
        slice (`slice(None)`), representing the entire range or collection until 
        further narrowed.

        Args:
            name: The name of the attribute, field, or column to append.

        Returns:
            KeyPath: A new `KeyPath` instance containing the existing path
                components followed by `(name, slice(None))`.

        Examples:
            >>> path = KeyPath().append("user")
            >>> path.components[-1]
            ('user', slice(None, None, None))

            >>> nested = path.append("address").append("city")
            >>> nested.path
            ('root', 'user', 'address', 'city')
        """
        return KeyPath([*self.components, (name, slice(None))])


    def narrow_index(self, index: int) -> KeyPath:
        """Narrows the terminal component's slice down to a concrete integer index.

        Composes a relative scalar index with the slice currently held at the tail
        of the key path, projecting the sub-index into the parent's coordinate frame.

        Resolution rules:
            1. **Identity Slice (`[:]`)**: The slice is directly replaced by `index`.
            2. **Known Fixed Span (`[start:stop]`)**: When both `start` and `stop` are
            non-negative, the window size is constant. Both positive indices
            (`0 <= index < span`) and negative indices (`-span <= index < 0`) are
            fully validated and projected into absolute coordinates.
            3. **Open Right Bound (`[start:]`)**: Positive indices are resolved as
            `start + index`. Negative indices cannot be validated without knowing
            the underlying sequence length.
            4. **Open Left Bound with Negative Stop (`[:-k]`)**: Negative indices are
            resolved further backwards from the terminal offset as `stop + index`.
            Positive indices cannot be validated without knowing sequence length.

        Args:
            index: The relative integer index to select from the existing slice.
                Accepts both positive and negative values where determinable.

        Returns:
            KeyPath: A new `KeyPath` instance whose terminal component has been
                narrowed to an integer index.

        Raises:
            IndexError: If the target slice has a known span and `index` falls
                outside `[-span, span - 1]`.
            ValueError: In any of the following conditions:
                - The final component is already an integer index (cannot be narrowed).
                - The final component cannot be indexed.
                - The combination of slice anchors and `index` sign cannot be
                resolved statically without the concrete sequence length
                (e.g., negative index on an open-ended slice).
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("items")  # items[:]
            >>> path.narrow_index(2).indices[-1]
            2

            >>> path = KeyPath().append("items").narrow_slice(slice(5, 10))  # items[5:10]
            >>> path.narrow_index(1).indices[-1]
            6
            >>> path.narrow_index(-1).indices[-1]
            9

            >>> open_path = KeyPath().append("items").narrow_slice(slice(3, None))  # items[3:]
            >>> open_path.narrow_index(4).indices[-1]
            7
            >>> open_path.narrow_index(-1)
            ValueError: Cannot resolve index -1 on slice slice(3, None, None) without knowing sequence length.
        """
        
        index = int(index)

        match self.components:
            case (*stem, (name, parent_index)) if parent_index == slice(None, None, None):
                return KeyPath([*stem, (name, index)])
            case (*stem, (name, slice() as parent_index)):
                if (parent_index.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_index.start or 0
                p_stop = parent_index.stop

                # If the span is known, we can compute the offset directly for
                # positive and negative indices
                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                    offset = span - index if index < 0 else index
                    
                    if not (0 <= offset < span):
                        raise IndexError(f"Index {index} out of bounds for span {span}.")
                    
                    return KeyPath([*stem, (name, p_start + offset)])

                # Left-anchored start with open stop: positive index only
                if p_start >= 0 and p_stop is None and index >= 0:
                    return KeyPath([*stem, (name, p_start + index)])

                # Right-anchored stop: negative index relative to right edge
                # e.g., parent[:-2] with index=-1 => -2 + (-1) = -3
                if p_stop is not None and p_stop < 0 and index < 0 and parent_index.start is None:
                    return KeyPath([*stem, (name, p_stop + index)])

                raise ValueError(
                    f"Cannot resolve index {index} on slice {parent_index} without knowing sequence length."
                )
                
            case _:
                raise ValueError


    def narrow_slice(self, index: slice) -> KeyPath:
        """Narrows the terminal component's slice by composing it with a subslice.

        Projects a relative `slice` into the coordinate frame of the parent slice 
        currently held at the tail of the key path. Only slices with a step of 1 
        are supported.

        Resolution rules:
            1. **Identity Slice (`[:]`)**: If the parent component holds an unbounded
            identity slice (`slice(None, None, None)`), it is directly replaced 
            by `index`.
            2. **Known Fixed Span (`[start:stop]`)**: When both `start` and `stop` 
            are non-negative, the window size is constant. Python's native 
            `slice.indices()` handles bounds clamping and resolves both positive 
            and negative child bounds into absolute coordinates.
            3. **Open-ended or Right-anchored Fallback**:
            - **Start**: Composed using `_add_bound(parent.start, index.start)`. 
                Requires compatible anchor signs (both positive or both negative).
            - **Stop**:
                - If `index.stop` is `None`, inherits `parent.stop`.
                - If `index.stop >= 0`, offset is computed relative to `parent.start`.
                - If `index.stop < 0`, offset is computed relative to `parent.stop`.

        Args:
            index: The relative subslice to compose with the existing slice.
                Must have `step=1` (or `None`).

        Returns:
            KeyPath: A new `KeyPath` instance with the narrowed terminal slice.

        Raises:
            ValueError: In any of the following conditions:
                - The incoming `index.step` is not 1 (or `None`).
                - The final component cannot be sliced (e.g. already a scalar index).
                - The composition involves mixed-sign bounds (e.g. left-anchored 
                offset combined with right-anchored base) that cannot be resolved 
                without knowing the sequence length.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("items")  # items[:]
            >>> path.narrow_slice(slice(2, 8)).indices[-1]
            slice(2, 8, 1)

            >>> path = KeyPath().append("items").narrow_slice(slice(2, 10))  # items[2:10]
            >>> path.narrow_slice(slice(1, 4)).indices[-1]
            slice(3, 6, 1)
            >>> path.narrow_slice(slice(-4, -1)).indices[-1]
            slice(6, 9, 1)

            >>> open_path = KeyPath().append("items").narrow_slice(slice(5, None))  # items[5:]
            >>> open_path.narrow_slice(slice(2, 6)).indices[-1]
            slice(7, 11, 1)
            >>> open_path.narrow_slice(slice(-2, None))
            ValueError: Cannot compose mixed-sign bounds (5, -2) without sequence length.
        """

        if (index.step or 1) != 1:
            raise ValueError("Only step=1 is supported.")

        match self.components:
            case (*stem, (name, idx)) if idx == slice(None, None, None):
                return KeyPath([*stem, (name, index)])
            case (*stem, (name, slice() as parent)):
                if (parent.step or 1) != 1:
                    raise NotImplementedError("Composing over parent slice with step != 1 is not supported.")

                p_start = parent.start or 0

                if parent.stop is not None and parent.stop >= 0 and p_start >= 0:
                    span = max(0, parent.stop - p_start)
                    rel_start, rel_stop, _ = index.indices(span)
                    new_slice = slice(p_start + rel_start, p_start + rel_stop, 1)

                    return KeyPath([*stem, (name, new_slice)])

                # Fallback: parent span is open-ended or right-anchored
                new_start = _add_bound(parent.start, index.start)

                if index.stop is None:
                    new_stop = parent.stop
                elif index.stop >= 0:
                    new_stop = _add_bound(parent.start, index.stop)
                else:
                    new_stop = _add_bound(parent.stop, index.stop)

                return KeyPath([*stem, (name, slice(new_start, new_stop, 1))])
                
            case _:
                raise ValueError
    

    def __getitem__(self, index):
        return self.components[index]

    def __len__(self) -> int:
        return len(self.components)
