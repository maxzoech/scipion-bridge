from typing import Any, Optional, Sequence, Tuple, Union, TypeAlias
import numpy as np
from numpy.typing import NDArray

IndexType: TypeAlias = Union[slice, int, NDArray[np.int64]]


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

    def __init__(
        self, root: Sequence[Tuple[str, IndexType]] = (("root", slice(None)),)
    ) -> None:
        """Initializes a KeyPath with an initial sequence of components.

        Args:
            root: Initial sequence of `(name, index)` component pairs.
                Defaults to `(('root', slice(None)),)`.
        """
        super().__init__()

        self.components = root

    @property
    def path(self) -> Tuple[str, ...]:
        """Returns the ordered tuple of component names traversed by the path."""
        return tuple([n for (n, _) in self.components])

    @property
    def indices(self) -> Tuple[IndexType, ...]:
        """Returns the tuple of active index/slice components along the path."""
        return tuple([i for (_, i) in self.components])

    def extend(self, path: "KeyPath") -> "KeyPath":
        """Concatenates another `KeyPath` onto this key path.

        Appends all components from `path` to the end of `self.components`,
        returning a new `KeyPath` instance.

        Args:
            path: The `KeyPath` whose components should be concatenated.

        Returns:
            KeyPath: A new `KeyPath` instance containing the combined components.

        Examples:
            >>> p1 = KeyPath().append("users")
            >>> p2 = KeyPath((("orders", slice(0, 5)),))
            >>> combined = p1.extend(p2)
            >>> combined.path
            ('root', 'users', 'orders')
            >>> combined.indices[-1]
            slice(0, 5, None)
        """
        return KeyPath([*self.components, *path.components])

    def append(self, name: str) -> "KeyPath":
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

    def narrow_index(self, index: int, length: Optional[int] = None) -> "KeyPath":
        """Narrows the terminal component's slice or index array to a concrete integer index.

        Composes a relative scalar index with the component currently held at the tail
        of the key path, projecting the sub-index into the parent's coordinate frame.

        Resolution rules:
            1. **Slice with Known Span or Sequence Length**: When the parent slice has
               non-negative bounds (`[start:stop]`), `span = max(0, stop - start)`.
               Alternatively, if `length` is provided, `span` is resolved from `length`.
               Bounds-checks `index` against `[-span, span - 1]` (`IndexError`), normalizes
               negative indices (`offset = span + index if index < 0 else index`), and
               projects `start + offset`.
            2. **Open-Ended Slice (`[start:]`) without Length**: If `start >= 0`,
               `stop is None`, and `index >= 0`, projects to `start + index`. Negative
               indices cannot be resolved without sequence length and raise `ValueError`.
            3. **Right-Anchored Slice (`[:stop]`) without Length**: If `start is None`,
               `stop < 0`, and `index < 0`, projects relative to the negative right edge
               as `stop + index`. Positive indices cannot be resolved without sequence
               length and raise `ValueError`.
            4. **Existing Index Array (`NDArray`)**: When the terminal component is already
               an index array, performs a scalar take: bounds-checks `index` against
               `len(parent_arr)` (handling negative indices) and yields `int(parent_arr[offset])`.
               Raises `IndexError` if out of bounds.
            5. **Scalar Index (`int`)**: Cannot be narrowed further; raises `ValueError`.

        Args:
            index: The relative integer index to select from the existing component.
                Accepts both positive and negative values where determinable.
            length: Optional known sequence length of the active dimension, used
                for span resolution and bounds checking.

        Returns:
            KeyPath: A new `KeyPath` instance whose terminal component has been
                narrowed to an integer index.

        Raises:
            IndexError: If `index` falls outside valid bounds for a known span or length.
            ValueError: In any of the following conditions:
                - The terminal component cannot be indexed (e.g. already an integer index).
                - The combination of slice anchors and `index` sign cannot be
                  resolved statically without knowing the sequence length.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("users")  # root.users[:]
            >>> path.narrow_index(3).indices[-1]
            3

            >>> path.narrow_slice(slice(10, 20)).narrow_index(2).indices[-1]
            12

            >>> path.narrow_slice(slice(10, 20)).narrow_index(-1).indices[-1]
            19

            >>> path.narrow_indices([10, 20, 30]).narrow_index(1).indices[-1]
            20
        """
        index = int(index)

        match self.components:
            case (*stem, (name, slice() as parent_index)):
                if (parent_index.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_index.start or 0
                p_stop = parent_index.stop

                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                else:
                    span = length

                if span is not None:
                    offset = span + index if index < 0 else index
                    if not (0 <= offset < span):
                        raise IndexError(
                            f"Index {index} out of bounds for span {span}."
                        )
                    return KeyPath([*stem, (name, p_start + offset)])

                # Left-anchored start with open stop: positive index only
                if p_start >= 0 and p_stop is None and index >= 0:
                    return KeyPath([*stem, (name, p_start + index)])

                # Right-anchored stop: negative index relative to right edge
                # e.g., parent[:-2] with index=-1 => -2 + (-1) = -3
                if (
                    p_stop is not None
                    and p_stop < 0
                    and index < 0
                    and parent_index.start is None
                ):
                    return KeyPath([*stem, (name, p_stop + index)])

                raise ValueError(
                    f"Cannot resolve index {index} on slice {parent_index} without knowing sequence length."
                )

            case (*stem, (name, np.ndarray() as parent_arr)):
                span = len(parent_arr)
                offset = span + index if index < 0 else index
                if not (0 <= offset < span):
                    raise IndexError(f"Index {index} out of bounds for span {span}.")
                return KeyPath([*stem, (name, int(parent_arr[offset]))])

            case _:
                raise ValueError(
                    f"Cannot narrow terminal component with index {index}."
                )

    def narrow_slice(self, index: slice, length: Optional[int] = None) -> "KeyPath":
        """Narrows the terminal component by composing it with a subslice.

        Projects a relative `slice` into the coordinate frame of the component
        currently held at the tail of the key path. Only slices with a step of 1
        are supported.

        Resolution rules:
            1. **Slice with Known Span or Sequence Length**: When the parent slice has
               non-negative bounds (`[start:stop]`), `span = max(0, stop - start)`.
               Alternatively, if `length` is provided, `span` is resolved from `length`.
               Composes the subslice via `index.indices(span)` and produces a canonical
               slice `slice(start + rel_start, start + rel_stop, 1)`. Clamps and normalizes
               negative or out-of-bounds indices automatically.
            2. **Unbounded / Symbolic Composition (without Length)**:
               - The new start is composed algebraically via `_add_bound(parent.start, index.start)`.
               - If `index.stop is None`, the new stop remains `parent.stop`.
               - If `index.stop >= 0`, the new stop is anchored to start: `_add_bound(parent.start, index.stop)`.
               - If `index.stop < 0`, the new stop is anchored to stop: `_add_bound(parent.stop, index.stop)`.
               - Mixed-sign composition (e.g. adding a negative offset to a non-negative anchor)
                 raises `ValueError` when sequence length is unknown.
            3. **Existing Index Array (`NDArray`)**: Slices the index array in-place
               via `parent_arr[index]`.
            4. **Scalar Index (`int`)**: Cannot be sliced; raises `ValueError`.

        Args:
            index: The relative subslice to compose with the existing slice or array.
                Must have `step=1` (or `None`).
            length: Optional known sequence length of the active dimension, used
                for span resolution and concrete slice computation.

        Returns:
            KeyPath: A new `KeyPath` instance with the narrowed terminal component.

        Raises:
            ValueError: If `index.step != 1`, terminal component cannot be sliced,
                or mixed-sign composition cannot be resolved without sequence length.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("items")  # items[:]
            >>> path.narrow_slice(slice(2, 8)).indices[-1]
            slice(2, 8, 1)

            >>> path.narrow_slice(slice(2, 8)).narrow_slice(slice(1, 4)).indices[-1]
            slice(3, 6, 1)

            >>> path.narrow_slice(slice(5, None)).narrow_slice(slice(2, None)).indices[-1]
            slice(7, None, 1)

            >>> path.narrow_indices([10, 20, 30, 40, 50]).narrow_slice(slice(1, 4)).indices[-1]
            array([20, 30, 40])
        """
        if (index.step or 1) != 1:
            raise ValueError("Only step=1 is supported.")

        match self.components:
            case (*stem, (name, slice() as parent)):
                if (parent.step or 1) != 1:
                    raise NotImplementedError(
                        "Composing over parent slice with step != 1 is not supported."
                    )

                p_start = parent.start or 0
                p_stop = parent.stop

                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                else:
                    span = length

                if span is not None:
                    rel_start, rel_stop, _ = index.indices(span)
                    return KeyPath(
                        [
                            *stem,
                            (name, slice(p_start + rel_start, p_start + rel_stop, 1)),
                        ]
                    )

                # Fallback: parent span is open-ended or right-anchored
                new_start = _add_bound(parent.start, index.start)

                if index.stop is None:
                    new_stop = parent.stop
                elif index.stop >= 0:
                    new_stop = _add_bound(parent.start, index.stop)
                else:
                    new_stop = _add_bound(parent.stop, index.stop)

                return KeyPath([*stem, (name, slice(new_start, new_stop, 1))])

            case (*stem, (name, np.ndarray() as parent_arr)):
                return KeyPath([*stem, (name, parent_arr[index])])

            case _:
                raise ValueError(f"Cannot slice terminal component with {index}.")

    def narrow_indices(
        self,
        indices: Union[Sequence[int], NDArray[np.integer]],
        length: Optional[int] = None,
    ) -> "KeyPath":
        """Narrows the terminal component to an array of integer indices (gather/take).

        Composes relative integer indices with the component currently held at the tail
        of the key path, projecting sub-indices into the parent coordinate frame.

        Resolution rules:
            1. **Slice with Known Span or Sequence Length**: When the parent slice has
               non-negative bounds (`[start:stop]`), `span = max(0, stop - start)`.
               Alternatively, if `length` is provided, `span` is resolved from `length`.
               Checks all indices against `[-span, span - 1]` (`IndexError`), normalizes
               negative indices (`span + arr`), and projects `start + offset`.
            2. **Open-Ended Slice (`[start:]`) without Length**: When `start >= 0` and
               `stop is None`, non-negative indices are projected as `start + arr`. Any
               negative index raises `ValueError` because sequence length is required to
               compute negative offsets.
            3. **Slice with Negative Bounds without Length**: Cannot be resolved without
               sequence length; raises `ValueError`.
            4. **Existing Index Array (`NDArray`)**: Vectorized take/gather operation. Checks
               all indices against `[-len(parent_arr), len(parent_arr) - 1]` (`IndexError`),
               normalizes negative indices, and gathers from the parent array via
               `parent_arr[offset]`.
            5. **Scalar Index (`int`)**: Cannot be indexed into; raises `ValueError`.

        Args:
            indices: Sequence or 1D integer array of relative indices to select.
            length: Optional known sequence length of the active dimension, used
                for span resolution and bounds checking.

        Returns:
            KeyPath: A new `KeyPath` instance whose terminal component has been
                narrowed to a 1D `np.int64` array of absolute indices.

        Raises:
            IndexError: If any index falls outside valid bounds for a known span or length.
            ValueError: In any of the following conditions:
                - The terminal component cannot be indexed (e.g. already a scalar index).
                - A negative index is provided for an open-ended slice when sequence
                  length is unknown.
                - Slicing with negative bounds when sequence length is unknown.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("values")  # values[:]
            >>> path.narrow_indices([0, 2, 4], length=5).indices[-1]
            array([0, 2, 4])

            >>> path.narrow_slice(slice(10, 20)).narrow_indices([0, -1, 2]).indices[-1]
            array([10, 19, 12])

            >>> path.narrow_indices([10, 20, 30, 40]).narrow_indices([3, 1]).indices[-1]
            array([40, 20])
        """
        arr = np.asarray(indices, dtype=np.int64)

        match self.components:
            case (*stem, (name, slice() as parent_slice)):
                if (parent_slice.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_slice.start or 0
                p_stop = parent_slice.stop

                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                else:
                    span = length

                if span is not None:
                    if np.any(arr < -span) or np.any(arr >= span):
                        raise IndexError(f"Index out of bounds for span {span}.")

                    offset = np.where(arr < 0, span + arr, arr)
                    return KeyPath([*stem, (name, p_start + offset)])

                if p_start >= 0 and p_stop is None:
                    if np.any(arr < 0):
                        raise ValueError(
                            f"Cannot resolve negative index on slice {parent_slice} without sequence length."
                        )
                    return KeyPath([*stem, (name, p_start + arr)])

                raise ValueError(
                    f"Cannot narrow slice {parent_slice} with indices without knowing sequence length."
                )

            case (*stem, (name, np.ndarray() as parent_arr)):
                span = len(parent_arr)
                if np.any(arr < -span) or np.any(arr >= span):
                    raise IndexError(f"Index out of bounds for span {span}.")
                offset = np.where(arr < 0, span + arr, arr)
                return KeyPath([*stem, (name, parent_arr[offset])])

            case _:
                raise ValueError("Cannot index into the terminal component.")

    def narrow_mask(
        self,
        mask: Union[Sequence[bool], NDArray[np.bool_]],
        length: Optional[int] = None,
    ) -> "KeyPath":
        """Narrows the terminal component using a boolean mask (filtering).

        Converts the boolean mask to active integer indices via `np.flatnonzero`
        and projects them into the parent coordinate frame.

        Resolution rules:
            1. **Identity Slice (`[:]`)**: If `length` is provided, validates
               `len(mask) == length` (raising `IndexError` on mismatch). Returns
               `np.flatnonzero(mask)`. If `length is None`, projects without length validation.
            2. **Known Fixed Span (`[start:stop]`)**: When both `start` and `stop` are
               non-negative, `span = max(0, stop - start)`. Validates `len(mask) == span`
               (`IndexError`) and projects `start + np.flatnonzero(mask)`.
            3. **Slice with Known Sequence Length**: When `length` is provided for an
               open-ended (`[start:]`) or negative-bound slice (e.g. `[:-2]`, `[-5:]`),
               `span` is resolved to `length`. Validates `len(mask) == span` (`IndexError`)
               and projects `start + np.flatnonzero(mask)`.
            4. **Open Right Bound (`[start:]`) without Length**: If `start >= 0` and
               `stop is None`, projects `start + np.flatnonzero(mask)` without length validation.
            5. **Negative Bounds without Length**: Raises `ValueError` because the span
               cannot be determined.
            6. **Existing Index Array (`NDArray`)**: Validates `len(mask) == len(parent_arr)`
               (`IndexError`) and filters the array in-place via `parent_arr[bool_mask]`.
            7. **Scalar Index (`int`)**: Cannot be masked; raises `ValueError`.

        Args:
            mask: Sequence or 1D boolean array indicating elements to retain.
            length: Optional known sequence length of the active dimension, used
                for span resolution and length validation.

        Returns:
            KeyPath: A new `KeyPath` instance whose terminal component has been
                narrowed to a 1D `np.int64` array of absolute indices.

        Raises:
            IndexError: If `len(mask)` does not match the known span or container
                length of the target component.
            ValueError: In any of the following conditions:
                - The terminal component cannot be masked (e.g. a scalar index).
                - A mask is applied to a slice with negative bounds when sequence
                  length is unknown.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("items")
            >>> path.narrow_mask([True, False, True]).indices[-1]
            array([0, 2])

            >>> slice_path = path.narrow_slice(slice(2, 7))  # items[2:7], span=5
            >>> slice_path.narrow_mask([True, False, False, True, False]).indices[-1]
            array([2, 5])

            >>> array_path = path.narrow_indices([10, 20, 30])
            >>> array_path.narrow_mask([True, False, True]).indices[-1]
            array([10, 30])
        """
        bool_mask = np.asarray(mask, dtype=bool)
        mask_len = len(bool_mask)

        match self.components:
            case (*stem, (name, slice() as parent_index)) if parent_index == slice(
                None, None, None
            ):
                if length is not None and mask_len != length:
                    raise IndexError(
                        f"Boolean mask length {mask_len} does not match container length {length}."
                    )

                return KeyPath([*stem, (name, np.flatnonzero(bool_mask))])

            case (*stem, (name, slice() as parent_slice)):
                if (parent_slice.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_slice.start or 0
                p_stop = parent_slice.stop

                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                else:
                    span = length

                if span is not None:
                    if mask_len != span:
                        raise IndexError(
                            f"Boolean mask length {mask_len} does not match span {span}."
                        )

                    return KeyPath([*stem, (name, p_start + np.flatnonzero(bool_mask))])

                if p_start >= 0 and p_stop is None:
                    return KeyPath([*stem, (name, p_start + np.flatnonzero(bool_mask))])

                raise ValueError(
                    f"Cannot apply boolean mask to slice {parent_slice} with negative bounds without knowing sequence length."
                )

            case (*stem, (name, np.ndarray() as parent_arr)):
                span = len(parent_arr)
                if mask_len != span:
                    raise IndexError(
                        f"Boolean mask length {mask_len} does not match span {span}."
                    )

                return KeyPath([*stem, (name, parent_arr[bool_mask])])

            case _:
                raise ValueError("Cannot apply boolean mask to the terminal component.")

    def __getitem__(self, index: Any) -> Any:
        """Returns the component or slice of components at the given index."""
        return tuple(self.components[index])

    def __len__(self) -> int:
        """Returns the number of components in the key path."""
        return len(self.components)

    def __str__(self) -> str:
        """Returns the string representation of the key path (e.g. 'root[:].users[0]')."""
        return ".".join(f"{name}{_format_index(idx)}" for name, idx in self.components)

    def __repr__(self) -> str:
        """Returns the formal representation of the KeyPath instance."""
        return f"KeyPath('{self}')"

    def __eq__(self, other: Any) -> bool:
        """Checks equality with another KeyPath or sequence of components.

        Handles array index components using `np.array_equal`.
        """
        # Combine early exit type and length checks
        if not isinstance(other, KeyPath) or len(self.components) != len(
            other.components
        ):
            return False

        for (n1, i1), (n2, i2) in zip(self.components, other.components):
            if n1 != n2:
                return False

            match i1, i2:
                # Both are arrays
                case np.ndarray(), np.ndarray():
                    if not np.array_equal(i1, i2):
                        return False

                # Only one is an array (mismatched types)
                case (np.ndarray(), _) | (_, np.ndarray()):
                    return False

                # Neither are arrays, rely on standard equality
                case _ if i1 != i2:
                    return False

        return True

    def __hash__(self) -> int:
        """Computes the hash of the key path.

        A `KeyPath` is hashable as long as none of its components contain an
        `NDArray` index.

        Returns:
            int: The hash value of the key path components.

        Raises:
            TypeError: If any component contains a NumPy array index, since NumPy
                arrays are mutable and unhashable.
        """
        hashed_components = []

        for name, idx in self.components:
            match idx:
                case np.ndarray():
                    raise TypeError(
                        "unhashable type: 'KeyPath' containing numpy array index"
                    )

                # Unpack slice attributes directly in the pattern match
                case slice(start=start, stop=stop, step=step):
                    hashed_components.append((name, (start, stop, step)))

                case _:
                    hashed_components.append((name, idx))

        return hash(tuple(hashed_components))


def _add_bound(base: int | None, delta: int | None) -> int | None:
    """Adds a relative index bound delta to an existing base bound.

    Used for algebraic slice composition when sequence length is unknown.
    Both bounds must share the same sign (both non-negative or both negative).

    Args:
        base: The base bound (start or stop) from the parent slice.
        delta: The relative bound offset to add.

    Returns:
        The composed bound, or None if both bounds are None.

    Raises:
        ValueError: If base and delta have mixed signs and sequence length is unknown.
    """
    if delta is None:
        return base

    if base is None:
        return delta

    if (base >= 0) != (delta >= 0):
        raise ValueError(
            f"Cannot compose mixed-sign bounds ({base}, {delta}) without sequence length."
        )

    return base + delta


def _format_index(index: IndexType) -> str:
    """Formats an index component (slice, integer, or array) for string display."""
    match index:
        case slice(start=start, stop=stop, step=step):
            start_str = "" if start is None else str(start)
            stop_str = "" if stop is None else str(stop)
            step_str = f":{step}" if step not in (None, 1) else ""
            return f"[{start_str}:{stop_str}{step_str}]"

        case np.ndarray():
            return f"[{index.tolist()}]"

        case _:
            return f"[{index}]"
