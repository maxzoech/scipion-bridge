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

    def __init__(self, root: Sequence[Tuple[str, IndexType]] = (("root", slice(None)),)) -> None:
        super().__init__()

        self.components = root

    @property
    def path(self) -> Tuple[str, ...]:
        return tuple([n for (n, _) in self.components])

    @property
    def indices(self) -> Tuple[IndexType, ...]:
        return tuple([i for (_, i) in self.components])


    def extend(self, path: "KeyPath") -> "KeyPath":
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


    def narrow_index(self, index: int) -> "KeyPath":
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
            case (*stem, (name, slice() as parent_index)) if parent_index == slice(None, None, None):
                return KeyPath([*stem, (name, index)])
            case (*stem, (name, slice() as parent_index)):
                if (parent_index.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_index.start or 0
                p_stop = parent_index.stop

                # If the span is known, we can compute the offset directly for
                # positive and negative indices.
                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                    offset = span + index if index < 0 else index
                    
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

            case (*stem, (name, np.ndarray() as parent_arr)):
                span = len(parent_arr)
                offset = span + index if index < 0 else index
                
                if not (0 <= offset < span):
                    raise IndexError(f"Index {index} out of bounds for span {span}.")
                
                return KeyPath([*stem, (name, int(parent_arr[offset]))])

            case _:
                raise ValueError(f"Cannot narrow terminal component with index {index}.")


    def narrow_slice(self, index: slice) -> "KeyPath":
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
            case (*stem, (name, slice() as idx)) if idx == slice(None, None, None):
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
            1. **Identity Slice (`[:]`)**:
               - If `length` is given, negative indices are resolved (`length + idx`) and
                 bounds `[-length, length - 1]` are validated.
               - If `length` is None, only non-negative indices are supported; negative
                 indices raise `ValueError`.
            2. **Known Fixed Span (`[start:stop]`)**: When both `start` and `stop` are
               non-negative, `span = stop - start`. Indices are bounds-checked (`-span <= idx < span`),
               negative indices are resolved, and offsets are projected to absolute coordinates
               (`start + offset`).
            3. **Open Right Bound (`[start:]`)**: Positive indices are projected as `start + idx`.
               Negative indices raise `ValueError`.
            4. **Existing Index Array (`NDArray`)**: Composes relative indices over the existing
               array via vectorized take (`parent_arr[offset]`).
            5. **Scalar Index (`int`)**: Cannot be indexed; raises `ValueError`.

        Args:
            indices: Sequence or 1D integer array of relative indices to select.
                Supports positive and negative indices where sequence length is known.
            length: Optional known sequence length of the active dimension, used
                for resolving negative indices and bounds-checking on unbounded slices.

        Returns:
            KeyPath: A new `KeyPath` instance whose terminal component has been
                narrowed to a 1D `np.int64` array of absolute indices.

        Raises:
            IndexError: If any index falls outside valid bounds for a known span or length.
            ValueError: In any of the following conditions:
                - The terminal component cannot be indexed (e.g. already a scalar index).
                - A negative index is provided on an open-ended slice without length.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("items")  # items[:]
            >>> path.narrow_indices([0, 2], length=5).indices[-1]
            array([0, 2])

            >>> slice_path = path.narrow_slice(slice(3, 10))  # items[3:10], span=7
            >>> slice_path.narrow_indices([0, -1, 2]).indices[-1]
            array([3, 9, 5])
        """
        arr = np.asarray(indices, dtype=np.int64)

        match self.components:
            case (*stem, (name, slice() as parent_index)) if parent_index == slice(None, None, None):
                if length is not None:
                    if np.any(arr < -length) or np.any(arr >= length):
                        raise IndexError(f"Index out of bounds for length {length}.")
                    
                    offset = np.where(arr < 0, length + arr, arr)
                    return KeyPath([*stem, (name, offset)])

                if np.any(arr < 0):
                    raise ValueError("Cannot resolve negative index on unbounded slice without sequence length.")
                
                return KeyPath([*stem, (name, arr)])

            case (*stem, (name, slice() as parent_slice)):
                if (parent_slice.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_slice.start or 0
                p_stop = parent_slice.stop

                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
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
            1. **Identity Slice (`[:]`)**: If `length` is provided, validates `len(mask) == length`.
               Returns `np.flatnonzero(mask)`.
            2. **Known Fixed Span (`[start:stop]`)**: When both `start` and `stop` are
               non-negative, `span = stop - start`. Validates `len(mask) == span` and
               projects `start + np.flatnonzero(mask)`.
            3. **Open Right Bound (`[start:]`)**: Projects `start + np.flatnonzero(mask)`.
            4. **Existing Index Array (`NDArray`)**: Validates `len(mask) == len(parent_arr)`
               and filters the array in-place via `parent_arr[mask]`.
            5. **Scalar Index (`int`)**: Cannot be masked; raises `ValueError`.

        Args:
            mask: Sequence or 1D boolean array indicating elements to retain.
            length: Optional known sequence length of the active dimension, used
                for length validation when filtering unbounded slices.

        Returns:
            KeyPath: A new `KeyPath` instance whose terminal component has been
                narrowed to a 1D `np.int64` array of absolute indices.

        Raises:
            IndexError: If `len(mask)` does not match the known span of the target component.
            ValueError: In any of the following conditions:
                - The terminal component cannot be masked (e.g. a scalar index).
                - A mask is applied to a slice with negative bounds without sequence length.
            NotImplementedError: If the parent slice has a non-unit step (`step != 1`).

        Examples:
            >>> path = KeyPath().append("items")
            >>> path.narrow_mask([True, False, True]).indices[-1]
            array([0, 2])

            >>> slice_path = path.narrow_slice(slice(2, 7))  # items[2:7], span=5
            >>> slice_path.narrow_mask([True, False, False, True, False]).indices[-1]
            array([2, 5])
        """
        bool_mask = np.asarray(mask, dtype=bool)
        mask_len = len(bool_mask)

        match self.components:
            case (*stem, (name, slice() as parent_index)) if parent_index == slice(None, None, None):
                if length is not None and mask_len != length:
                    raise IndexError(f"Boolean mask length {mask_len} does not match container length {length}.")
                
                return KeyPath([*stem, (name, np.flatnonzero(bool_mask))])

            case (*stem, (name, slice() as parent_slice)):
                if (parent_slice.step or 1) != 1:
                    raise NotImplementedError("Only step=1 is supported.")

                p_start = parent_slice.start or 0
                p_stop = parent_slice.stop

                if p_start >= 0 and p_stop is not None and p_stop >= 0:
                    span = max(0, p_stop - p_start)
                    if mask_len != span:
                        raise IndexError(f"Boolean mask length {mask_len} does not match span {span}.")
                    
                    return KeyPath([*stem, (name, p_start + np.flatnonzero(bool_mask))])

                if p_start >= 0 and p_stop is None:
                    return KeyPath([*stem, (name, p_start + np.flatnonzero(bool_mask))])

                raise ValueError(
                    f"Cannot apply boolean mask to slice {parent_slice} with negative bounds without knowing sequence length."
                )

            case (*stem, (name, np.ndarray() as parent_arr)):
                span = len(parent_arr)
                if mask_len != span:
                    raise IndexError(f"Boolean mask length {mask_len} does not match span {span}.")
                
                return KeyPath([*stem, (name, parent_arr[bool_mask])])

            case _:
                raise ValueError("Cannot apply boolean mask to the terminal component.")
    

    def __getitem__(self, index):
        return tuple(self.components[index])

    def __len__(self) -> int:
        return len(self.components)

    def __str__(self) -> str:
        return ".".join(f"{name}{_format_index(idx)}" for name, idx in self.components)

    def __repr__(self) -> str:
        return f"KeyPath('{self}')"

    def __eq__(self, other: Any) -> bool:
        # Combine early exit type and length checks
        if not isinstance(other, KeyPath) or len(self.components) != len(other.components):
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
        hashed_components = []
        
        for name, idx in self.components:
            match idx:
                case np.ndarray():
                    raise TypeError("unhashable type: 'KeyPath' containing numpy array index")
                
                # Unpack slice attributes directly in the pattern match
                case slice(start=start, stop=stop, step=step):
                    hashed_components.append((name, (start, stop, step)))
                
                case _:
                    hashed_components.append((name, idx))

                    
        return hash(tuple(hashed_components))



def _add_bound(base: int | None, delta: int | None) -> int | None:
    if delta is None:
        return base
    
    if base is None:
        return delta
    
    if (base >= 0) != (delta >= 0):
        raise ValueError(f"Cannot compose mixed-sign bounds ({base}, {delta}) without sequence length.")
    
    return base + delta

def _format_index(index: IndexType) -> str:
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