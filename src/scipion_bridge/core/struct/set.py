"""Set container implementation.

A Set is a fixed-capacity sequence container for Struct items, stored as flattened
N-dimensional Zarr arrays across the outer capacity dimension.
"""

from __future__ import annotations
from typing import (
    Type,
    Generic,
    TypeVar,
    Dict,
    Union,
    Any,
    ForwardRef,
    Tuple,
    Optional,
    Sequence,
    overload,
)
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from ._type_checks import is_struct_type
from .entries import (
    Entry,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaSetEntry,
    _StructEntry,
    SchemaConvertible,
    _StorageView,
)
from .schema import Schema

import numpy as np


def _convert_array_entry(entry: _ArrayEntry) -> Entry:
    """Convert a standard _ArrayEntry into its Set counterpart (_ArraySetEntry or _RaggedArraySetEntry)."""
    if entry.is_static:
        assert entry.min_shape == entry.max_shape
        assert entry.min_shape is not None

        return _ArraySetEntry(
            dtype=entry.dtype, storage=entry.storage, shape=entry.min_shape
        )
    else:
        return _RaggedArraySetEntry(
            dtype=entry.dtype,
            storage=entry.storage,
            min_shape=entry.min_shape,
            max_shape=entry.max_shape,
            preferred_shape=entry.preferred_shape,
        )


def generate_set_schema(cls: Type) -> Schema:
    """Build a set-context schema from a Struct type.

    Converts ``_ArrayEntry`` fields to their set equivalents
    (``_ArraySetEntry`` / ``_RaggedArraySetEntry``) and recursively
    handles nested structs.
    """
    if not is_struct_type(cls):
        raise TypeError("Element of a set has to be of type Struct.")

    fields: Dict[str, Entry] = {}
    for k, v in cls.schema().fields.items():
        if isinstance(v, _StructEntry):
            set_schema = generate_set_schema(v.struct_cls)
            fields[k] = _StructEntry(struct_cls=v.struct_cls, schema=set_schema)
        elif isinstance(v, _ArrayEntry):
            fields[k] = _convert_array_entry(entry=v)
        elif isinstance(v, _SchemaSetEntry):
            fields[k] = v
        else:
            raise NotImplementedError(f"Unknown entry in schema: {v}")

    return Schema(fields)


T = TypeVar("T")


class Set(Generic[T], SchemaConvertible):
    """Generic fixed-capacity sequence container of Struct items backed by Zarr arrays.

    Parameters:
        elements (Sequence[T]): Initial elements to populate into the set.
        capacity (int): The total maximum capacity (length) of the set.
    """

    __runtime_args__ = None
    _generic_cache: Dict = {}

    def configure_array_storage(self) -> Any:
        """Pre-allocate array storage for the configured set capacity."""
        return super().configure_array_storage(shape_prefix=(self._capacity,))

    @overload
    def __init__(self, elements: Sequence[T], *, capacity: Optional[int] = None) -> None: ...

    @overload
    def __init__(self, *, capacity: int) -> None: ...

    def __init__(
        self,
        elements: Sequence[Any] = (),
        *,
        capacity: Optional[int] = None,
    ):
        if not elements and capacity is None:
            raise ValueError("Must specify capacity when initializing an empty Set.")

        if capacity is None:
            capacity = len(elements)
        elif capacity < len(elements):
            raise ValueError(
                f"Specified capacity ({capacity}) cannot be smaller than number of elements ({len(elements)})."
            )

        self._capacity = capacity
        self._view: Optional[_StorageView] = None

        super().__init__()

        for i, elem in enumerate(elements):
            self[i] = elem


    @classmethod
    def to_schema_entry(cls) -> _SchemaSetEntry:
        return _SchemaSetEntry(
            schema=cls.schema(),
            capacity=cls.capacity(),
            item_type=cls.item_type()
        )

    @classmethod
    def _validate_as_field(cls, key_path: str) -> dict:
        from .schema import _validate_struct_datatypes

        wrapped_type = cls.item_type()
        is_serializable = _validate_struct_datatypes(wrapped_type, root=key_path)
        is_serializable[key_path] = all(is_serializable.values())
        return is_serializable

    @classmethod
    def __class_getitem__(cls, params):
        type_args = params if isinstance(params, tuple) else (params,)

        # Use Generic[T] behavior for TypeVars for type checkers
        if any(isinstance(t, TypeVar) for t in type_args):
            return super().__class_getitem__((params[0],))

        # TODO: Correctly handle forward-declared references
        if any(isinstance(t, (str, ForwardRef)) for t in type_args):
            return super().__class_getitem__((params[0],))

        cache_key = (cls, params)
        if cache_key in Set._generic_cache:
            return Set._generic_cache[cache_key]

        param_names = ",".join(getattr(t, "__name__", str(t)) for t in type_args)
        new_cls_name = f"{cls.__name__}[{param_names}]"

        new_cls = type(
            new_cls_name,
            (cls,),
            {
                "__module__": cls.__module__,
                "__runtime_args__": type_args,
                "__origin__": cls,
                "__args__": type_args,
            },
        )

        Set._generic_cache[cache_key] = new_cls
        return new_cls

    @classmethod
    def item_type(cls) -> Type:
        if not cls.__runtime_args__:
            raise TypeError(
                f"You must subscript {cls.__name__} (e.g., Set[CTF]) before calling schema()"
            )

        return cls.__runtime_args__[0]

    @classmethod
    def capacity(cls) -> Optional[int]:
        if not cls.__runtime_args__:
            raise TypeError(
                f"You must subscript {cls.__name__} (e.g., Set[CTF]) before calling schema()"
            )

        return cls.__runtime_args__[1] if len(cls.__runtime_args__) > 1 else None

    @classmethod
    def schema(cls):
        if not hasattr(cls, "_cached_schema"):
            item_type = cls.item_type()
            cls._cached_schema = generate_set_schema(item_type)
        return cls._cached_schema
    
    def _compute_slice_bounds(self, index: slice):
        start = index.start if index.start is not None else 0
        stop = index.stop if index.stop is not None else self._capacity

        if start < 0:
            start = self._capacity + start

        if stop < 0:
            stop = self._capacity + stop

        if index.step is not None and index.step != 1:
            raise NotImplementedError("Slicing with stride is not supported yet")
        assert stop > start

        return start, stop

    def _write_array(self, path: str, index: int, arr: np.ndarray) -> None:
        """Write an array value to Zarr slice at path and index."""

        if self._view is not None:
            full_key = f"{self._view.prefix}{path}"
            full_indices = (*self._view.indices, index)
            self._view.owner._zarr_group[full_key][full_indices] = np.asarray(arr)
        else:
            self._zarr_group[path][index] = np.asarray(arr)

    def _write_nested_set(self, path: str, index: int, nested_set: Set) -> None:
        """Write all leaf array fields of a nested Set into storage at the given index, with padding if needed."""
        target_set = self._view.owner if self._view is not None else self
        target_prefix = f"{self._view.prefix}{path}." if self._view is not None else f"{path}."
        target_indices = (*self._view.indices, index) if self._view is not None else (index,)

        for leaf_path, _ in nested_set.schema().tree_iter():
            full_path = f"{target_prefix}{leaf_path}"
            arr = nested_set._get_leaf_array(leaf_path)
            z_arr = target_set._zarr_group[full_path]
            if arr.ndim > 0 and z_arr.ndim > 1 and arr.shape[0] < z_arr.shape[1]:
                z_arr[target_indices + (slice(0, arr.shape[0]),)] = arr
            else:
                z_arr[target_indices] = arr

    def _get_leaf_array(self, path: str) -> np.ndarray:
        """Get the numpy array for a leaf path, delegating to owner set if this is a proxy view."""
        if self._view is not None:
            full_key = f"{self._view.prefix}{path}"
            return np.array(self._view.owner._zarr_group[full_key][self._view.indices])
        return np.array(self._zarr_group[path])

    def _get_el(self, index: int) -> T:
        item_type = self.item_type()
        new_el: T = item_type()

        if self._view is not None:
            new_el._view = self._view.sub_index_view(index)
        else:
            new_el._view = _StorageView(owner=self, indices=(index,), prefix="")
        
        return new_el

    def _get_slice(self, index: slice) -> Self:
        start, stop = self._compute_slice_bounds(index)

        cls = type(self)
        new_set = cls(capacity=stop - start)

        for path, _ in self.schema().tree_iter():
            new_set._zarr_group[path][:] = self._zarr_group[path][start:stop]

        return new_set

    def _set_el(self, index: int, value: T) -> None:
        if not isinstance(value, self.item_type()):
            provided = (
                f"Set of '{type(value)}'"
                if isinstance(value, Set)
                else f"'{type(value).__name__}'"
            )
            raise TypeError(
                f"Cannot assign {provided} to element of Set of '{self.item_type().__name__}'"
            )

        def _write_node(schema: Schema, obj: Any, *, prefix: str = ""):
            for name, entry in schema.fields.items():
                path = f"{prefix}.{name}" if prefix else name
                val = getattr(obj, name)

                if isinstance(entry, (_ArraySetEntry, _ArrayEntry)):
                    self._write_array(path, index, np.array(val))
                elif isinstance(entry, _StructEntry):
                    _write_node(entry.schema, val, prefix=path)
                elif isinstance(entry, _SchemaSetEntry):
                    self._write_nested_set(path, index, val)
                else:
                    raise NotImplementedError(f"Cannot write element for schema type {type(entry)}")

        _write_node(self.schema(), value)

    def _set_slice(self, index: slice, value: Set[T]) -> None:
        start, stop = self._compute_slice_bounds(index)
        if not (isinstance(value, Set) and value.item_type() == self.item_type()):
            provided = (
                f"Set of '{value.item_type()}'"
                if isinstance(value, Set)
                else f"'{type(value).__name__}'"
            )
            raise TypeError(
                f"Cannot assign {provided} to a slice of Set of '{self.item_type()}'"
            )

        slice_length = stop - start
        if slice_length != len(value):
            raise ValueError(
                f"Cannot assign a Set of capacity {len(value)} to a slice of length {slice_length}"
            )

        for path, _ in self.schema().tree_iter():
            self._zarr_group[path][start:stop] = value._zarr_group[path][:]

    def __len__(self):
        return self._capacity

    def __iter__(self):
        for i in range(self._capacity):
            yield self[i]

    def _normalize_index(self, key: Any) -> int:
        try:
            idx = int(key)
        except (TypeError, ValueError):
            raise TypeError(f"Indexing with {type(key).__name__} is not supported.")

        if idx < 0:
            idx = self._capacity + idx
        if idx < 0 or idx >= self._capacity:
            raise IndexError(
                f"Index {key} out of range for Set of capacity {self._capacity}"
            )
        return idx

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._get_slice(key)
        return self._get_el(self._normalize_index(key))

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            return self._set_slice(key, value)
        self._set_el(self._normalize_index(key), value)
