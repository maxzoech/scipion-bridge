import abc

from typing import Any, Optional, Tuple, Union
from .schema import (
    Schema,
    _ArrayEntryBase,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
)

import numpy as np
from numpy.typing import ArrayLike


from scipion_bridge.backend.standalone.container import Container
from scipion_bridge.core.environment.storage import ArrayStorageProvider
from dependency_injector.wiring import Provide, inject

IndexType = Union[slice, int]

class _BaseStorage(metaclass=abc.ABCMeta):

    def __init__(self,
                 schema: Schema,
                 parent: Optional["_BaseStorage"],
                 path: Tuple[str, ...] = (),
                 offset: Optional[Tuple[Union[slice, int], ...]] = None
        ) -> None:
        super().__init__()

        self._schema = schema
        self.parent = parent
        self.offset = offset
        self.path = tuple(path)

    @property
    def root(self) -> str:
        return ".".join(self.path)

    def compute_offset(self, item: IndexType) -> Tuple[IndexType, ...]:
        """Compute the offset tuple when indexing or slicing along the active batch dimension."""
        offset = self.offset or ()

        if offset and isinstance(offset[-1], slice):

            base = offset[-1].start or 0
            new_last = (
                slice(base + item.start, base + item.stop, item.step)
                if isinstance(item, slice)
                else base + item
            )

            return (*offset[:-1], new_last)
        else:

            return (item,)
        
    def compute_slice_offset(self, start: int, stop: int) -> Tuple[IndexType, ...]:
        return self.compute_offset(slice(start, stop, 1))

    def compute_index_offset(self, index: int) -> Tuple[IndexType, ...]:
        return self.compute_offset(index)

    def schema(self) -> Schema:
        return self._schema

    @abc.abstractmethod
    def write_static_array(self, key: str, entry: _ArrayEntryBase, data: ArrayLike):
        ...

    @abc.abstractmethod
    def read_static_array(self, key: str, entry: _ArrayEntryBase) -> ArrayLike:
        ...

    @abc.abstractmethod
    def read_ragged_array(self, key: str, entry: _RaggedArraySetEntry) -> Any:
        ...

    @abc.abstractmethod
    def write_ragged_array(self, key: str, entry: _RaggedArraySetEntry, data: Any) -> None:
        ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
        ...
        

def _lookup_entry(schema: Schema, key: str) -> Optional[_ArrayEntryBase]:
    parts = key.split(".")
    curr: Optional[Schema] = schema
    for part in parts[:-1]:
        if curr is not None and part in curr.fields and curr.fields[part].children is not None:
            curr = curr.fields[part].children
        else:
            return None
    if curr is None:
        return None
    leaf = curr.fields.get(parts[-1])
    return leaf if isinstance(leaf, _ArrayEntryBase) else None


class ArrayStorage(_BaseStorage):

    @inject
    def __init__(self,
                 schema: Schema,
                 storage_provider: Optional[ArrayStorageProvider] = Provide[Container.storage_provider],
        ) -> None:
        super().__init__(schema=schema, parent=None, path=(), offset=None)
        assert storage_provider is not None

        self._schema = schema
        self._storage_group = storage_provider.create_group()


    def write_static_array(self, key: str, entry: _ArrayEntryBase, data: ArrayLike):

        if isinstance(entry, _RaggedArraySetEntry):
            raise NotImplementedError(
                f"Cannot write ragged array field '{key}' via static array storage. "
                "Ragged array storage backend is not implemented yet."
            )

        if not isinstance(entry, (_ArrayEntry, _ArraySetEntry)):
            raise TypeError(
                f"Expected static array entry for key '{key}', got {type(entry).__name__}."
            )

        arr = np.asanyarray(data)
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )
        
        arr = arr.astype(entry.dtype, copy=False)
        entry_shape = tuple(entry.shape)

        if isinstance(entry, _ArraySetEntry):
            entry_ndim = len(entry_shape)
            if arr.ndim < entry_ndim or arr.shape[-entry_ndim:] != entry_shape:
                raise ValueError(
                    f"Shape mismatch for key '{key}': expected trailing dimensions {entry_shape}, "
                    f"got data shape {arr.shape}."
                )
            if entry.capacity is not None and arr.shape[0] > entry.capacity:
                raise ValueError(
                    f"Capacity mismatch for key '{key}': data batch size {arr.shape[0]} exceeds capacity {entry.capacity}."
                )
        elif isinstance(entry, _ArrayEntry):
            if arr.ndim != len(entry_shape):
                raise ValueError(
                    f"Dimension count mismatch for key '{key}': expected {len(entry_shape)} dimensions, "
                    f"got {arr.ndim} (shape {arr.shape})."
                )
            for dim_idx, (expected_dim, actual_dim) in enumerate(zip(entry_shape, arr.shape)):
                if expected_dim is not None and expected_dim != actual_dim:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': expected dimension {dim_idx} to be {expected_dim}, "
                        f"got {actual_dim}."
                    )

        self._storage_group[key] = arr

    def read_static_array(self, key: str, entry: _ArrayEntryBase) -> ArrayLike:
        if isinstance(entry, _RaggedArraySetEntry):
            raise NotImplementedError(
                f"Cannot read ragged array field '{key}' via static array storage. "
                "Ragged array storage backend is not implemented yet."
            )

        if not isinstance(entry, (_ArrayEntry, _ArraySetEntry)):
            raise TypeError(
                f"Expected static array entry for key '{key}', got {type(entry).__name__}."
            )

        if key not in self._storage_group:
            schema_entry = _lookup_entry(self._schema, key) or entry
            if schema_entry.is_static:
                shape = (schema_entry.capacity, *schema_entry.shape) if isinstance(schema_entry, _ArraySetEntry) and schema_entry.capacity is not None else schema_entry.shape
                self._storage_group.create_dataset(key, shape=shape, dtype=schema_entry.dtype)
            else:
                raise AttributeError(
                    f"Field '{key}' is dynamic and has not been initialized."
                )

        arr = self._storage_group[key]._data
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )
        
        arr = arr.astype(entry.dtype, copy=False)
        return arr

    def read_ragged_array(self, key: str, entry: _RaggedArraySetEntry) -> Any:
        raise NotImplementedError(
            f"Ragged array storage reading for key '{key}' is not implemented yet."
        )

    def write_ragged_array(self, key: str, entry: _RaggedArraySetEntry, data: Any) -> None:
        raise NotImplementedError(
            f"Ragged array storage writing for key '{key}' is not implemented yet."
        )

    def __contains__(self, key: str) -> bool:
        return key in self._storage_group



class ArrayStorageView(_BaseStorage):

    def qualify_key(self, key: str) -> str:
        return ".".join((*self.path, key)) if self.path else key

    def read_static_array(self, key: str, entry: _ArrayEntryBase) -> ArrayLike:
        assert self.parent is not None

        full_key = self.qualify_key(key)
        indices = self.offset or tuple()

        arr = self.parent.read_static_array(full_key, entry)
        return arr[indices] # type: ignore


    def write_static_array(self, key: str, entry: _ArrayEntryBase, data: ArrayLike):
        assert self.parent is not None

        full_key = self.qualify_key(key)

        if self.offset:
            arr = self.parent.read_static_array(full_key, entry)
            arr[self.offset] = data # type: ignore
            return

        self.parent.write_static_array(full_key, entry, data)

    def read_ragged_array(self, key: str, entry: _RaggedArraySetEntry) -> Any:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        return self.parent.read_ragged_array(full_key, entry)

    def write_ragged_array(self, key: str, entry: _RaggedArraySetEntry, data: Any) -> None:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        self.parent.write_ragged_array(full_key, entry, data)


    def __contains__(self, key: str) -> bool:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        return full_key in self.parent