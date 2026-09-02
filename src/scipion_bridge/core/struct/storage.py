import abc

from typing import Any, Optional, Tuple, Union
from . import schema
from .schema import Schema, _ArrayEntryBase

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
    def __contains__(self, key: str) -> bool:
        ...
        

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

        if not isinstance(entry, (schema._ArrayEntry, schema._ArraySetEntry)) or not entry.is_static:
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
        entry_ndim = len(entry_shape)

        if arr.ndim < entry_ndim or arr.shape[-entry_ndim:] != entry_shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': expected trailing dimensions {entry_shape}, "
                f"got data shape {arr.shape}."
            )

        if isinstance(entry, schema._ArraySetEntry):
            # TODO: Validate input here
            pass

        self._storage_group[key] = arr

    def read_static_array(self, key: str, entry: _ArrayEntryBase) -> ArrayLike:
        if not isinstance(entry, (schema._ArrayEntry, schema._ArraySetEntry)) or not entry.is_static:
            raise TypeError(
                f"Expected static array entry for key '{key}', got {type(entry).__name__}."
            )

        if key not in self._storage_group:
            self._storage_group.create_dataset(key, shape=entry.shape, dtype=entry.dtype)

        arr = self._storage_group[key]._data
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )
        
        arr = arr.astype(entry.dtype, copy=False)
        return arr

    def __contains__(self, key: str) -> bool:
        return key in self._storage_group



class ArrayStorageView(_BaseStorage):

    def qualify_key(self, key: str) -> str:
        return ".".join((*self.path, key))

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
            raise NotImplementedError

        self.parent.write_static_array(full_key, entry, data)


    def __contains__(self, key: str) -> bool:
        assert self.parent is not None
        full_key = self.qualify_key(key)
        return full_key in self.parent