import abc

from typing import Any, Optional, Tuple, Union
from . import schema
from .schema import Schema, _ArrayEntryBase

import numpy as np
from numpy.typing import ArrayLike


from scipion_bridge.backend.standalone.container import Container
from scipion_bridge.core.environment.storage import ArrayStorageProvider
from dependency_injector.wiring import Provide, inject

class _BaseStorage(metaclass=abc.ABCMeta):

    def __init__(self,
                 schema: Schema,
                 parent: Optional["_BaseStorage"],
                 root: str,
                 offset: Optional[Tuple[Union[slice, int], ...]]
        ) -> None:
        super().__init__()

        self._schema = schema
        self.parent = parent
        self.offset = offset
        self.root = root

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
        super().__init__(schema=schema, parent=None, root="", offset=None)
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

    def read_static_array(self, key: str, entry: _ArrayEntryBase) -> ArrayLike:
        assert self.parent is not None

        key = ".".join([self.root, key])

        indices = self.offset or tuple()
        arr = self.parent.read_static_array(key, entry)
        return arr[indices] # type: ignore


    def write_static_array(self, key: str, entry: _ArrayEntryBase, data: ArrayLike):
        assert self.parent is not None

        key = ".".join([self.root, key])

        if self.offset:
            raise NotImplementedError

        # indices = self.offset or tuple()

        self.parent.write_static_array(key, entry, data)


    def __contains__(self, key: str) -> bool:
        raise NotImplementedError