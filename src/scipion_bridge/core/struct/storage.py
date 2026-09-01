import abc

from typing import Any, Optional
from . import schema
from .schema import Schema, _ArrayEntryBase

import numpy as np
from numpy.typing import ArrayLike


from scipion_bridge.backend.standalone.container import Container
from scipion_bridge.core.environment.storage import ArrayStorageProvider
from dependency_injector.wiring import Provide, inject

class Storage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def schema(self) -> Schema:
        ...

    @abc.abstractmethod
    def write_static_array(self, key: str, entry: _ArrayEntryBase, data: ArrayLike):
        ...

    @abc.abstractmethod
    def read_static_array(self, key: str, entry: _ArrayEntryBase) -> ArrayLike:
        ...

    def __setitem__(self, key, value):
        if isinstance(key, str):
            fields = self.schema().fields
            if key not in fields:
                raise ValueError

            entry = fields[key]
            if isinstance(entry, schema._ArrayEntryBase) and entry.is_static:
                self.write_static_array(key, entry, value)
            else:
                raise NotImplementedError

    def __getitem__(self, key):

        if isinstance(key, str):
            fields = self.schema().fields
            if key not in fields:
                raise ValueError

            entry = fields[key]
            if isinstance(entry, schema._ArrayEntryBase) and entry.is_static:
                return self.read_static_array(key, entry)
            else:
                raise NotImplementedError
            

class ArrayStorage(Storage):

    @inject
    def __init__(self,
                 schema: Schema,
                 storage_provider: Optional[ArrayStorageProvider] = Provide[Container.storage_provider],
        ) -> None:
        super().__init__()
        assert storage_provider is not None

        self._schema = schema
        self._storage_group = storage_provider.create_group()


    def schema(self) -> Schema:
        return self._schema

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

        arr = self._storage_group[key]._data
        if not np.can_cast(arr.dtype, entry.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast data of dtype '{arr.dtype}' to field '{key}' dtype '{entry.dtype}'."
            )
        
        arr = arr.astype(entry.dtype, copy=False)
        return arr

    def __contains__(self, key: str) -> bool:
        return key in self._storage_group