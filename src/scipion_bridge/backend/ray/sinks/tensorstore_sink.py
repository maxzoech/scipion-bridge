from __future__ import annotations
from typing import Dict, Any, Union, List
from pathlib import Path
import asyncio
import numpy as np

import tensorstore as ts

from scipion_bridge.core.streaming.sink_writer import SinkWriter
from scipion_bridge.core.struct import Struct, Set
from scipion_bridge.core.struct.collection import Collection
from scipion_bridge.core.struct.schema import (
    SchemaConvertible,
    Schema,
    SchemaEntry,
    SchemaSetEntry,
    CollectionEntry,
    ArrayEntryBase,
    SetEntryBase,
    ArrayEntry,
)


def _get_schema_tree(item: SchemaConvertible) -> Schema:
    """Extract the complete schema tree for a SchemaConvertible item."""
    entry = item.convert_to_entry()
    match entry:
        case SchemaSetEntry(schema=s, capacity=cap):
            return s.to_set_schema(capacity=cap)
        case SchemaEntry(schema=s):
            return s
        case CollectionEntry():
            assert entry.children is not None
            return entry.children
        case _:
            if entry.children is not None:
                return entry.children
            raise TypeError(
                f"Cannot extract schema tree from entry type {type(entry).__name__}"
            )


class TensorStoreSinkWriter(SinkWriter):
    """
    Asynchronous SinkWriter persisting SchemaConvertible items to Zarr groups via TensorStore.

    Implements the core reduction algebra:
    - SetEntryBase (ArraySetEntry) -> Concat / Append along axis 0.
    - ArrayEntry -> Last-Writer-Wins overwrite.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._stores: Dict[str, Any] = {}

    async def _get_or_open_store(
        self,
        path_str: str,
        leaf_entry: ArrayEntryBase,
        arr: np.ndarray,
    ) -> Any:
        if path_str in self._stores:
            return self._stores[path_str]

        zarr_dir = self.path / path_str
        dtype_str = np.dtype(arr.dtype).str

        if isinstance(leaf_entry, SetEntryBase):
            element_shape = arr.shape[1:]
            initial_shape = [0, *element_shape]
            chunk_shape = [max(1, arr.shape[0]), *element_shape]
        else:
            initial_shape = list(arr.shape)
            chunk_shape = list(arr.shape) if arr.shape else [1]

        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(zarr_dir),
            },
            "metadata": {
                "dtype": dtype_str,
                "shape": initial_shape,
                "chunks": chunk_shape,
            },
            "create": True,
            "open": True,
        }
        store = await ts.open(spec)
        self._stores[path_str] = store
        return store

    async def write(self, item: SchemaConvertible) -> None:
        """Write an incoming batch/item into TensorStore Zarr datasets."""
        if not isinstance(item, (Set, Struct, Collection)):
            raise TypeError(
                f"Expected Set, Struct, or Collection, got {type(item)}",
            )

        schema = _get_schema_tree(item)
        commit_futures: List[Any] = []

        for path, leaf_entry in schema.tree_iter():
            assert isinstance(leaf_entry, ArrayEntryBase)
            path_str = "/".join(str(p) for p in path.path)

            full_path = item.storage.root.extend(path)
            raw_data = item.storage.read(full_path, leaf_entry)
            try:
                arr = np.asarray(raw_data)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert data at '{path}' to numpy array: {e}"
                ) from e

            if arr.dtype == object:
                raise ValueError(
                    f"Dynamic array at '{path}' cannot be converted to a dense numpy array (ragged dimensions detected).",
                )

            store = await self._get_or_open_store(path_str, leaf_entry, arr)

            if isinstance(leaf_entry, SetEntryBase):
                # Concat / Append reduction algebra
                current_len = store.shape[0]
                new_len = current_len + arr.shape[0]
                store = await store.resize(exclusive_max=[new_len, *store.shape[1:]])
                self._stores[path_str] = store
                commit_futures.append(store[current_len:new_len].write(arr))

            else:
                # Last-Writer-Wins overwrite reduction algebra
                if list(store.shape) != list(arr.shape):
                    store = await store.resize(exclusive_max=list(arr.shape))
                    self._stores[path_str] = store
                commit_futures.append(store[...].write(arr))

        if commit_futures:
            await asyncio.gather(*commit_futures)

    async def finalize(self) -> None:
        """Finalize all open stores."""
        self._stores.clear()
