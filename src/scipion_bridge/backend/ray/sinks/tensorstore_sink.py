from __future__ import annotations
from typing import Dict, Any, Union, List, Optional, Set as SetType
from pathlib import Path
import asyncio
import numpy as np

import tensorstore as ts

from scipion_bridge.core.streaming.sink_writer import CommittingSinkWriter, Receipt
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


class TensorStoreSinkWriter(CommittingSinkWriter):
    """
    Asynchronous SinkWriter persisting SchemaConvertible items to Zarr groups via TensorStore.

    Implements the core reduction algebra:
    - SetEntryBase (ArraySetEntry) -> Concat / Append along axis 0.
    - ArrayEntry -> Last-Writer-Wins overwrite.

    With a run store, the writer owns its directory: the receipt of a write
    holds the rows of every appended array and the overwritten arrays, and
    ``recover`` cuts every array back to the rows of the last commit (arrays
    created after it, to none), so that the items written again are not
    appended twice.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._stores: Dict[str, Any] = {}
        # The paths of the arrays appended to and of those overwritten.
        self._appended: SetType[str] = set()
        self._overwritten: SetType[str] = set()

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
                self._appended.add(path_str)
                current_len = store.shape[0]
                new_len = current_len + arr.shape[0]
                store = await store.resize(exclusive_max=[new_len, *store.shape[1:]])
                self._stores[path_str] = store
                commit_futures.append(store[current_len:new_len].write(arr))

            else:
                # Last-Writer-Wins overwrite reduction algebra
                self._overwritten.add(path_str)
                if list(store.shape) != list(arr.shape):
                    store = await store.resize(exclusive_max=list(arr.shape))
                    self._stores[path_str] = store
                commit_futures.append(store[...].write(arr))

        if commit_futures:
            await asyncio.gather(*commit_futures)

    async def write_identified(self, item: Any, lid: bytes) -> Optional[Receipt]:
        """Write ``item``; the receipt holds the rows of the appended arrays
        and the overwritten arrays after it."""
        await self.write(item)
        return {
            "rows": {path: int(self._stores[path].shape[0]) for path in self._appended},
            "overwritten": sorted(self._overwritten),
        }

    async def recover(self, receipt: Optional[Receipt]) -> None:
        """Cut every array in the directory back to the rows of ``receipt``;
        an array it does not hold, to none."""
        rows: Dict[str, int] = {} if receipt is None else receipt["rows"]
        overwritten = set() if receipt is None else set(receipt["overwritten"])
        for path_str in _arrays_in(self.path):
            store = await ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": str(self.path / path_str)},
                    "open": True,
                },
            )
            if path_str not in overwritten and store.rank > 0:
                store = await store.resize(
                    exclusive_max=[rows.get(path_str, 0), *store.shape[1:]],
                    shrink_only=True,
                )
            self._stores[path_str] = store
        self._appended = set(rows)
        self._overwritten = overwritten

    async def finalize(self) -> None:
        """Finalize all open stores."""
        self._stores.clear()


def _arrays_in(path: Path) -> List[str]:
    """The Zarr arrays below ``path``, as paths relative to it."""
    return sorted(
        array.parent.relative_to(path).as_posix() for array in path.rglob(".zarray")
    )
