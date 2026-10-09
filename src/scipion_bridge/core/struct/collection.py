"""Statically indexed sequence container for Struct items backed by columnar storage."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import Self

import pyarrow as pa

from .schema import (
    CollectionEntry,
    Entry,
    Schema,
    SchemaConvertible,
    SchemaEntry,
)
from .storage import _BaseStorage, RootEngine
from .struct import Struct, leaf_columns, struct_leaf_arrays, write_struct_row
from .set import Set
from .utils.arrow_utils import leaves_to_batch, null_row
from ..utils.marker import Marker

T = TypeVar("T", bound=Struct)


class Collection(Marker[T], SchemaConvertible):
    """Statically indexed sequence container for Struct items backed by columnar storage."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls._dtype is not None:
            if not (isinstance(cls._dtype, type) and issubclass(cls._dtype, Struct)):
                raise TypeError(
                    f"Element of Collection has to be a Struct, got '{cls._dtype}'",
                )

    def __init__(
        self,
        size: int,
        items: Optional[Union[Sequence[T], Dict[int, T]]] = None,
        storage: Optional[_BaseStorage] = None,
        dtype: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype=dtype, **kwargs)

        if size <= 0:
            raise ValueError(f"Collection size must be positive, got {size}")

        if self._dtype is not None:
            if not (isinstance(self._dtype, type) and issubclass(self._dtype, Struct)):
                raise TypeError(
                    f"Element of Collection has to be a Struct, got '{self._dtype}'",
                )

        self.size = size
        self._storage = storage if storage is not None else RootEngine()
        self._owner_cls: Optional[type] = None

        if items is not None:
            self._populate_items(items)

    def _populate_items(
        self,
        items: Union[Sequence[T], Dict[int, T]],
    ) -> None:
        match items:
            case dict():
                for idx, item in items.items():
                    self[idx] = item

            case list() | tuple():
                for idx, item in enumerate(items):
                    self[idx] = item

            case _:
                raise TypeError(
                    f"Collection items must be a sequence or dict, got {type(items).__name__}",
                )

    def _check_bounds(self, index: int) -> None:
        if not (0 <= index < self.size):
            raise IndexError(
                f"Index {index} out of bounds for Collection(size={self.size})",
            )

    @property
    def storage(self) -> _BaseStorage:
        return self._storage

    def is_initialized(self, index: int) -> bool:
        self._check_bounds(index)
        slot_path = self._storage.root.append(str(index))
        return self._storage.is_initialized(slot_path)

    def initialized_indices(self) -> list[int]:
        return [i for i in range(self.size) if self.is_initialized(i)]

    def _get_element_entry(self) -> Entry:
        dt = self._dtype
        if dt is None:
            raise TypeError(
                "Cannot convert unsubscripted Collection to schema entry. Missing element type.",
            )

        if isinstance(dt, type) and issubclass(dt, Struct):
            return SchemaEntry(schema=dt.schema())

        raise TypeError(
            f"Unsupported element type for Collection: {dt!r}. Collection elements must be Struct subclasses.",
        )

    @classmethod
    def default(cls) -> SchemaConvertible:
        if cls._dtype is None:
            raise TypeError(
                "Cannot create default for unsubscripted Collection.",
            )
        raise ValueError(
            "Missing required argument 'size' for Collection. "
            "Collections must declare a static size (e.g. Collection[T](size=10)).",
        )

    @classmethod
    def item_type(cls) -> Type[Struct]:
        """Return the element Struct type of the Collection."""
        assert (
            cls._dtype is not None
        ), "Cannot retrieve item_type from unsubscripted Collection"
        return cls._dtype

    @classmethod
    def schema(cls) -> Schema:
        raise NotImplementedError(
            "Cannot get schema directly from an uninstantiated Collection class. "
            "Instantiate Collection with a size parameter first.",
        )

    def convert_to_entry(self) -> Entry:
        return CollectionEntry(
            element_entry=self._get_element_entry(),
            size=self.size,
        )

    def __len__(self) -> int:
        return self.size

    def __contains__(self, index: Any) -> bool:
        match index:
            case int() if 0 <= index < self.size:
                return self.is_initialized(index)

            case _:
                return False

    def __iter__(self) -> Iterator[T]:
        for idx in self.initialized_indices():
            yield self[idx]

    def items(self) -> Iterator[Tuple[int, T]]:
        for idx in self.initialized_indices():
            yield (idx, self[idx])

    def keys(self) -> list[int]:
        return self.initialized_indices()

    def values(self) -> list[T]:
        return list(self)

    def __getitem__(self, index: int) -> T:
        self._check_bounds(index)
        if not self.is_initialized(index):
            raise KeyError(
                f"Collection slot {index} is not initialized.",
            )

        assert self._dtype is not None, "Collection element type is not set."
        slot_storage = self._storage.append(str(index))
        return self._dtype(storage=slot_storage)

    def __setitem__(self, index: int, value: T) -> None:
        self._check_bounds(index)
        if not isinstance(value, Struct):
            raise TypeError(
                f"Collection elements must be Struct instances, got '{type(value).__name__}'.",
            )

        assert self._dtype is not None, "Collection element type is not set."
        if not isinstance(value, self._dtype):
            raise TypeError(
                f"Expected instance of {self._dtype.__name__}, got {type(value).__name__}.",
            )

        slot_storage = self._storage.append(str(index))
        if (
            value.storage.root == slot_storage.root
            and value.storage.root_storage is slot_storage.root_storage
        ):
            return

        self._storage.clear(slot_storage.root)

        for path, entry in value.schema().tree_iter():
            source_path = value.storage.root.extend(path)
            if source_path not in value.storage:
                continue

            data = value.storage.read(source_path, entry)
            target_path = slot_storage.root.extend(path)
            slot_storage.write(target_path, entry, data)

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._owner_cls = owner

    def __get__(
        self,
        instance: Optional[Struct],
        owner: Optional[Type[Struct]] = None,
    ) -> Any:
        if instance is None:
            return self

        assert self.name is not None, "Collection descriptor name is not set."
        slot_storage = instance.storage.append(self.name)
        return type(self)(
            size=self.size,
            storage=slot_storage,
            dtype=self._dtype,
        )

    def __set__(self, instance: Optional[Struct], value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(
                f"Cannot set Collection field '{self.name}' on non-Struct instance of type {type(instance).__name__}.",
            )

        if self.name is None:
            raise AttributeError("Collection descriptor name is not set.")

        if not isinstance(value, Collection):
            raise TypeError(
                f"Expected Collection value for '{self.name}', got '{type(value).__name__}'.",
            )

        if value.size != self.size:
            raise ValueError(
                f"Collection size mismatch for '{self.name}': expected {self.size}, got {value.size}.",
            )

        bound: Collection[T] = self.__get__(instance, type(instance))
        target_root = instance.storage.root.append(self.name)
        if (
            value._storage.root == target_root
            and value._storage.root_storage is instance.storage.root_storage
        ):
            return

        instance.storage.clear(target_root)
        for idx in value.initialized_indices():
            bound[idx] = value[idx]

    def to_set(self) -> Set[T]:
        assert self._dtype is not None, "Cannot convert untyped Collection to Set."
        active_items = [self[i] for i in self.initialized_indices()]
        return Set[self._dtype](active_items)

    @property
    def is_descriptor(self) -> bool:
        return self.name is not None

    def _pickle_cls(self) -> Type[SchemaConvertible]:
        # The element type may be given per instance (``Collection(dtype=...)``).
        return Collection[self._element_type()]

    def _element_type(self) -> Type[Struct]:
        assert self._dtype is not None, "Collection element type is not set."
        return self._dtype

    def to_arrow(self) -> pa.RecordBatch:
        """Export the Collection to a RecordBatch with one row per slot.

        Uninitialized slots, and fields not initialized in a slot, are null.
        The size is stored in the batch metadata.
        """
        schema = self._element_type().schema()
        slots = {
            index: struct_leaf_arrays(self._storage.append(str(index)), schema)
            for index in self.initialized_indices()
        }
        null_rows = {
            path: null_row(leaf)
            for leaves in slots.values()
            for path, leaf in leaves.items()
        }
        leaves = {
            path: pa.concat_arrays(
                [slots.get(index, {}).get(path, missing) for index in range(self.size)],
            )
            for path, missing in null_rows.items()
        }
        return leaves_to_batch(schema, leaves, metadata=_size_metadata(self.size))

    @classmethod
    def from_arrow(cls, batch: pa.RecordBatch) -> Self:
        """Construct a Collection from a RecordBatch produced by :meth:`to_arrow`."""
        collection = cls(size=_size_from_metadata(batch.schema.metadata))
        columns = leaf_columns(batch, collection._element_type().schema())
        for index in range(batch.num_rows):
            write_struct_row(
                collection._storage.append(str(index)),
                columns,
                row=index,
            )
        return collection

    def __repr__(self) -> str:
        elem_type = (
            getattr(self._dtype, "__name__", str(self._dtype))
            if self._dtype is not None
            else "?"
        )
        return f"Collection[{elem_type}](size={self.size}, active={len(self.initialized_indices())})"


_SIZE_KEY = b"scipion_bridge.size"


def _size_metadata(size: int) -> Dict[bytes, bytes]:
    return {_SIZE_KEY: str(size).encode()}


def _size_from_metadata(metadata: Optional[Dict[bytes, bytes]]) -> int:
    assert (
        metadata is not None and _SIZE_KEY in metadata
    ), "RecordBatch was not produced by Collection.to_arrow(): missing size."
    return int(metadata[_SIZE_KEY])
