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

from .schema import (
    CollectionEntry,
    Entry,
    Schema,
    SchemaConvertible,
    SchemaEntry,
)
from .storage import _BaseStorage, StagingEngine
from .struct import Struct
from .set import Set
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
        self._storage = storage if storage is not None else StagingEngine()
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
        instance.storage.clear(target_root)
        for idx in value.initialized_indices():
            bound[idx] = value[idx]

    def to_set(self) -> Set[T]:
        assert self._dtype is not None, "Cannot convert untyped Collection to Set."
        active_items = [self[i] for i in self.initialized_indices()]
        return Set[self._dtype](active_items)

    def __repr__(self) -> str:
        elem_type = (
            getattr(self._dtype, "__name__", str(self._dtype))
            if self._dtype is not None
            else "?"
        )
        return f"Collection[{elem_type}](size={self.size}, active={len(self.initialized_indices())})"
