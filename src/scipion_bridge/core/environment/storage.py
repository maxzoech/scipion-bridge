"""Storage providers for schema-driven Struct and Set data containers."""

import abc
from typing import Any, Dict, Optional, Tuple
import numpy as np


class NumPyStorageArray:
    """Storage array backed directly by numpy.ndarray."""

    def __init__(self, shape: Tuple[int, ...], dtype: Any, data: Optional[np.ndarray] = None):
        if data is not None:
            self._data = np.asarray(data, dtype=dtype)
        else:
            self._data = np.zeros(shape, dtype=dtype)
        self.attrs: Dict[str, Any] = {}

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> Any:
        return self._data.dtype

    @property
    def size(self) -> int:
        return self._data.size

    def __getitem__(self, item: Any) -> Any:
        return self._data[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        self._data[item] = value

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype is None:
            arr = self._data
        else:
            arr = self._data.astype(dtype)
        if copy:
            return arr.copy()
        return arr


class NumPyStorageGroup:
    """Group container mimicking a zarr.Group interface backed by NumPy storage."""

    def __init__(self):
        self._arrays: Dict[str, NumPyStorageArray] = {}

    def create_dataset(self, name: str, shape: Tuple[int, ...], dtype: Any, overwrite: bool = True) -> NumPyStorageArray:
        arr = NumPyStorageArray(shape, dtype)
        self._arrays[name] = arr
        return arr

    # Alias for backward/forward compatibility
    create_array = create_dataset

    def __getitem__(self, name: str) -> NumPyStorageArray:
        if name not in self._arrays:
            raise KeyError(f"Storage group has no key '{name}'")
        return self._arrays[name]

    def __setitem__(self, name: str, value: Any) -> None:
        if isinstance(value, NumPyStorageArray):
            self._arrays[name] = value
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            val_arr = np.asarray(value)
            arr = NumPyStorageArray(val_arr.shape, val_arr.dtype, data=val_arr)
            self._arrays[name] = arr
        else:
            val_arr = np.asarray(value)
            arr = NumPyStorageArray(val_arr.shape, val_arr.dtype, data=val_arr)
            self._arrays[name] = arr

    def __contains__(self, name: str) -> bool:
        return name in self._arrays


class ArrayStorageProvider(abc.ABC):
    """Abstract base class for array storage backend providers."""

    @abc.abstractmethod
    def create_group(self, shape_prefix: Tuple[int, ...] = ()) -> Any:
        """Create and return a new array storage group instance."""
        ...


class NumPyStorageProvider(ArrayStorageProvider):
    """Default array storage provider backed by NumPy in-memory data structures."""

    def create_group(self, shape_prefix: Tuple[int, ...] = ()) -> NumPyStorageGroup:
        return NumPyStorageGroup()
