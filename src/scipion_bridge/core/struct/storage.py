from functools import cache
import numpy as np
import zarr
from zarr.storage import MemoryStore
from zarr.storage import LocalStore as DiskStore

from .schema import create_schema, Schema, _ArrayEntry


class Struct:

    @classmethod
    @cache
    def schema(cls) -> Schema:
        return create_schema(cls)

    def __init__(self) -> None:
        store = MemoryStore()
        self._zarr_group: zarr.Group = zarr.group(store=store)

    def __setattr__(self, name, value):
        schema = type(self).schema()
        if name not in schema.entries() or isinstance(value, Struct):
            super().__setattr__(name, value)
        else:

            input_has_shape = hasattr(value, "shape") or hasattr(value, "__len__")

            entry = schema.fields[name]
            if isinstance(entry, _ArrayEntry):
                value = np.array(value).astype(entry.dtype)
                orig_shape = value.shape

                value = np.reshape(value, [-1])
                is_scalar = not input_has_shape and value.size == 1

                buffer = self._zarr_group.create_array(
                    name=name,
                    shape=value.shape,
                    dtype=entry.dtype,
                    overwrite=True
                )

                buffer.attrs['orig_shape'] = orig_shape
                buffer.attrs['is_scalar'] = is_scalar

                buffer[:] = value
            else:  # pragma: no cover
                super().__setattr__(name, value)

    def __getattribute__(self, name):
        schema = type(self).schema()
        attrs = set(schema.entries())

        if name not in attrs:
            return super().__getattribute__(name)
        elif isinstance(schema.fields[name], type) and issubclass(schema.fields[name], Struct):
            return super().__getattribute__(name)
        else:
            try:
                buffer = self._zarr_group[name]
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            orig_shape = buffer.attrs['orig_shape']
            is_scalar = buffer.attrs['is_scalar']

            value = np.array(buffer).reshape(orig_shape)
            if is_scalar:
                value = value.item()

            return value

    @classmethod
    def print_schema(cls) -> None:  # pragma: no cover
        schema = create_schema(cls)
        schema.print_tree(cls.__qualname__)

    def print_storage(self) -> None:  # pragma: no cover
        print(self._zarr_group.tree())
