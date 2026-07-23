from functools import cache
import numpy as np
import zarr
from zarr.storage import MemoryStore
from zarr.storage import LocalStore as DiskStore

from .schema import create_schema, Schema
from .entries import _ArrayEntry, _StructEntry, SchemaConvertible

from typing import Any

class Struct(SchemaConvertible):

    _bridge_struct_marker = True  # Sentinel used by _type_checks.is_struct_type()

    def configure_array_storage(self) -> zarr.Group:
        store = MemoryStore()
        return zarr.group(store=store)

    @classmethod
    def to_schema_entry(cls) -> _StructEntry:
        return _StructEntry(struct_cls=cls, schema=cls.schema())

    @classmethod
    def _validate_as_field(cls, key_path: str) -> dict:
        from .schema import _validate_struct_datatypes
        return _validate_struct_datatypes(cls, root=key_path)

    @classmethod
    @cache
    def schema(cls) -> Schema:
        return create_schema(cls)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)


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
        elif isinstance(schema.fields[name], _StructEntry):
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