"""Struct base class implementation.

A Struct is a schema-driven container backed by Zarr storage. Struct classes
define their field layouts via type annotations. Fields are stored internally
in a Zarr group and can be accessed or mutated via standard Python attribute access.
"""

from functools import cache
import numpy as np
import zarr
from zarr.storage import MemoryStore
from zarr.storage import LocalStore as DiskStore

from .schema import create_schema, Schema
from .entries import Entry, _ArrayEntry, _StructEntry, _SchemaSetEntry, SchemaConvertible
from .set import Set

from typing import Any


class Struct(SchemaConvertible):
    """Base class for schema-defined data structures backed by Zarr array storage.

    Subclasses define fields via type annotations (e.g. ``voltage_kv: float``).
    Array and scalar values are stored internally in `_zarr_group`.
    """

    _bridge_struct_marker = True  # Sentinel used by _type_checks.is_struct_type()

    # def configure_array_storage(self) -> zarr.Group:
    #     """Initialize a blank Zarr storage group for standalone Struct instances."""
    #     store = MemoryStore()
    #     return zarr.group(store=store)

    @classmethod
    def to_schema_entry(cls) -> _StructEntry:
        """Convert this Struct class into a ``_StructEntry`` for parent schemas."""
        return _StructEntry(struct_cls=cls, schema=cls.schema())

    @classmethod
    def _validate_as_field(cls, key_path: str) -> dict:
        """Validate that all fields of this Struct support array serialization."""
        from .schema import _validate_struct_datatypes
        return _validate_struct_datatypes(cls, root=key_path)

    @classmethod
    @cache
    def schema(cls) -> Schema:
        """Return the cached Schema describing the field layout of this Struct class."""
        return create_schema(cls)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a Struct instance, setting initial field values from keyword arguments."""
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)


    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept field assignment to write values into the underlying Zarr storage."""
        schema = type(self).schema()
        if name not in schema.entries() or isinstance(value, SchemaConvertible):
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
            else:
                raise NotImplementedError(f"Setting entry {entry} not supported")

    def _instantiate_nested_field(self, name: str, entry: Entry) -> Any:
        """Lazily instantiate and populate a child Struct or Set for a nested schema field."""
        if isinstance(entry, _StructEntry):
            child = entry.struct_cls()
        elif isinstance(entry, _SchemaSetEntry):
            child = Set[entry.item_type](capacity=entry.capacity)
        else:
            raise NotImplementedError(f"Cannot instantiate nested field for entry type {entry}")

        prefix = f"{name}."
        for leaf_path, _ in entry.schema.tree_iter():
            full_key = f"{prefix}{leaf_path}"
            if full_key in self._zarr_group:
                child._zarr_group[leaf_path] = np.array(self._zarr_group[full_key])

        super().__setattr__(name, child)
        return child

    def __getattribute__(self, name: str) -> Any:
        """Intercept attribute access to read values lazily from underlying Zarr storage."""
        schema = type(self).schema()
        attrs = set(schema.entries())

        if name not in attrs:
            return super().__getattribute__(name)

        entry = schema.fields[name]
        if isinstance(entry, (_StructEntry, _SchemaSetEntry)):
            try:
                return super().__getattribute__(name)
            except AttributeError:
                return self._instantiate_nested_field(name, entry)

        try:
            buffer = self._zarr_group[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        value = np.array(buffer)

        if "orig_shape" in buffer.attrs and "is_scalar" in buffer.attrs:
            orig_shape = buffer.attrs['orig_shape']
            is_scalar = buffer.attrs['is_scalar']

            value = value.reshape(orig_shape)
            
            if is_scalar:
                value = value.item()

        return value
