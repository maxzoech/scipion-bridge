"""Descriptor and metaprogramming layer for Struct definition and materialization."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Self, TypeAlias
import awkward as ak
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
from functools import cache

from .schema import (
    ArrayEntry,
    ArrayEntryBase,
    SchemaConvertible,
    Entry,
    Schema,
    SchemaEntry,
    SchemaSetEntry,
    SetEntryBase,
)
from ..utils.marker import Marker
from .storage import _BaseStorage, RootEngine
from .exceptions import UninitializedFieldError
from .key_path import KeyPath
from .utils.arrow_utils import (
    find_nested_column,
    is_regular_awkward,
    leaf_from_arrow,
    leaf_to_arrow,
    leaves_to_batch,
)


def _is_supported_scalar_value(cls: Type) -> bool:
    try:
        dt = np.dtype(cls)
        return dt.kind != "O"
    except (TypeError, ValueError):
        return False


def _init_default_trait_field(owner_cls: Type["Trait"], dtype: Type) -> Any:
    """Initialize a default field marker for a Trait annotation."""
    if _is_supported_scalar_value(dtype):
        return Array(
            dtype=np.dtype(dtype),
            shape=(Dim(1),),
            is_scalar=True,
        )
    elif isinstance(dtype, type) and issubclass(dtype, SchemaConvertible):
        return dtype.default()
    else:
        raise TypeError(
            f"Unsupported field type {dtype!r}. "
            "Expected a supported scalar type or a valid schema convertible type."
        )


class Arg:
    """Class-level dimension specification or named parameter."""

    def __init__(
        self,
        value: Optional[Union[int, "Arg"]] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        if value is not None and (
            not isinstance(value, (int, Arg)) or isinstance(value, bool)
        ):
            raise TypeError(
                f"Expected Dim, int, or None, but got {type(value).__name__}: {value!r}"
            )

        self._value = value
        self.name = name
        self._owner: Optional[type] = None

    def __set_name__(self, owner: Any, name: str) -> None:
        if self.name is None:
            self.name = name
        self._owner = owner

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            return self
        return self.value

    def __set__(self, instance: Any, value: Any) -> None:
        raise AttributeError(
            f"Dimension '{self.name}' on {type(instance).__name__} is a class-level schema parameter and cannot be modified on an instance."
        )

    @classmethod
    def new(
        cls, value: Optional[Union["Arg", int]] = None, *, name: Optional[str] = None
    ) -> "Arg":
        if isinstance(value, Arg):
            if name and value.name is None:
                value.name = name
            return value

        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise TypeError(
                f"Expected Dim, int, or None, but got {type(value).__name__}: {value!r}"
            )

        return Arg(value, name=name)

    @property
    def value(self) -> Optional[int]:
        return self._resolve_value()

    def _resolve_value(self) -> Optional[int]:
        """Recursively resolves down to the underlying integer or None."""
        if isinstance(self._value, Arg):
            return self._value._resolve_value()
        return self._value

    def validate(self, other: Any) -> None:
        if other is not None and (
            not isinstance(other, (Arg, int)) or isinstance(other, bool)
        ):
            raise TypeError(
                f"Expected Dim, int, or None, but got {type(other).__name__}: {other!r}"
            )

        other_val = other.value if isinstance(other, Arg) else other
        if other_val is None and self.value is not None:
            raise ValueError(
                f"Cannot override fixed dimension '{self.name}' "
                f"(value={self.value}) with None."
            )

    @property
    def is_static(self) -> bool:
        return isinstance(self.value, int) and self.value >= 0

    def __repr__(self) -> str:
        parts = []
        if self._value is not None or self.name is None:
            parts.append(repr(self._value))
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        if self.name is not None and self.value is not None:
            return f"{self.name}:{self.value}"
        return self.name or (str(self.value) if self.value is not None else "?")

    def __int__(self) -> int:
        if self.value is None:
            raise TypeError("Cannot convert unassigned Dim to int")
        return int(self.value)

    def __eq__(self, other: Any) -> bool:
        other_val = other.value if isinstance(other, Arg) else other
        return self.value == other_val

    def __hash__(self) -> int:
        return id(self)


Dim: TypeAlias = Arg

T = TypeVar("T", bound=Union[np.generic, float, int, bool])


class Array(Marker[T], SchemaConvertible):
    """Descriptor and schema representation for array attributes on Struct classes."""

    def __init__(
        self,
        dtype: Optional[Union[np.dtype, type, str]] = None,
        *,
        shape: Optional[Union[Tuple[Union[Dim, int, None], ...], list]] = None,
        is_scalar: bool = False,
    ) -> None:
        super().__init__(dtype)

        if shape is None:
            raise ValueError(
                "Missing required argument 'shape' for Array. "
                "Expected a tuple of dimensions (e.g., shape=(1,), shape=(Dim('N'), 3), or shape=(None,))."
            )

        if not isinstance(shape, (tuple, list)):
            raise TypeError(
                f"Expected shape to be a tuple or list of dimensions, but got {type(shape).__name__}: {shape!r}"
            )

        shape_items: list[Dim] = []
        for v in shape:
            if isinstance(v, int) and not isinstance(v, bool) and v < 0:
                raise ValueError(f"Array dimension cannot be negative, got: {v}")

            shape_items.append(Dim.new(v))

        self.is_scalar = is_scalar
        self.shape_spec: Tuple[Dim, ...] = tuple(shape_items)

        self._owner_cls: Optional[type] = None

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        return tuple([d.value for d in self.shape_spec])

    @property
    def dtype(self) -> Optional[np.dtype]:
        return np.dtype(self._dtype) if self._dtype is not None else None

    @classmethod
    def default(cls) -> SchemaConvertible:
        raise ValueError(
            "Missing required argument 'shape' for Array. "
            "Expected a tuple of dimensions (e.g., shape=(1,), shape=(Dim('N'), 3), or shape=(None,))."
        )

    @classmethod
    def schema(cls) -> Schema:
        raise NotImplementedError(
            "Cannot get schema directly from an uninstantiated Array class."
        )

    @property
    def is_descriptor(self) -> bool:
        return True

    def to_arrow(self) -> pa.RecordBatch:
        raise TypeError("Array is a field descriptor and holds no data.")

    @classmethod
    def from_arrow(cls, batch: pa.RecordBatch) -> Self:
        raise TypeError("Array is a field descriptor and holds no data.")

    def convert_to_entry(self) -> Entry:
        if self.dtype is None:
            owner_name = (
                f" on '{self._owner_cls.__name__}'"
                if self._owner_cls is not None
                else ""
            )
            raise TypeError(
                f"Array field '{self.name}' on {owner_name} is missing a dtype specification."
            )

        if self._owner_cls is not None and not (
            isinstance(self._owner_cls, type) and issubclass(self._owner_cls, Trait)
        ):
            raise TypeError(
                f"Owner class '{self._owner_cls}' must be a subclass of Trait."
            )

        shape = tuple([d.value for d in self.shape_spec])
        return ArrayEntry(np.dtype(self.dtype), shape=shape)

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._owner_cls = owner

    @overload
    def __get__(self, instance: None, owner: Any) -> "Array[T]": ...

    @overload
    def __get__(
        self, instance: "Struct", owner: Optional[Any] = None
    ) -> Union[NDArray, Any]: ...

    def __get__(self, instance: Optional[Struct], owner: Optional[type] = None) -> Any:
        if instance is None:
            return self

        if not isinstance(instance, Struct):
            raise TypeError(
                f"Cannot read Array field '{self.name}' on non-Struct instance of type {type(instance).__name__}."
            )

        if self.name is None:
            raise AttributeError("Array descriptor name is not set.")

        path = instance.storage.root.append(self.name)
        val = instance.storage.read(
            path,
            self.entry,
        )
        if self.is_scalar:
            return val.item()

        return val

    def __set__(self, instance: Struct, value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(
                f"Cannot assign Array field '{self.name}' on non-Struct instance of type {type(instance).__name__}."
            )

        if self.name is None:
            raise AttributeError("Array descriptor name is not set.")

        # TODO: Validate the shape against the array spec

        path = instance.storage.root.append(self.name)
        instance.storage.write(
            path,
            self.entry,
            value,
        )

    def __repr__(self) -> str:
        dtype_str = (
            getattr(
                self.dtype, "name", getattr(self.dtype, "__name__", str(self.dtype))
            )
            if self.dtype is not None
            else "?"
        )
        owner_str = (
            f", owner={self._owner_cls.__name__}" if self._owner_cls is not None else ""
        )
        return f"Array[{dtype_str}](shape={self.shape}{owner_str})"


class Trait:
    """Specification layer: accumulates fields, dimensions, and shape descriptors."""

    _cls_fields: Dict[str, Any]
    _arg_specs: Dict[str, Arg]
    _bridge_trait_marker: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.__name__ == "Struct" and cls.__module__ == __name__:
            cls._cls_fields = {}
            cls._arg_specs = {}
            return

        cls_name = cls.__name__
        annotations = cls.__dict__.get("__annotations__", {})

        # Collect all candidate field names in declaration order
        assigned_fields = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

        unassigned_fields = {
            k: _init_default_trait_field(cls, v)
            for k, v in annotations.items()
            if not k.startswith("_") and k not in assigned_fields
        }

        # Check that every assigned field is a SchemaConvertible, Arg, or Marker
        for name, v in assigned_fields.items():
            if not isinstance(v, (SchemaConvertible, Arg, Marker)):
                if name not in annotations:
                    raise TypeError(
                        f"Struct '{cls_name}' contains class-level attributes missing type annotations. "
                        f"Field '{name}' was assigned {type(v).__name__!r} (value: {v!r}) without an annotation. "
                        f"Did you mean '{name}: {type(v).__name__} = {v!r}' or '{name} = B.Dim({v!r})'?"
                    )
                else:
                    raise TypeError(
                        f"Field '{name}' in '{cls_name}' was assigned an invalid default value of type "
                        f"'{type(v).__name__}' (value: {v!r}). Expected a SchemaConvertible or Arg specification."
                    )

        cls_fields: Dict[str, Any] = {}
        arg_specs: Dict[str, Arg] = {}

        for base in reversed(cls.__mro__):
            if hasattr(base, "_cls_fields"):
                cls_fields.update(base._cls_fields)
            if hasattr(base, "_arg_specs"):
                arg_specs.update(base._arg_specs)

        # Validate overriding dimensions from base classes
        for k, v in assigned_fields.items():
            if k in arg_specs:
                arg_specs[k].validate(v)
            if isinstance(v, Arg):
                arg_specs[k] = v

        for k, v in unassigned_fields.items():
            if isinstance(v, (Array, Marker, SchemaConvertible)):
                v.__set_name__(cls, k)
                setattr(cls, k, v)

        cls_fields.update(assigned_fields)
        cls_fields.update(unassigned_fields)

        cls._arg_specs = arg_specs
        cls._cls_fields = cls_fields

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not issubclass(type(self), Struct):
            raise TypeError(
                f"Cannot instantiate pure Trait '{type(self).__name__}'. "
                f"Mix it into a B.Struct to create a concrete entity."
            )


class Struct(Trait, SchemaConvertible):
    """Materialization layer: builds the finalized Schema and binds array storage."""

    _bridge_struct_marker: bool = True
    _bridge_schema: Schema
    _is_field_descriptor: bool = False

    @classmethod
    def default(cls) -> "Struct":
        return cls()

    @classmethod
    def schema(cls) -> Schema:
        return cls._bridge_schema

    @property
    def storage(self) -> _BaseStorage:
        assert self._storage is not None
        return self._storage

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Use getattr to trigger descriptors
        schema_specs: Dict[str, SchemaConvertible] = {}
        for k, v in cls._cls_fields.items():
            resolved = getattr(cls, k, v)
            if isinstance(resolved, SchemaConvertible):
                schema_specs[k] = resolved

        cls._bridge_schema = Schema(
            dtype=cls,
            fields={k: v.convert_to_entry() for k, v in schema_specs.items()},
        )

    def __init__(
        self,
        storage: Optional[_BaseStorage] = None,
        **kwargs: Any,
    ) -> None:

        if storage is not None and not isinstance(storage, _BaseStorage):
            raise TypeError(
                f"Expected _BaseStorage instance, got {type(storage).__name__}"
            )

        self._storage = storage
        self._storage = storage if storage is not None else RootEngine()

        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_initialized(self, field_name: str) -> bool:
        """Check if a field has been initialized in storage."""
        path = self.storage.root.append(field_name)
        return self.storage.is_initialized(path)

    def initialized_fields(self) -> list[str]:
        """Return the names of all fields that have been initialized."""
        return [name for name in self.schema().fields if self.is_initialized(name)]

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self._is_field_descriptor = True

    def __get__(
        self,
        instance: Optional["Struct"],
        owner: Optional[Type["Struct"]] = None,
    ):

        if instance is None:
            return self

        assert self.name is not None, "Struct descriptor name is not set."

        if not isinstance(instance, Struct):
            raise TypeError(
                f"Cannot read Array field '{self.name}' on non-Struct instance of type {type(instance).__name__}."
            )

        return type(self)(storage=instance.storage.append(self.name))

    def __set__(self, instance: Optional["Struct"], value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(
                f"Expected Struct instance, got '{type(instance).__name__}'."
            )

        if not isinstance(value, Struct):
            raise TypeError(f"Expected Struct value, got '{type(value).__name__}'.")

        if self.name is None:
            raise AttributeError("Struct descriptor name is not set.")

        field_entry = instance.schema().fields[self.name]
        if not isinstance(field_entry, (SchemaEntry, SchemaSetEntry)):
            raise AttributeError

        dtype = field_entry.schema.dtype
        assert dtype is not None

        if not isinstance(value, dtype):
            raise ValueError

        target_root = instance.storage.root.append(self.name)
        if (
            value.storage.root == target_root
            and value.storage.root_storage is instance.storage.root_storage
        ):
            return

        for path, entry in field_entry.schema.tree_iter():
            source_path = value.storage.root.extend(path)
            if source_path not in value.storage:
                continue

            data = value.storage.read(source_path, entry)
            target_path = target_root.extend(path)
            instance.storage.write(target_path, entry, data)

    def convert_to_entry(self) -> Entry:
        return SchemaEntry(schema=self.schema())

    @property
    def is_descriptor(self) -> bool:
        return self._is_field_descriptor

    def to_arrow(self) -> pa.RecordBatch:
        """Export the initialized fields to a RecordBatch with a single row.

        For a view (a Set element, a Collection item or a nested field), only
        the data of the view is exported. Nested Sets become nested list
        columns.
        """
        return leaves_to_batch(
            self.schema(),
            struct_leaf_arrays(self.storage, self.schema()),
            metadata={},
        )

    @classmethod
    def from_arrow(cls, batch: pa.RecordBatch) -> Self:
        """Construct a Struct owning its storage from a batch of :meth:`to_arrow`."""
        struct = cls()
        write_struct_row(struct.storage, leaf_columns(batch, cls.schema()), row=0)
        return struct


def struct_leaf_arrays(
    storage: _BaseStorage,
    schema: Schema,
) -> Dict[KeyPath, pa.Array]:
    """Export the initialized leaves of the Struct at ``storage.root`` as 1-row arrays."""
    return {
        path: leaf_to_arrow(_with_row_axis(storage.read(key, entry)), entry)
        for path, entry in schema.tree_iter()
        if (key := storage.root.extend(path)) in storage
    }


_LeafColumn = Tuple[ArrayEntryBase, Union[np.ndarray, ak.Array], np.ndarray]


def leaf_columns(
    batch: pa.RecordBatch,
    schema: Schema,
) -> Dict[KeyPath, _LeafColumn]:
    """Convert the leaf columns of ``batch`` into buffers, with their validity per row.

    Every column is converted as a whole: Awkward ignores the offset of sliced
    Arrow arrays with a validity bitmap, so rows are indexed afterwards.
    """
    return {
        path: (
            entry,
            leaf_from_arrow(column, entry),
            column.is_valid().to_numpy(zero_copy_only=False),
        )
        for path, entry in schema.tree_iter()
        if (column := find_nested_column(batch, path)) is not None
    }


def write_struct_row(
    storage: _BaseStorage,
    columns: Dict[KeyPath, _LeafColumn],
    row: int,
) -> None:
    """Write row ``row`` of the leaf columns into the Struct at ``storage.root``.

    Leaves that are missing from the columns or null in the row stay
    uninitialized.
    """
    for path, (entry, values, valid) in columns.items():
        if not valid[row]:
            continue

        storage.write(
            storage.root.extend(path),
            entry,
            _row_value(values[row], entry),
        )


def _with_row_axis(data: Union[np.ndarray, ak.Array]) -> Union[np.ndarray, ak.Array]:
    """Add a leading axis of length 1, turning a Struct leaf into a 1-row column."""
    match data:
        case np.ndarray():
            # Unlike np.newaxis, reshape gives the new axis a regular stride,
            # which Arrow requires for tensor columns.
            return data.reshape((1, *data.shape))
        case ak.Array():
            return ak.unflatten(data, [len(data)], axis=0)
        case _:
            raise TypeError(
                f"Cannot export Struct leaf of type '{type(data).__name__}' to Arrow.",
            )


def _row_value(value: Any, entry: ArrayEntryBase) -> Any:
    """Convert a row of a restored column into the leaf value of a Struct.

    Regular array fields are restored as NumPy arrays, as they were written;
    nested Sets keep their (possibly ragged) Awkward representation.
    """
    match (entry, value):
        case (_, ak.Array()) if entry.is_static:
            # A row of a ragged nested Set column; its own shape is regular.
            return ak.to_numpy(value)
        case (SetEntryBase(), _):
            return value
        case (_, ak.Array()) if is_regular_awkward(value):
            return ak.to_numpy(value)
        case _:
            return value
