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
import numpy as np
from numpy.typing import NDArray

from .schema import (
    ArrayEntryBase,
    ArrayEntry,
    SchemaConvertible,
    Entry,
    Schema,
    SchemaEntry,
    _ArrayEntryBase,
    _SchemaEntry,
    _ArrayEntry,
)
from ..utils.marker import Marker
from .storage import _BaseStorage, ArrayStorage, ArrayStorageView


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
            owner_cls=owner_cls,
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
        owner_cls: Optional[Type["Trait"]] = None,
        name: Optional[str] = None,
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

        self.shape_spec: Tuple[Dim, ...] = tuple(shape_items)
        self._owner_cls: Optional[Type["Trait"]] = owner_cls
        self.is_scalar: bool = is_scalar
        if name is not None:
            self.name = name

    @property
    def dtype(self) -> Optional[np.dtype]:
        if self._dtype is None:
            return None
        if isinstance(self._dtype, np.dtype):
            return self._dtype
        try:
            return np.dtype(self._dtype)
        except (TypeError, ValueError):
            return self._dtype  # type: ignore

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        def _resolve(dim: Union[Dim, int, None]) -> Optional[int]:
            if isinstance(dim, Arg):
                if not dim.name:
                    return dim.value
                if self._owner_cls is not None:
                    target = getattr(self._owner_cls, dim.name, dim)
                    return target.value if isinstance(target, Arg) else target
                return dim.value
            return dim

        return tuple(_resolve(d) for d in self.shape_spec)

    @classmethod
    def default(cls) -> SchemaConvertible:
        raise ValueError(
            "Missing required argument 'shape' for Array. "
            "Expected a tuple of dimensions (e.g., shape=(1,), shape=(Dim('N'), 3), or shape=(None,))."
        )

    @classmethod
    def schema(cls) -> Schema:
        raise NotImplementedError("Cannot get schema directly from an uninstantiated Array class.")

    def convert_to_entry(self) -> Entry:
        if self.dtype is None:
            owner_name = f" on '{self._owner_cls.__name__}'" if self._owner_cls is not None else ""
            raise TypeError(
                f"Array field '{self.name}'{owner_name} is missing a dtype specification."
            )
        if self._owner_cls is not None and not (
            isinstance(self._owner_cls, type) and issubclass(self._owner_cls, Trait)
        ):
            raise TypeError(
                f"Owner class '{self._owner_cls}' must be a subclass of Trait."
            )
        return ArrayEntry(np.dtype(self.dtype), shape=self.shape)

    def _bind(self, owner: Type["Trait"]) -> "Array[T]":
        return type(self)(
            dtype=self._dtype,
            shape=self.shape_spec,
            owner_cls=owner,
            name=self.name,
            is_scalar=self.is_scalar,
        )

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._owner_cls = owner

    @overload
    def __get__(self, instance: None, owner: Any) -> "Array[T]": ...

    @overload
    def __get__(self, instance: "Struct", owner: Optional[Any] = None) -> Union[NDArray, Any]: ...

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            if owner is None:
                return self
            if self.dtype is None:
                raise TypeError(
                    f"Array field '{self.name}' on '{owner.__name__}' is missing a dtype specification. "
                    f"Specify a dtype using Array[dtype](...) or Array(dtype=...)."
                )
            if self._owner_cls is None or self._owner_cls != owner:
                return self._bind(owner)
            return self

        if not isinstance(instance, Struct):
            raise TypeError(
                f"Cannot access Array field '{self.name}' on non-Struct instance of type {type(instance).__name__}."
            )
        if self.name is None:
            raise AttributeError("Array descriptor name is not set.")

        entry = instance.schema().fields.get(self.name)
        if entry is None or not isinstance(entry, ArrayEntryBase):
            raise AttributeError(f"Field '{self.name}' not found in Struct schema.")

        arr = instance.storage.read((self.name,), entry=entry)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray for field '{self.name}', got {type(arr).__name__}.")

        if self.is_scalar:
            return arr.item()
        return arr

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(
                f"Cannot assign Array field '{self.name}' on non-Struct instance of type {type(instance).__name__}."
            )
        if self.name is None:
            raise AttributeError("Array descriptor name is not set.")

        schema = instance.schema()
        entry = schema.fields.get(self.name)
        if entry is None or not isinstance(entry, ArrayEntryBase):
            raise AttributeError(f"Field '{self.name}' not found in Struct schema.")

        if np.ndim(value) == 0 and self.is_scalar:
            value = np.asarray(value).reshape([1])

        instance.storage.write(
            (self.name,),
            entry=entry,
            data=value,
        )

    def __repr__(self) -> str:
        dtype_str = getattr(self.dtype, "name", getattr(self.dtype, "__name__", str(self.dtype))) if self.dtype is not None else "?"
        owner_str = f", owner={self._owner_cls.__name__}" if self._owner_cls is not None else ""
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

    @classmethod
    def default(cls) -> "Struct":
        return cls()

    @classmethod
    def schema(cls) -> Schema:
        return cls._bridge_schema

    @property
    def storage(self) -> _BaseStorage:
        return self._storage

    def __init_subclass__(
        cls, specializations: Dict[str, int] = {}, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)

        for k, v in specializations.items():
            if k not in cls._arg_specs:
                raise TypeError(
                    f"Unknown schema overwrite argument '{k}' for '{cls.__name__}'. "
                    f"Available dimensions: {list(cls._arg_specs.keys())}"
                )

            new_arg = Arg.new(v, name=k)
            cls._arg_specs[k].validate(new_arg)

            cls._arg_specs[k] = new_arg
            cls._cls_fields[k] = new_arg
            setattr(cls, k, new_arg)

        # Use getattr to trigger descriptors (Array -> bound Array, Set -> BoundSetView)
        schema_specs: Dict[str, SchemaConvertible] = {}
        for k, v in cls._cls_fields.items():
            resolved = getattr(cls, k, v)
            if isinstance(resolved, SchemaConvertible):
                schema_specs[k] = resolved

        cls._bridge_schema = Schema(
            dtype=cls,
            fields={k: v.convert_to_entry() for k, v in schema_specs.items()},
        )

    def __init__(self, **kwargs: Any) -> None:
        storage = kwargs.pop("_storage_view", None)
        if storage is None:
            storage = ArrayStorage(schema=self._bridge_schema)

        if not isinstance(storage, _BaseStorage):
            raise TypeError(f"Expected _BaseStorage instance, got {type(storage).__name__}")

        self._storage = storage

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __get__(
        self,
        instance: Optional["Struct"],
        owner: Optional[Type["Struct"]] = None,
    ) -> Any:
        if isinstance(instance, Struct):
            if self._name is None:
                raise AttributeError("Struct descriptor name is not set.")
            parent = instance._storage.parent or instance._storage
            new_path = (*instance._storage.path, self._name)

            subview = ArrayStorageView(
                self.schema(),
                parent=parent,
                path=new_path,
                offset=instance._storage.offset,
            )

            return type(self)(_storage_view=subview)

        return self

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(instance, Struct):
            raise TypeError(f"Expected Struct instance, got '{type(instance).__name__}'.")
        if not isinstance(value, Struct):
            raise TypeError(f"Expected Struct value, got '{type(value).__name__}'.")
        if self._name is None:
            raise AttributeError("Struct descriptor name is not set.")

        for key, entry in value.schema().tree_iter():
            data = value._storage.read(key, entry)
            instance._storage.write((self._name, *key), entry=entry, data=data)

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @classmethod
    def static(cls: Type[Self], **kwargs: Union[int, Arg]) -> Type[Self]:
        """Create a new specialized subclass of this Struct with concrete dimension values."""
        for k in kwargs:
            if k not in cls._arg_specs:
                raise TypeError(
                    f"'{cls.__name__}.static()' got unexpected dimension argument: {k!r}. "
                    f"Available dimensions: {list(cls._arg_specs.keys())}"
                )

        args_suffix = "__".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
        subclass_name = f"{cls.__name__}_{args_suffix}" if args_suffix else f"{cls.__name__}_Static"

        subtype = type(
            subclass_name,
            (cls,),
            {},
            specializations=kwargs,
        )

        return cast(Type[Self], subtype)

    def convert_to_entry(self) -> Entry:
        return SchemaEntry(schema=self.schema())
