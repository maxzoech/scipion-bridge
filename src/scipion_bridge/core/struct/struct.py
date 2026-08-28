import numpy as np
import copy

from typing import Mapping, Type, TypeVar, Tuple, Union, Any, Optional, Dict, Callable, cast
from typing_extensions import Self

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from functools import reduce

from .schema import SchemaConvertible, Entry, Schema, _SchemaEntry, _ArrayEntry
from .storage import SchemaArrayStorage
from ..utils.marker import Marker

T = TypeVar("T")


def _is_supported_scalar_value(cls: Type) -> bool:
    try:
        dt = np.dtype(cls)
        return dt.kind != "O"
    except (TypeError, ValueError):
        return False


class Arg:

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


class BoundArrayView(SchemaConvertible):
    """Read-only view returned when accessing an Array attribute on a Struct class."""

    def __init__(
        self, dtype: np.dtype, shape_spec: Tuple[Dim, ...], owner_cls: Type["Trait"]
    ) -> None:
        self._dtype = dtype
        self._shape_spec = shape_spec

        self._owner_cls = owner_cls

    @property
    def dtype(self) -> Optional[np.dtype]:
        return self._dtype

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        def _resolve(dim: Dim) -> Optional[int]:
            if not dim.name:
                return dim.value

            target = getattr(self._owner_cls, dim.name, dim)
            return target.value if isinstance(target, Arg) else target

        return tuple(_resolve(d) for d in self._shape_spec)

    @classmethod
    def default(cls) -> SchemaConvertible:
        raise ValueError(
            "Missing required argument 'shape' for Array. "
            "Expected a tuple of dimensions (e.g., shape=(1,), shape=(Dim('N'), 3), or shape=(None,))."
        )

    def convert_to_entry(self) -> Entry:
        assert self.dtype is not None
        assert issubclass(self._owner_cls, Trait)
        return _ArrayEntry(np.dtype(self.dtype), shape=self.shape)


class Array(Marker[T]):

    def __init__(
        self,
        dtype: Optional[np.dtype] = None,
        *,
        shape: Optional[Tuple[Union[Dim, int, None], ...]] = None,
    ) -> None:
        super().__init__(dtype)

        if shape is None:
            raise ValueError(
                "Missing required argument 'shape' for Array. "
                "Expected a tuple of dimensions (e.g., shape=(1,), shape=(Dim('N'), 3), or shape=(None,))."
            )

        self.shape_spec: Tuple[Dim, ...] = tuple([Dim.new(v) for v in shape])

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        assert owner is not None
        assert self.dtype is not None

        return BoundArrayView(self.dtype, self.shape_spec, owner_cls=owner)


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

        def _init_default(dtype: Type):
            if _is_supported_scalar_value(dtype):
                return BoundArrayView(
                    dtype=np.dtype(dtype),
                    shape_spec=(Dim(1),),
                    owner_cls=cls,
                )
            elif isinstance(dtype, type) and issubclass(dtype, SchemaConvertible):
                return dtype.default()
            else:
                raise TypeError(
                    f"Unsupported field type {dtype!r}. "
                    "Expected a supported scalar type or a valid schema convertible type."
                )

        cls_name = cls.__name__
        annotations = cls.__dict__.get("__annotations__", {})

        # Collect all candidate field names in declaration order
        assigned_fields = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

        unassigned_fields = {
            k: _init_default(v)
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


class Struct(Trait, SchemaConvertible, SchemaArrayStorage):
    """Materialization layer: builds the finalized Schema and binds array storage."""

    _bridge_struct_marker: bool = True
    _bridge_schema: Schema

    @classmethod
    def default(cls) -> "Struct":
        return cls()

    @classmethod
    def schema(cls) -> Schema:
        return cls._bridge_schema

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

        # Use getattr to trigger descriptors (Array -> BoundArrayView, Set -> BoundSetView)
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
        allowed_keys = set(self._arg_specs)
        extra_keys = set(kwargs) - allowed_keys
        if extra_keys:
            raise TypeError(
                f"'{type(self).__name__}' got unexpected keyword argument(s): {list(extra_keys)}"
            )

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
        return _SchemaEntry(
            schema=self.schema(),
        )
