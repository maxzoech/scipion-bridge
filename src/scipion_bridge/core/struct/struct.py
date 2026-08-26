import numpy as np
from typing import Type, TypeVar, Tuple, Union, Any, Optional
from typing_extensions import TypeAlias

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
        self._value = value
        self.name = name

        self._owner = None

    def __set_name__(self, owner: Any, name: str) -> None:
        if self.name is None:
            self.name = name

    @classmethod
    def new(
        cls, value: Optional[Union["Arg", int]] = None, *, name: Optional[str] = None
    ) -> "Arg":
        if isinstance(value, Arg):
            if name and value.name is None:
                value.name = name
            return value

        if value is not None and not isinstance(value, int):
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

    def infer(self, context: Optional[dict[Any, Any]] = None) -> "Arg":
        """Infer the concrete value of Dim using the given substitution context."""
        ctx = context or {}

        # Case 1: Direct substitution.
        # Occurs when `self` is a template Dim registered in `ctx` by Struct.__init__
        # (e.g. Array shape referencing Particle.H when Particle(H=128) is instantiated).
        if self in ctx:
            target = ctx[self]
            if isinstance(target, Arg):
                # If target Dim is also mapped in ctx, follow the chain; otherwise preserve the target Dim reference
                return target.infer(ctx) if target in ctx else target

            return Arg(target, name=self.name)

        # Case 2: Chained alias / parameter forwarding.
        # Occurs when `self` is not directly in `ctx`, but holds a reference to another Dim in `self.value`
        # (e.g. nested struct Particle.H referencing outer Class2D.H, or square dimension constraints H=Dim(size)).
        if isinstance(self._value, Arg):
            resolved_target = self._value.infer(ctx)
            val = resolved_target if resolved_target._value is not None else self._value
            return Arg(val, name=self.name)

        # Case 3: Standalone fallback.
        # Occurs when `self` is an independent literal default (e.g. Dim(64)) or unassigned dynamic Dim (None).
        return Arg(self._value, name=self.name)

    @property
    def is_static(self) -> bool:
        return isinstance(self.value, int) and self.value >= 0

    def __repr__(self) -> str:
        parts = []
        if self._value is not None or self.name is None:
            parts.append(repr(self._value))
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"{self.__class__.__name__}({', '.join(parts)}, id: {id(self):#x})"

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


class Array(Marker[T], SchemaConvertible):

    def __init__(
        self,
        dtype: Optional[np.dtype] = None,
        *,
        shape: Optional[Tuple[Union[Dim, int, None], ...]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype)

        if shape is None:
            raise ValueError(
                "Missing required argument 'shape' for Array. "
                "Expected a tuple of dimensions (e.g., shape=(1,), shape=(Dim('N'), 3), or shape=(None,))."
            )

        self.shape: Tuple[Dim, ...] = tuple([Dim.new(v) for v in shape])
        self.options = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def specialize(self, context: Optional[dict[Any, Any]] = None) -> "Array":
        if context is None:
            context = {}

        return Array(
            dtype=self.dtype,
            shape=tuple(dim.infer(context) for dim in self.shape),
            **self.options,
        )

    def convert_to_entry(self) -> Entry:
        assert self.dtype is not None
        resolved_shape = tuple(dim.value for dim in self.shape)
        return _ArrayEntry(np.dtype(self.dtype), shape=resolved_shape)

    def validate(self, other: Any) -> None:
        """Checks if another Array specification can be assigned to this marker.

        Raises:
            TypeError: If `other` is not an Array marker or has incompatible dtypes.
            ValueError: If rank differs or a static dimension would be overwritten.
        """
        if not isinstance(other, Array):
            raise TypeError(
                f"Expected an Array marker specification, but got '{type(other).__name__}'."
            )

        if len(self.shape) != len(other.shape):
            raise ValueError(
                f"Rank mismatch: cannot assign Array with rank {len(other.shape)} "
                f"(shape={list(other.shape)}) to target with rank {len(self.shape)} "
                f"(shape={list(self.shape)})."
            )

        for axis, (expected_dim, incoming_dim) in enumerate(
            zip(self.shape, other.shape)
        ):
            exp_val = expected_dim.value
            in_val = incoming_dim.value

            if exp_val is not None and in_val != exp_val:
                raise ValueError(
                    f"Dimension mismatch at axis {axis}: static dimension {exp_val} "
                    f"cannot be overwritten by {in_val} "
                    f"(target shape={list(self.shape)}, incoming shape={list(other.shape)})."
                )


class Struct(SchemaArrayStorage, SchemaConvertible):

    _schema_specs: dict[str, SchemaConvertible]
    _dim_specs: dict[str, Dim]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        def _init_default(dtype: Type):
            if _is_supported_scalar_value(dtype):
                return Array(dtype=np.dtype(dtype), shape=(1,))
            elif isinstance(dtype, type) and issubclass(dtype, SchemaConvertible):
                new_subtype = dtype()
                assert isinstance(new_subtype, SchemaConvertible)
                return new_subtype.specialize({})
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
            if k not in assigned_fields
        }

        cls_fields = {**assigned_fields, **unassigned_fields}

        dim_specs = {k: v for k, v in cls_fields.items() if isinstance(v, Arg)}
        schema_specs = {k: v for k, v in cls_fields.items() if not isinstance(v, Arg)}

        # Inherit specs from base classes in reverse MRO order
        for base in reversed(cls.__mro__):
            if hasattr(base, "_dim_specs"):
                dim_specs.update(base._dim_specs)
            if hasattr(base, "_schema_specs"):
                schema_specs.update(base._schema_specs)

        # Check that every assigned field in a struct is a SchemaConvertible type
        for name, v in assigned_fields.items():
            if not isinstance(v, (SchemaConvertible, Arg)):
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

        cls._dim_specs = dim_specs
        cls._schema_specs = schema_specs

    def __init__(self, **kwargs: Any) -> None:
        allowed_keys = set(self._schema_specs) | set(self._dim_specs)
        extra_keys = set(kwargs) - allowed_keys
        if extra_keys:
            raise TypeError(
                f"'{type(self).__name__}' got unexpected keyword argument(s): {list(extra_keys)}"
            )

        dim_context: dict[Any, Any] = {}

        for dim_name, default_dim in self._dim_specs.items():
            if dim_name in kwargs:
                val = kwargs[dim_name]

                if val is None and default_dim.value is not None:
                    raise ValueError(
                        f"Cannot override fixed dimension '{dim_name}' "
                        f"(value={default_dim.value}) with None."
                    )
                specialized_dim = Dim.new(val, name=dim_name)
            else:
                specialized_dim = default_dim.infer(dim_context)

            setattr(self, dim_name, specialized_dim)
            dim_context[default_dim] = specialized_dim

        for field_name, default_spec in self._schema_specs.items():
            specialized_field = default_spec.bind(kwargs.get(field_name), dim_context)
            setattr(self, field_name, specialized_field)

        self.schema = self._create_schema()

    def validate(self, other: Any) -> None:
        """Checks if another Struct specification can be assigned to this marker."""
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Expected field of type '{type(self).__name__}', but got '{type(other).__name__}'."
            )

    def specialize(self, context: Optional[dict[Any, Dim]] = None) -> "Struct":
        if context is None:
            context = {}

        resolved_kwargs = {}
        for name, value in self._dim_specs.items():
            replaced = context.get(name, value).infer(context)
            resolved_kwargs[name] = replaced

        for field_name in self._schema_specs:
            resolved_kwargs[field_name] = getattr(self, field_name).specialize(context)

        return type(self)(**resolved_kwargs)

    def _create_schema(self) -> Schema:
        return Schema(
            dtype=type(self),
            fields={
                k: getattr(self, k).convert_to_entry()
                for k in self._schema_specs.keys()
            },
        )

    def convert_to_entry(self) -> Entry:
        return _SchemaEntry(
            schema=self._create_schema(),
        )
