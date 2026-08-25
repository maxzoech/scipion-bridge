import numpy as np
from typing import Type, TypeVar, Tuple, Union, Any, Optional

from . import schema
from .schema import SchemaConvertable
from .storage import SchemaArrayStorage
from ..utils.marker import Marker
from ..utils.type_annotation import has_untyped_class_definitions

T = TypeVar("T")


def _is_array_type(cls: Type) -> bool:
    try:
        dt = np.dtype(cls)
        return dt.kind != "O"
    except (TypeError, ValueError):
        return False


class Dim:

    def __init__(
        self,
        value: Optional[Union[int, "Dim"]] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        self.value = value
        self.name = name

    def __set_name__(self, owner: Any, name: str) -> None:
        if self.name is None:
            self.name = name

    @classmethod
    def new(cls, value: Optional[Union["Dim", int]] = None, *, name: Optional[str] = None) -> "Dim":
        if isinstance(value, Dim):
            if name and value.name is None:
                value.name = name
            return value

        if value is not None and not isinstance(value, int):
            raise TypeError(
                f"Expected Dim, int, or None, but got {type(value).__name__}: {value!r}"
            )

        return Dim(value, name=name)

    def resolve_value(self) -> Optional[int]:
        """Recursively resolves down to the underlying integer or None."""
        if isinstance(self.value, Dim):
            return self.value.resolve_value()
        
        return self.value

    def hydrate(self, context: Optional[dict[Any, Any]] = None) -> "Dim":
        """Hydrates this Dim against the given substitution context."""
        ctx = context or {}

        # Case 1: Direct substitution.
        # Occurs when `self` is a template Dim registered in `ctx` by Struct.__init__
        # (e.g. Array shape referencing Particle.H when Particle(H=128) is instantiated).
        if self in ctx:
            target = ctx[self]
            if isinstance(target, Dim):
                # If target Dim is also mapped in ctx, follow the chain; otherwise preserve the target Dim reference
                return target.hydrate(ctx) if target in ctx else target

            return Dim(target, name=self.name)

        # Case 2: Chained alias / parameter forwarding.
        # Occurs when `self` is not directly in `ctx`, but holds a reference to another Dim in `self.value`
        # (e.g. nested struct Particle.H referencing outer Class2D.H, or square dimension constraints H=Dim(size)).
        if isinstance(self.value, Dim):
            resolved_target = self.value.hydrate(ctx)
            val = resolved_target if resolved_target.value is not None else self.value
            return Dim(val, name=self.name)

        # Case 3: Standalone fallback.
        # Occurs when `self` is an independent literal default (e.g. Dim(64)) or unassigned dynamic Dim (None).
        return Dim(self.value, name=self.name)

    @property
    def is_static(self) -> bool:
        val = self.resolve_value()
        return isinstance(val, int) and val >= 0

    def __repr__(self) -> str:
        parts = []
        if self.value is not None or self.name is None:
            parts.append(repr(self.value))
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        val = self.resolve_value()
        if self.name is not None and val is not None:
            return f"{self.name}:{val}"
        return self.name or (str(val) if val is not None else "?")

    def __int__(self) -> int:
        val = self.resolve_value()
        if val is None:
            raise TypeError("Cannot convert unassigned Dim to int")
        return int(val)

    def __eq__(self, other: Any) -> bool:
        other_val = other.resolve_value() if isinstance(other, Dim) else other
        return self.resolve_value() == other_val

    def __hash__(self) -> int:
        return id(self)


class Array(Marker[T], schema.SchemaConvertable):

    def __init__(
        self,
        dtype: Optional[np.dtype] = None,
        *,
        shape: Tuple[Union[Dim, int, None], ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype)

        self.shape: Tuple[Dim, ...] = tuple([Dim.new(v) for v in shape])
        self.options = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_static(self) -> bool:
        """Returns True if all shape dimensions are defined integers (no None or dynamic dims)."""
        return all(dim.is_static for dim in self.shape)

    def hydrate(self, context: Optional[dict[Any, Any]] = None) -> "Array":
        if context is None:
            context = {}

        return Array(
            dtype=self.dtype,
            shape=tuple(dim.hydrate(context) for dim in self.shape),
            **self.options,
        )

    def convert_to_entry(self) -> schema.Entry:
        assert self.dtype is not None
        resolved_shape = tuple(dim.resolve_value() for dim in self.shape)
        return schema._ArrayEntry(np.dtype(self.dtype), shape=resolved_shape)

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
            exp_val = expected_dim.resolve_value()
            in_val = incoming_dim.resolve_value()

            if exp_val is not None and in_val is not None and in_val != exp_val:
                raise ValueError(
                    f"Dimension mismatch at axis {axis}: static dimension {exp_val} "
                    f"cannot be overwritten by {in_val} "
                    f"(target shape={list(self.shape)}, incoming shape={list(other.shape)})."
                )

    def default(self) -> "Array":
        return self.hydrate({})


class Struct(SchemaArrayStorage, SchemaConvertable):

    _schema_specs: dict[str, SchemaConvertable]
    _dim_specs: dict[str, Dim]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls_name = cls.__name__
        annotations = getattr(cls, "__annotations__", {})

        if has_untyped_class_definitions(cls):
            raise TypeError(
                f"Schema class '{cls_name}' contains class-level attributes missing type annotations. "
                f"All fields in a Schema must be explicitly annotated. "
                f"Example: 'field_name: int = 0' instead of 'field_name = 0'."
            )

        dim_fields: dict[str, Dim] = {}
        spec_fields: dict[str, schema.SchemaConvertable] = {}

        # Inherit specs from base classes in reverse MRO order
        for base in reversed(cls.__mro__):
            if hasattr(base, "_dim_specs"):
                dim_fields.update(base._dim_specs)
            if hasattr(base, "_schema_specs"):
                spec_fields.update(base._schema_specs)

        for field_name, field_type in annotations.items():
            default_val = getattr(cls, field_name, None)

            if field_type is Dim:
                dim_fields[field_name] = Dim.new(default_val, name=field_name)

            elif _is_array_type(field_type):
                spec_fields[field_name] = Array(
                    dtype=np.dtype(field_type),
                    shape=(),
                    _scalar_type=field_type,
                )

            elif isinstance(field_type, type) and issubclass(field_type, Array):
                if not default_val:
                    raise ValueError(
                        f"Field '{field_name}' in '{cls_name}' is typed as '{field_type}', "
                        f"but is missing a default Array specification. "
                        f"Expected: {field_name}: {field_type} = Array(shape=(...))"
                    )

                spec_fields[field_name] = default_val

            elif isinstance(field_type, type) and issubclass(
                field_type, schema.SchemaConvertable
            ):
                if default_val is None:
                    default_val = field_type()

                if not isinstance(default_val, SchemaConvertable):
                    raise TypeError(
                        f"Field '{field_name}' in '{cls_name}' expects a default value of type '{field_type.__name__}' "
                        f"(subclass of SchemaConvertable), but got '{type(default_val).__name__}'."
                    )

                spec_fields[field_name] = default_val

            else:
                raise TypeError(
                    f"Invalid type annotation '{cls!r}' for field '{field_name}' in Schema '{cls}'. "
                    f"Expected a primitive numeric/scalar type (e.g., float, int, bool) or an Array type (e.g., Array[float]), "
                    f"but got an unsupported or non-convertible type."
                )

        cls._dim_specs = dim_fields
        cls._schema_specs = spec_fields

    def validate(self, other: Any) -> None:
        """Checks if another Struct specification can be assigned to this marker."""
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Expected field of type '{type(self).__name__}', but got '{type(other).__name__}'."
            )

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
                hydrated_dim = Dim.new(kwargs[dim_name], name=dim_name)
            else:
                hydrated_dim = default_dim.hydrate(dim_context)

            setattr(self, dim_name, hydrated_dim)
            dim_context[default_dim] = hydrated_dim

        for field_name, default_spec in self._schema_specs.items():
            hydrated_field = default_spec.bind(kwargs.get(field_name), dim_context)
            setattr(self, field_name, hydrated_field)

        self.schema = self._create_schema()

    def hydrate(self, context: Optional[dict[Any, Any]] = None) -> "Struct":
        if context is None:
            context = {}

        resolved_kwargs: dict[str, Any] = {
            dim_name: getattr(self, dim_name, default_dim).hydrate(context)
            for dim_name, default_dim in self._dim_specs.items()
        }

        for field_name, default_spec in self._schema_specs.items():
            bound_field = getattr(self, field_name, default_spec)
            if field_name in self.__dict__ and bound_field is not default_spec:
                resolved_kwargs[field_name] = bound_field.hydrate(context)

        return type(self)(**resolved_kwargs)

    def default(self) -> "Struct":
        return self.hydrate({})

    def is_static(self) -> bool:
        return all(f.is_static for f in self.schema.fields.values())

    def _create_schema(self) -> schema.Schema:
        return schema.Schema(
            dtype=type(self),
            fields={
                k: getattr(self, k).convert_to_entry()
                for k in self._schema_specs.keys()
            },
        )

    def convert_to_entry(self) -> schema.Entry:
        return schema._SchemaEntry(
            schema=self._create_schema(),
        )
