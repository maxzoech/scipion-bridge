import numpy as np
from typing import Type, TypeVar, Tuple, Union, Any, Optional

from .schema import SchemaConvertible, Entry, Schema, _SchemaEntry, _ArrayEntry
from .storage import SchemaArrayStorage
from ..utils.marker import Marker
from ..utils.type_annotation import has_untyped_class_definitions

T = TypeVar("T")


def _is_supported_scalar_value(cls: Type) -> bool:
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

        self._owner = None

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

    def infer(self, context: Optional[dict[Any, Any]] = None) -> "Dim":
        """Infer the concrete value of Dim using the given substitution context."""
        ctx = context or {}

        # Case 1: Direct substitution.
        # Occurs when `self` is a template Dim registered in `ctx` by Struct.__init__
        # (e.g. Array shape referencing Particle.H when Particle(H=128) is instantiated).
        if self in ctx:
            target = ctx[self]
            if isinstance(target, Dim):
                # If target Dim is also mapped in ctx, follow the chain; otherwise preserve the target Dim reference
                return target.infer(ctx) if target in ctx else target

            return Dim(target, name=self.name)

        # Case 2: Chained alias / parameter forwarding.
        # Occurs when `self` is not directly in `ctx`, but holds a reference to another Dim in `self.value`
        # (e.g. nested struct Particle.H referencing outer Class2D.H, or square dimension constraints H=Dim(size)).
        if isinstance(self.value, Dim):
            resolved_target = self.value.infer(ctx)
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
        return f"{self.__class__.__name__}({', '.join(parts)}, id: {id(self):#x})"

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


class Array(Marker[T], SchemaConvertible):

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
        """Returns True if all shape dimensions are defined integers."""
        return all(dim.is_static for dim in self.shape)

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
        resolved_shape = tuple(dim.resolve_value() for dim in self.shape)
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
            exp_val = expected_dim.resolve_value()
            in_val = incoming_dim.resolve_value()

            if exp_val is not None and in_val != exp_val:
                raise ValueError(
                    f"Dimension mismatch at axis {axis}: static dimension {exp_val} "
                    f"cannot be overwritten by {in_val} "
                    f"(target shape={list(self.shape)}, incoming shape={list(other.shape)})."
                )

    def default(self) -> "Array":
        return self.specialize({})


class Struct(SchemaArrayStorage, SchemaConvertible):

    _schema_specs: dict[str, SchemaConvertible]
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
        spec_fields: dict[str, SchemaConvertible] = {}

        # Inherit specs from base classes in reverse MRO order
        for base in reversed(cls.__mro__):
            if hasattr(base, "_dim_specs"):
                dim_fields.update(base._dim_specs)
            if hasattr(base, "_schema_specs"):
                spec_fields.update(base._schema_specs)

        for field_name, field_type in annotations.items():
            default_val = getattr(cls, field_name, None)

            # 1. Dimension field
            if field_type is Dim:
                dim_fields[field_name] = Dim.new(default_val, name=field_name)

            # 2. Primitive scalar shorthand (float, int, bool -> Array with shape (1,))
            elif _is_supported_scalar_value(field_type):
                spec_fields[field_name] = Array(
                    dtype=np.dtype(field_type),
                    shape=(1,),
                )

            # 3. SchemaConvertible field (Array, Struct, Set)
            elif isinstance(field_type, type) and issubclass(
                field_type, SchemaConvertible
            ):
                if issubclass(field_type, Array) and not default_val:
                    raise ValueError(
                        f"Field '{field_name}' in '{cls_name}' is typed as '{field_type}', "
                        f"but is missing a default Array specification. "
                        f"Expected: {field_name}: {field_type} = Array(shape=(...))"
                    )

                default_val = default_val if default_val is not None else field_type()
                if not isinstance(default_val, SchemaConvertible):
                    raise TypeError(
                        f"Field '{field_name}' in '{cls_name}' expects a default value of type '{field_type.__name__}' "
                        f"(subclass of SchemaConvertible), but got '{type(default_val).__name__}'."
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

                if val is None and default_dim.resolve_value() is not None:
                    raise ValueError(
                        f"Cannot override fixed dimension '{dim_name}' "
                        f"(value={default_dim.resolve_value()}) with None."
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

        # resolved_kwargs: dict[str, Any] = {
        #     dim_name: getattr(self, dim_name, default_dim).infer(context)
        #     for dim_name, default_dim in self._dim_specs.items()
        # }

        for field_name in self._schema_specs:
            resolved_kwargs[field_name] = getattr(self, field_name).specialize(context)

        return type(self)(**resolved_kwargs)

    def default(self) -> "Struct":
        return self.specialize({})

    def is_static(self) -> bool:
        return all(f.is_static for f in self.schema.fields.values())

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
