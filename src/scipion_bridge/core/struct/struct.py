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


class Array(Marker[T], schema.SchemaConvertable):

    def __init__(
        self,
        dtype: Optional[np.dtype] = None,
        *,
        shape: Tuple[Union[int, None], ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype)

        self.shape = shape
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_static(self) -> bool:
        """Returns True if all shape dimensions are defined integers (no None or dynamic dims)."""
        return all(isinstance(dim, int) and dim >= 0 for dim in self.shape)

    def convert_to_entry(self) -> schema.Entry:
        assert self.dtype is not None
        return schema._ArrayEntry(np.dtype(self.dtype), shape=self.shape)

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
                f"Rank mismatch: cannot assign Array with rank {len(self.shape)} "
                f"(shape={list(self.shape)}) to target with rank {len(other.shape)} "
                f"(shape={list(other.shape)})."
            )

        for axis, (incoming_dim, expected_dim) in enumerate(
            zip(self.shape, other.shape)
        ):
            if expected_dim is not None and incoming_dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch at axis {axis}: static dimension {expected_dim} "
                    f"cannot be overwritten by {incoming_dim} "
                    f"(target shape={list(other.shape)}, incoming shape={list(self.shape)})."
                )

    def default(self, value: Optional[Any] = None) -> "Array":
        if value is None:
            return Array(dtype=self.dtype, shape=self.shape)

        if not isinstance(value, Array):
            raise TypeError(
                f"Expected an Array marker specification, but got '{type(value).__name__}'."
            )

        value.validate(self)
        if value.dtype is None:
            value._dtype = self.dtype
        return value


class Struct(SchemaArrayStorage, SchemaConvertable):

    _schema_specs: dict[str, SchemaConvertable]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls_name = cls.__name__
        annotations = getattr(cls, "__annotations__", {})
        fields: dict[str, schema.SchemaConvertable] = {}

        if has_untyped_class_definitions(cls):
            raise TypeError(
                f"Schema class '{cls_name}' contains class-level attributes missing type annotations. "
                f"All fields in a Schema must be explicitly annotated. "
                f"Example: 'field_name: int = 0' instead of 'field_name = 0'."
            )

        for field_name, field_type in annotations.items():
            default_val = getattr(cls, field_name, None)

            if _is_array_type(field_type):
                fields[field_name] = Array(
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

                fields[field_name] = default_val
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

                fields[field_name] = default_val
            else:
                raise TypeError(
                    f"Invalid type annotation '{cls!r}' for field '{field_name}' in Schema '{cls}'. "
                    f"Expected a primitive numeric/scalar type (e.g., float, int, bool) or an Array type (e.g., Array[float]), "
                    f"but got an unsupported or non-convertible type."
                )

        cls._schema_specs = fields

    def __init__(self, **fields: Any) -> None:
        extra_keys = set(fields) - set(self._schema_specs)
        if extra_keys:
            raise TypeError(
                f"'{type(self).__name__}' got unexpected keyword argument(s): {list(extra_keys)}"
            )

        for name, spec in self._schema_specs.items():
            setattr(self, name, spec.default(fields.get(name)))

        self.schema = self._create_schema()

    def default(self, value: Optional[Any] = None) -> "Struct":
        if value is None:
            return type(self)()

        if not isinstance(value, type(self)):
            raise TypeError(
                f"Expected field of type '{type(self).__name__}', but got '{type(value).__name__}'."
            )
        return value

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
