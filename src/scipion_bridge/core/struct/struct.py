import numpy as np

from .schema import Schema
from typing import Type, TypeVar, Tuple, Union, Any, Optional, get_origin

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

class Array(Marker[T]):

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


    def validate(self, other: "Array") -> None:
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

        for axis, (expected_dim, incoming_dim) in enumerate(zip(self.shape, other.shape)):
            if expected_dim is not None and incoming_dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch at axis {axis}: static dimension {expected_dim} "
                    f"cannot be overwritten by {incoming_dim} "
                    f"(target shape={list(self.shape)}, incoming shape={list(other.shape)})."
                )

class Struct(SchemaArrayStorage):

    _schema_specs: dict[str, Array]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls_name = cls.__name__
        annotations = getattr(cls, "__annotations__", {})
        fields: dict[str, Array] = {}

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

            else:
                raise TypeError(
                    f"Invalid type annotation '{cls!r}' for field '{field_name}' in Schema '{cls}'. "
                    f"Expected a primitive numeric/scalar type (e.g., float, int, bool) or an Array type (e.g., Array[float]), "
                    f"but got an unsupported or non-convertible type."
                )

        cls._schema_specs = fields


    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            schema_spec = self._schema_specs[name]

            if not isinstance(value, Array):
                raise TypeError(
                    f"Field '{name}' expects an Array marker, but got '{type(value).__name__}'."
                )
            
            schema_spec.validate(value)

            setattr(self, name, value)
