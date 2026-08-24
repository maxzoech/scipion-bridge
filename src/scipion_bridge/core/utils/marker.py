from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    get_args,
    get_type_hints,
)

T = TypeVar("T")


class Marker(Generic[T]):
    """Generic base class supporting static typing and runtime annotation introspection."""

    _dtype: Optional[Any] = None
    _generic_cache: Dict[Tuple[Any, ...], type] = {}

    def __init__(self, dtype: Optional[Any] = None, **kwargs: Any) -> None:
        self._dtype = dtype if dtype is not None else self._dtype
        self.name: Optional[str] = None
        self.options = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def dtype(self) -> Optional[Any]:
        return self._dtype

    def __set_name__(self, owner: type, name: str) -> None:
        """Automatically extracts the generic type argument from owner's annotations."""
        self.name = name

        hints = get_type_hints(owner)
        annotations = hints.get(name)
        if not annotations:
            return

        args = get_args(annotations)[0]
        if args and not isinstance(args, TypeVar):
            self._dtype = args
        