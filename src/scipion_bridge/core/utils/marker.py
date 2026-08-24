import types
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

        args = getattr(annotations, "__args__", tuple())

        if args and not isinstance(args[0], TypeVar):
            self._dtype = args[0]

    def __class_getitem__(cls, params):
            type_args = params if isinstance(params, tuple) else (params,)
            if any(isinstance(t, TypeVar) for t in type_args):
                    return types.GenericAlias(cls, type_args)
   
            cache_key = (cls, params)
            if cache_key in Marker._generic_cache:
                return Marker._generic_cache[cache_key]

            param_names = ", ".join(getattr(t, "__name__", str(t)) for t in type_args)
            new_cls_name = f"{cls.__name__}[{param_names}]"

            new_cls = type(
                new_cls_name,
                (cls,),
                {
                    "__module__": cls.__module__,
                    "__origin__": cls,
                    "__args__": type_args,
                    "_dtype": type_args[0],
                },
            )

            Marker._generic_cache[cache_key] = new_cls
            return new_cls
