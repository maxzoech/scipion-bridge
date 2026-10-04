from abc import ABCMeta
import types
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

T = TypeVar("T")


import importlib
import copyreg
import cloudpickle

def _lookup_marker_class(module_name: str, qualname: str) -> type:
    mod = importlib.import_module(module_name)
    return getattr(mod, qualname)


def _rebuild_generic_marker(origin: type, params: Any) -> type:
    return origin[params]  # type: ignore


def _reduce_marker_meta(obj: type) -> Tuple[Any, Tuple[Any, ...]]:
    origin = get_origin(obj)
    args = get_args(obj)

    if origin is not None and args:
        params = args[0] if len(args) == 1 else args
        return _rebuild_generic_marker, (origin, params)
    
    return _lookup_marker_class, (obj.__module__, obj.__qualname__)


class MarkerMeta(ABCMeta):
    """Metaclass ensuring generic Marker subclasses serialize and deserialize by reference."""

    def __reduce__(cls):
        return _reduce_marker_meta(cls)


# Register with standard Python pickle / copyreg
copyreg.pickle(MarkerMeta, _reduce_marker_meta)

# Hook into cloudpickle class reduction to prevent dynamic class re-creation
_orig_cloudpickle_class_reduce = cloudpickle.cloudpickle._class_reduce


def _scipion_cloudpickle_class_reduce(obj: Any) -> Any:
    if isinstance(type(obj), MarkerMeta) or issubclass(type(obj), MarkerMeta):
        origin = getattr(obj, "__origin__", None)
        args = getattr(obj, "__args__", None)
        if origin is not None and args:
            params = args[0] if len(args) == 1 else args
            return _rebuild_generic_marker, (origin, params)
    return _orig_cloudpickle_class_reduce(obj)


cloudpickle.cloudpickle._class_reduce = _scipion_cloudpickle_class_reduce


class Marker(Generic[T], metaclass=MarkerMeta):
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

        args: Tuple[Any, ...] = getattr(annotations, "__args__", tuple())

        if args and not isinstance(args[0], TypeVar):
            self._dtype = args[0]

    def __class_getitem__(cls, params):
        type_args = params if isinstance(params, tuple) else (params,)
        if any(isinstance(t, str) for t in type_args):
            raise TypeError(
                "Forward declarations using type strings are not supported yet."
            )

        if any(isinstance(t, TypeVar) for t in type_args):
            return types.GenericAlias(cls, type_args)

        cache_key = (cls, params)
        if cache_key in Marker._generic_cache:
            return Marker._generic_cache[cache_key]

        param_names = ", ".join(getattr(t, "__name__", str(t)) for t in type_args)
        new_cls_name = f"{cls.__name__}[{param_names}]"

        new_cls = MarkerMeta(
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
