"""Generic marker classes that keep their identity across processes.

Every specialization ``Origin[Args]`` of a :class:`Marker` is a real class. To
make such classes picklable *by name* (so that stdlib ``pickle``,
``cloudpickle``, ``ray.cloudpickle`` and others send a reference instead of
re-creating the class), each specialization gets a ``__qualname__`` that fully
encodes its origin and type arguments without using dots, e.g.::

    Set[scipion_bridge/single_particle/particle:Particle]

The specialization is stored under that name on the module of its origin, and
every module defining a marker class gets a module-level ``__getattr__``
(PEP 562) that rebuilds a specialization from its encoded name. A process that
unpickles a specialization it has never built therefore resolves it through
``Origin[Args]`` and the shared ``_generic_cache``.

Specializations whose origin or arguments cannot be imported by name (classes
defined in ``__main__`` or in a local scope, non-class arguments) keep their
readable name and are serialized by value, as before.
"""

from abc import ABCMeta
from functools import reduce
import importlib
import sys
import types
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    get_type_hints,
)

T = TypeVar("T")

_PATH_SEPARATOR = "/"
_REF_SEPARATOR = ":"


def _is_importable(t: Any) -> bool:
    """Whether ``t`` is a class that can be imported by module and qualname."""
    return (
        isinstance(t, type)
        and t.__module__ != "__main__"
        and "<locals>" not in t.__qualname__
        and t.__dict__.get("_encoded_by_name", True)
    )


def _encode_ref(t: type) -> str:
    """Encode an importable class as ``module/path:Qual/Name``."""
    module = t.__module__.replace(".", _PATH_SEPARATOR)
    qualname = t.__qualname__.replace(".", _PATH_SEPARATOR)
    return f"{module}{_REF_SEPARATOR}{qualname}"


def _encode_specialization(origin: type, type_args: Tuple[Any, ...]) -> Optional[str]:
    """Return the dot-free qualname of ``origin[type_args]``, or None if not encodable."""
    if not all(_is_importable(t) for t in (origin, *type_args)):
        return None

    origin_name = origin.__qualname__.replace(".", _PATH_SEPARATOR)
    args = ",".join(_encode_ref(t) for t in type_args)
    return f"{origin_name}[{args}]"


def _split_top_level(encoded_args: str) -> List[str]:
    """Split comma-separated references, ignoring commas inside brackets."""
    parts: List[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(encoded_args):
        match char:
            case "[":
                depth += 1
            case "]":
                depth -= 1
            case "," if depth == 0:
                parts.append(encoded_args[start:i])
                start = i + 1
            case _:
                pass
    parts.append(encoded_args[start:])
    return parts


def _resolve_ref(ref: str) -> Any:
    """Import the class referenced by ``module/path:Qual/Name``."""
    module_path, qualname = ref.split(_REF_SEPARATOR, 1)
    module = importlib.import_module(module_path.replace(_PATH_SEPARATOR, "."))

    if "[" in qualname:
        # A specialization: its encoded name is an attribute of the module.
        return getattr(module, qualname)
    else:
        return reduce(getattr, qualname.split(_PATH_SEPARATOR), module)


class _SpecializationResolver:
    """Module-level ``__getattr__`` rebuilding marker specializations by name."""

    def __init__(self, module: types.ModuleType) -> None:
        self._module = module

    def __call__(self, name: str) -> type:
        origin_name, _, encoded_args = name[:-1].partition("[")
        refs = _split_top_level(encoded_args)

        # Only encoded names (every argument is a ``module:Qual`` reference) can
        # be rebuilt. Readable names of by-value specializations, which tools
        # like cloudpickle look up, must be reported as missing.
        is_encoded = (
            "[" in name
            and name.endswith("]")
            and all(_REF_SEPARATOR in ref for ref in refs)
        )
        if not is_encoded:
            raise AttributeError(
                f"module '{self._module.__name__}' has no attribute '{name}'",
            )

        origin = reduce(getattr, origin_name.split(_PATH_SEPARATOR), self._module)
        type_args = tuple(_resolve_ref(ref) for ref in refs)
        params = type_args[0] if len(type_args) == 1 else type_args

        return origin[params]  # type: ignore


def _install_specialization_resolver(module: types.ModuleType) -> None:
    match module.__dict__.get("__getattr__"):
        case None:
            module.__getattr__ = _SpecializationResolver(module)  # type: ignore[attr-defined]
        case _SpecializationResolver():
            pass
        case existing:
            raise TypeError(
                f"Module '{module.__name__}' defines a module-level __getattr__ ({existing!r}), "
                "which conflicts with the resolver required to unpickle Marker specializations.",
            )


class MarkerMeta(ABCMeta):
    """Metaclass making Marker specializations resolvable by module and qualname."""

    def __init__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        _install_specialization_resolver(sys.modules[cls.__module__])

    def __repr__(cls) -> str:
        return f"<class '{cls.__module__}.{cls.__name__}'>"


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
        readable_name = f"{cls.__name__}[{param_names}]"
        encoded_name = _encode_specialization(cls, type_args)

        new_cls = MarkerMeta(
            readable_name,
            (cls,),
            {
                "__module__": cls.__module__,
                "__qualname__": encoded_name or readable_name,
                "__origin__": cls,
                "__args__": type_args,
                "_dtype": type_args[0],
                "_encoded_by_name": encoded_name is not None,
            },
        )

        Marker._generic_cache[cache_key] = new_cls
        if encoded_name is not None:
            setattr(sys.modules[cls.__module__], encoded_name, new_cls)

        return new_cls
