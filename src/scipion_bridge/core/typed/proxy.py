import os
import inspect
from pathlib import Path, PurePath
import logging
import warnings
from enum import Enum
from functools import partial, wraps
import shutil

from dependency_injector.wiring import Provide, inject
from ..environment.container import Container
from ..environment.temp_files import TemporaryFilesProvider
from ..utils.arc import manager as arc_manager
from ..utils.func_params import extract_func_params

from ..utils.arc import manager as arc_manager

from .resolve import current_registry, resolve_params, resolver, Registry
from typing import Optional, Generic, Protocol, Type, Union, TYPE_CHECKING, Any, cast
from typing_extensions import TypeAlias, TypeVar, get_args, get_origin


Casted = TypeVar("Casted", bound="Proxy")
T = TypeVar("T")

Intermediate = TypeVar("Intermediate", default=Any)
Origin = TypeVar("Origin", default=Any)


class FuncParam:
    def __init__(
        self, str_rep: str, dtype: Optional[Type[T]] = None, managed_proxy=False
    ) -> None:
        self.str_rep = str_rep
        self.dtype = dtype
        self.managed_proxy = managed_proxy

        if managed_proxy:
            arc_manager.add_reference(Path(str_rep))

    def __repr__(self) -> str:
        return f"{FuncParam.__name__} ({self.str_rep}, dtype={self.dtype}, managed_proxy={self.managed_proxy})"

    def __del__(self):
        if self.managed_proxy:
            arc_manager.remove_reference(Path(self.str_rep))


class ProxyMetaclass(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)

        def resolve_path_proxy(value: Path):
            proxy_ext: str = x.extension()  # type: ignore
            path_ext = value.suffix

            if not proxy_ext == path_ext:
                raise TypeError(
                    f"The file extension did not match the proxy (Expected {proxy_ext} but received {path_ext})"
                )

            return x(value, managed=False)

        # Put this in the scipion_bridge namespace so the user can shadow this
        # default resolver with their own if needed
        default_resolver_namespace = Registry._namespace_from_symbol(
            module=str(__package__),
            qualname=resolve_path_proxy.__name__,
            strip_last=True,
        )

        current_registry().add_resolver(
            Path,
            x,
            resolver=resolve_path_proxy,
            namespace=default_resolver_namespace,
        )

        return x


class Proxy(metaclass=ProxyMetaclass):

    def __init__(self, path: os.PathLike, managed=False, *args, **kwargs):

        self.path = Path(path)
        self.managed = managed

        if self.managed == True:
            arc_manager.add_reference(self.path)

        super().__init__(*args, **kwargs)

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return None

    @classmethod
    def extension(cls) -> Optional[str]:
        ext = cls.file_ext()
        if not ext:
            return ext
        else:
            assert ext.startswith("."), "File extension must start with ."
            return ext

    @classmethod
    def new_temporary_proxy(cls) -> "Proxy":
        file_ext = cls.file_ext()
        file_ext = file_ext if file_ext is not None else ""

        temp_file = arc_manager.new_managed_file(file_ext)

        return cls(temp_file, managed=True)

    def typed(self, *, astype: Type[Casted], copy_data=True) -> Casted:
        if self.file_ext() is not None:
            raise TypeError(
                f"Cannot add type to proxy with existing type {self.file_ext()}"
            )

        assert issubclass(astype, Proxy)

        new_ext = astype.file_ext()
        assert isinstance(new_ext, str)

        new_path = self.path.with_name(f"{self.path.name}{new_ext}")
        if copy_data:
            shutil.copy(str(self.path), str(new_path))

        new_proxy = astype(
            new_path,
            managed=self.managed,
        )

        return new_proxy

    @inject
    def __del__(
        self,
        temp_file_provider: TemporaryFilesProvider = Provide[
            Container.temp_file_provider
        ],
    ):

        try:
            if self.managed == True:
                arc_manager.remove_reference(self.path)

        except Exception as e:
            logging.warning(f"Failed to delete file at {self.path}: {e}")
            pass  # Fail silently

    def __str__(self):
        is_owned = "managed" if self.managed else "unmanaged"
        return f"<{self.__class__.__name__} for {self.path} ({is_owned})>"


class Output(Generic[T]):
    def __init__(self, dtype: Type[T]) -> None:
        assert issubclass(dtype, Proxy)
        self.dtype = dtype

        current_registry().add_resolver(Output, dtype, resolver=resolve_output_to_proxy)

class ProxyProtocol(Protocol):
    @classmethod
    def file_ext(cls) -> Optional[str]: ...

def namedproxy(typename: str, *, file_ext: str) -> Type[ProxyProtocol]:
    if not typename.isidentifier():
        raise ValueError("The typename must be a valid identifier")

    if not file_ext.startswith("."):
        raise ValueError("The file extension must start with a .")
    
    _ext = file_ext

    class ProxySubclass(Proxy):
        
        @classmethod
        def file_ext(cls) -> Optional[str]:
            return _ext

    ProxySubclass.__name__ = typename
    ProxySubclass.__qualname__ = typename
    return ProxySubclass


if TYPE_CHECKING:

    # class ProxyParam():
    #     pass  # Marker Type

    ProxyParam = Union[Output[Intermediate], Intermediate, Origin]
else:

    class ProxyParam(Generic[Intermediate, Origin]):
        pass  # Marker Type


def proxify(f):

    signature = inspect.signature(f)

    def _proxy_from_func_param(param: FuncParam):
        cls: Type[Proxy] = cast(
            Type[Proxy],
            param.dtype if param.dtype is not None else Proxy
        )  # Create new untyped proxy if we only pass a path here
        assert issubclass(cls, Proxy)

        return cls(Path(param.str_rep), managed=param.managed_proxy)

    def _resolve_proxy_arg(value, param: inspect.Parameter) -> FuncParam:
        intermediate = None

        if param.annotation is not None and get_origin(param.annotation) == ProxyParam:
            args = get_args(param.annotation)
            if args:
                arg = args[0]
                intermediate = arg if not arg == Any else None

        return current_registry().resolve(
            value, astype=FuncParam, intermediate=intermediate
        )

    @wraps(f)
    def wrapped(*args, **kwargs):

        func_args = extract_func_params(args, kwargs, signature)

        should_return = {
            k: isinstance(v.default, Output) for k, v in signature.parameters.items()
        }

        resolved = [
            (param.name, _resolve_proxy_arg(v, param)) for param, v in func_args.items()
        ]

        resolved_args = [v.str_rep for _, v in resolved[: len(args)]]
        resolved_kwargs = {k: v.str_rep for k, v in resolved[len(args) :]}

        out_val = f(*resolved_args, **resolved_kwargs)

        return_vals = [
            _proxy_from_func_param(v) for k, v in resolved if should_return[k]
        ]

        try:
            outputs = out_val if isinstance(out_val, tuple) else tuple([out_val])
            outputs_are_proxies = all(isinstance(o, Proxy) for o in outputs)
        except TypeError as e:
            outputs_are_proxies = False

        if not (out_val == 0 or out_val == None or outputs_are_proxies):
            warnings.warn(
                f"Wrapped function returns non-zero value; the value '{out_val}' will be discarded",
                UserWarning,
            )

        if len(return_vals) == 0:
            return out_val
        elif len(return_vals) == 1:
            return return_vals[0]
        else:
            return tuple(return_vals)

    return wrapped


@resolver
def resolve_path_to_func_param(value: Path) -> FuncParam:
    return FuncParam(str(value))


@resolver
def resolve_str_to_func_param(value: str) -> FuncParam:
    return FuncParam(value)


@resolver
def resolve_proxy_to_func_param(value: Proxy) -> FuncParam:
    return FuncParam(str(value.path), type(value), managed_proxy=value.managed)


@resolver
def resolve_path_to_untyped_proxy(value: Path) -> Proxy:
    return Proxy(value)


def resolve_output_to_proxy(
    value: Output,
) -> Proxy:

    new_proxy = value.dtype.new_temporary_proxy()

    assert isinstance(new_proxy, Proxy)
    return new_proxy
