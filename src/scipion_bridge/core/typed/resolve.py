import sys
import builtins
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
import types
import typing
import networkx as nx
import logging
import warnings
import time
from collections import namedtuple
from collections.abc import Generator
from functools import wraps, partial

from ..utils.func_params import extract_func_params
from .dijkstra import find_shortest_path, PathfindingContainer

from typing import (
    Hashable,
    Tuple,
    Set,
    List,
    Type,
    Any,
    Generic,
    Callable,
    Union,
    Optional,
    get_origin,
    cast,
    Iterator,
    Sized,
    TYPE_CHECKING,
)

from typing_extensions import TypeVar, get_args

if sys.version_info < (3, 11):
    warnings.warn(
        "Local scopes are not supported below Python 3.11; Type resolution behavior might be different.",
        RuntimeWarning,
    )

ResolveStep = namedtuple(
    "ResolveStep", ("cls", "requires_metadata", "requires_slice", "description")
)
ResolveContext = namedtuple(
    "ResolveContext", ("registry", "namespaces", "caller_namespace", "recursion_level")
)

Target = TypeVar("Target")
Origin = TypeVar("Origin")
Intermediate = TypeVar("Intermediate", default=Any)


if TYPE_CHECKING:
    Resolve = Union[Target, Intermediate]
else:

    class Resolve(Generic[Target, Intermediate]):
        pass  # Marker Type


class _DowncastProjector:
    def forward(self, x, **kwargs):
        return x

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)


def _make_projector_class(func: Callable) -> type:
    cached_cls = getattr(func, "_projector_cls", None)
    if cached_cls is not None:
        return cached_cls

    class DynamicFunctionProjector:
        @property
        def __code__(self):
            return getattr(func, "__code__", None)

        @wraps(func)
        def forward(self, *args, **kwargs):
            return func(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    DynamicFunctionProjector.__name__ = getattr(
        func, "__name__", "DynamicFunctionProjector"
    )
    DynamicFunctionProjector.__qualname__ = getattr(
        func, "__qualname__", "DynamicFunctionProjector"
    )
    DynamicFunctionProjector.__module__ = getattr(func, "__module__", __name__)
    DynamicFunctionProjector.__doc__ = getattr(func, "__doc__", None)

    try:
        setattr(func, "_projector_cls", DynamicFunctionProjector)
    except Exception:
        pass

    return DynamicFunctionProjector


def _normalize_resolver(
    resolver: Union[type, Callable],
    requires_metadata: Optional[bool] = None,
    requires_slice: Optional[bool] = None,
) -> Tuple[type, Callable, bool, bool]:
    if inspect.isclass(resolver):
        resolver_cls: Any = resolver
        if hasattr(resolver_cls, "__iter__"):
            raise TypeError(
                f"Resolver class '{resolver_cls.__name__}' must not implement '__iter__'. "
                f"A resolver is a transformer, not an iterator."
            )
        if not hasattr(resolver_cls, "forward"):
            raise TypeError(
                f"Resolver class '{resolver_cls.__name__}' must define a 'forward' method."
            )
        if "__call__" not in resolver_cls.__dict__:
            setattr(resolver_cls, "__call__", resolver_cls.forward)

        method = resolver_cls.forward
    else:
        resolver_cls = _make_projector_class(resolver)
        method = resolver

    sig = inspect.signature(method)
    if requires_metadata is None:
        requires_metadata = "metadata" in sig.parameters
    if requires_slice is None:
        requires_slice = "slice" in sig.parameters

    return resolver_cls, method, requires_metadata, requires_slice


def _find_calling_frame():
    frame = inspect.currentframe()
    while frame is not None:
        if not frame.f_globals["__name__"].startswith(__package__):
            return frame
        else:
            frame = frame.f_back
    else:
        raise RuntimeError("Could not find calling frame. This is a bug.")


def _get_qualname(co_func) -> Optional[str]:
    try:
        return co_func.co_qualname
    except AttributeError:
        return None


class ScopedPathfindingContainer(PathfindingContainer):

    ResolverNode = namedtuple(
        "ResolverNode", ["resolver_fn", "module", "requires_slice"], defaults=[False]
    )

    def __init__(
        self,
        value: Optional[Any],
        previous: Optional[Any],
        weight: int,
        incoming_edge_attributes: Optional[ResolverNode],
        local_scope_name: str,
    ) -> None:
        super().__init__(value, previous, weight)

        self.edge_attributes = incoming_edge_attributes
        self.local_scope_name = local_scope_name

    @property
    def is_local_scope(self):
        assert self.edge_attributes is not None

        return self.edge_attributes.module.startswith(self.local_scope_name)

    @property
    def resolution_priority(self) -> int:
        return 0 if self.is_local_scope else 1

    @property
    def slice_priority(self) -> int:
        assert self.edge_attributes is not None
        return 0 if self.edge_attributes.requires_slice else 1

    def __lt__(self, other):
        assert isinstance(other, ScopedPathfindingContainer)

        assert self.edge_attributes is not None
        assert other.edge_attributes is not None

        if self.weight != other.weight:
            return self.weight < other.weight

        if self.resolution_priority != other.resolution_priority:
            return self.resolution_priority < other.resolution_priority

        if self.slice_priority != other.slice_priority:
            return self.slice_priority < other.slice_priority

        symbol_path = f"{self.edge_attributes.module}.{self.edge_attributes.resolver_fn.__qualname__}"
        other_symbol_path = f"{other.edge_attributes.module}.{other.edge_attributes.resolver_fn.__qualname__}"

        path_length = len(symbol_path.split("."))
        other_path_length = len(other_symbol_path.split("."))

        return other_path_length < path_length


def build_default_container(
    graph: nx.DiGraph,
    value: Type,
    previous: Optional[Type],
    weight: int,
    local_scope_name: str,
):
    if not previous:
        edge_attributes = None
    else:
        attrs = graph.get_edge_data(previous, value)
        edge_attributes = ScopedPathfindingContainer.ResolverNode(
            attrs["resolver"],
            attrs["module"],
            attrs["requires_slice"],
        )

    return ScopedPathfindingContainer(
        value,
        previous,
        weight,
        edge_attributes,
        local_scope_name,
    )


@contextmanager
def resolution_context(
    registry: "Registry", namespace: Set[str], caller_namespace: str
) -> Generator[ResolveContext]:
    parent_ctx = _current_ctx.get()

    if parent_ctx is None:
        new_ctx = ResolveContext(
            registry, namespace, caller_namespace, recursion_level=0
        )

    else:
        new_ctx = ResolveContext(
            parent_ctx.registry,
            parent_ctx.namespaces,
            parent_ctx.caller_namespace,
            recursion_level=parent_ctx.recursion_level + 1,
        )

    token = _current_ctx.set(new_ctx)
    try:
        yield new_ctx
    finally:
        _current_ctx.reset(token)


def estimate_optimal_chunk_size(
    item_nbytes: Optional[int],
    target_bytes: Optional[int] = None,
    default_chunk_size: int = 100,
) -> int:
    """Calculate an optimal batch/chunk size targeting a memory footprint (default 32MB).

    If item_nbytes is None or <= 0, returns default_chunk_size.
    Otherwise returns max(1, target_bytes // item_nbytes).
    """
    if item_nbytes is None or item_nbytes <= 0:
        return default_chunk_size

    effective_target = target_bytes if target_bytes is not None else 32 * 1024 * 1024
    chunk_size = effective_target // item_nbytes
    if chunk_size < 1:
        return 1
    return chunk_size


class ComposedResolver(Generic[Origin, Target]):
    """An independent execution pipeline composed of resolved transformation steps."""

    def __init__(
        self,
        origin: Type[Origin],
        target: Type[Target],
        steps: List[ResolveStep],
    ) -> None:
        self.origin = origin
        self.target = target
        self.steps = steps

        for step in self.steps:
            try:
                step.cls.precompute()
            except AttributeError:
                pass

    def _validate_target(self, out: Any) -> None:
        target_check = get_origin(self.target) or self.target
        if not isinstance(out, target_check):
            resolve_desc = "\n".join([step.description for step in self.steps])
            target_name = getattr(self.target, "__qualname__", str(self.target))

            raise TypeError(
                f"The resolved output with type '{type(out).__qualname__}' did not match target data type '{target_name}'; "
                f"this is most likely a bug in a resolver function. Set log level to INFO debug resolver calls.\n"
                f"Resolvers used:\n{resolve_desc}"
            )

    def __call__(
        self,
        value: Origin,
        *,
        metadata: Optional[Any] = None,
        slice: Optional[builtins.slice] = None,
    ) -> Target:
        if not isinstance(value, self.origin):
            raise TypeError(
                f"The input value did not match origin data type "
                f"(expected {self.origin.__qualname__}, got {type(value).__qualname__})"
            )

        x: Any = value
        current_slice = slice

        for step in self.steps:
            logging.debug(step.description)
            instance = step.cls()

            kwargs: dict[str, Any] = {}
            if step.requires_metadata:
                kwargs["metadata"] = metadata
            if step.requires_slice:
                kwargs["slice"] = current_slice

            x = instance.forward(x, **kwargs)

            # Narrowing Rule: reset slice to None after the first slice-aware step
            if step.requires_slice and current_slice is not None:
                current_slice = None

        if current_slice is not None:
            raise TypeError(
                f"A slice was requested for '{self.origin.__qualname__}' -> '{self.target.__qualname__}', "
                f"but no resolver along the path supports slicing."
            )

        self._validate_target(x)
        return cast(Target, x)

    def iter(
        self,
        value: Origin,
        *,
        chunk_size: Optional[int] = None,
        target_bytes: Optional[int] = None,
        metadata: Optional[Any] = None,
    ) -> Iterator[Target]:
        """Iteratively resolve value in chunks by slicing the first slice-aware step.

        If chunk_size is None, optimal chunk size is estimated based on intermediate.estimated_item_nbytes
        targeting target_bytes (defaults to 32MB).

        Raises:
            TypeError: If value doesn't match origin, if no step in the path supports slicing,
                       or if the intermediate object before the slice step is not Sized.
            ValueError: If chunk_size or target_bytes <= 0.
        """
        if not isinstance(value, self.origin):
            raise TypeError(
                f"The input value did not match origin data type "
                f"(expected {self.origin.__qualname__}, got {type(value).__qualname__})"
            )

        if not any(step.requires_slice for step in self.steps):
            raise TypeError(
                f"Cannot iterate resolution for '{self.origin.__qualname__}' -> '{self.target.__qualname__}': "
                f"no resolver along the path supports slicing."
            )

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        if target_bytes is not None and target_bytes <= 0:
            raise ValueError(f"target_bytes must be positive, got {target_bytes}")

        return self._iter_impl(
            value,
            chunk_size=chunk_size,
            target_bytes=target_bytes,
            metadata=metadata,
        )

    def _iter_impl(
        self,
        value: Origin,
        *,
        chunk_size: Optional[int] = None,
        target_bytes: Optional[int] = None,
        metadata: Optional[Any] = None,
    ) -> Iterator[Target]:
        slice_idx = next(i for i, step in enumerate(self.steps) if step.requires_slice)
        prefix_steps = self.steps[:slice_idx]
        stream_steps = self.steps[slice_idx:]

        x: Any = value
        for step in prefix_steps:
            logging.debug(step.description)
            instance = step.cls()
            kwargs: dict[str, Any] = {}
            if step.requires_metadata:
                kwargs["metadata"] = metadata
            x = instance.forward(x, **kwargs)

        intermediate = x
        if not isinstance(intermediate, Sized):
            raise TypeError(
                f"Cannot iterate over '{type(intermediate).__qualname__}': object does not define __len__."
            )

        total = len(intermediate)
        actual_chunk_size = chunk_size or estimate_optimal_chunk_size(
            getattr(intermediate, "estimated_item_nbytes", None),
            target_bytes=target_bytes,
        )

        stream_instances = [step.cls() for step in stream_steps]

        for k in range(0, total, actual_chunk_size):
            chunk_slice = builtins.slice(k, min(k + actual_chunk_size, total))
            current_slice: Optional[builtins.slice] = chunk_slice
            out: Any = intermediate

            for step, inst in zip(stream_steps, stream_instances):
                kwargs: dict[str, Any] = {}
                if step.requires_metadata:
                    kwargs["metadata"] = metadata
                if step.requires_slice:
                    kwargs["slice"] = current_slice
                    current_slice = None
                out = inst.forward(out, **kwargs)

            self._validate_target(out)
            yield cast(Target, out)


class Registry:

    def __init__(self) -> None:
        self.graph: nx.DiGraph[Type[Hashable]] = nx.DiGraph()

    def get_registered_modules(self) -> Set[str]:
        modules = {v[2] for v in self.graph.edges.data("module")}  # type: ignore
        return modules

    @staticmethod
    def _namespace_from_symbol(
        *, module: str, qualname: Optional[str], strip_last=False
    ):
        if not qualname:
            return module

        path = f"{module}.{qualname}"
        if strip_last:
            parts = path.split(".")[:-1]
            if parts and parts[-1] == "<locals>":
                parts = parts[:-1]
            path = ".".join(parts)

        return path

    def add_resolver(
        self,
        origin: Type[Origin],
        target: Type[Target],
        resolver: Union[type, Callable],
        namespace: Optional[str] = None,
        requires_metadata: Optional[bool] = None,
        requires_slice: Optional[bool] = None,
    ):

        if namespace is None:
            frame = _find_calling_frame()
            module = frame.f_globals["__name__"]

            qualname = _get_qualname(frame.f_code)
            namespace = Registry._namespace_from_symbol(
                module=module, qualname=qualname, strip_last=True
            )
            del frame

        resolver_cls, _, requires_metadata, requires_slice = _normalize_resolver(
            resolver, requires_metadata, requires_slice
        )

        if self.graph.has_edge(origin, target):
            edge = self.graph.edges[(origin, target)]

            if edge["module"] == namespace and resolver_cls is not edge["resolver"]:
                warnings.warn(
                    f"Attempted register a resolver for existing transform '{origin.__qualname__}' -> '{target.__qualname__}' "
                    f"('{edge['resolver'].__qualname__}' vs '{resolver_cls.__qualname__}')",
                    UserWarning,
                )
                return

        def _add_downcasts(subclass: Type):
            for weight_idx, dtype in enumerate(inspect.getmro(subclass)):
                if subclass == dtype:
                    continue

                self.graph.add_edge(
                    subclass,
                    dtype,
                    resolver=_DowncastProjector,
                    weight=weight_idx,
                    module=__package__,
                    requires_metadata=False,
                    requires_slice=False,
                )

        self.graph.add_edge(
            origin,
            target,
            resolver=resolver_cls,
            weight=0,
            module=namespace,
            requires_metadata=requires_metadata,
            requires_slice=requires_slice,
        )

        # Add edges to downcast data
        _add_downcasts(origin)
        _add_downcasts(target)

    def find_resolve_func(
        self,
        namespace: Set[str],
        origin: Type[Origin],
        target: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
        local_scope_name: Optional[str] = None,
    ) -> ComposedResolver[Origin, Target]:
        assert local_scope_name is not None

        def _make_step(edge, data):
            u, v = edge
            resolver_cls = data["resolver"]
            mod = data["module"]
            metadata = data["requires_metadata"]
            requires_slice = data["requires_slice"]

            metadata_desc = " (requires metadata)" if metadata else ""
            slice_desc = " (requires slice)" if requires_slice else ""

            return ResolveStep(
                resolver_cls,
                metadata,
                requires_slice,
                f"{u.__qualname__} -> {v.__qualname__}: {resolver_cls.__qualname__} ({mod}{metadata_desc}{slice_desc})",
            )

        if origin == target:
            return ComposedResolver(origin, target, steps=[])

        selected_edges = [
            (u, v, e)
            for u, v, e in self.graph.edges(data=True)  # type: ignore
            if e["module"] in namespace
        ]
        subgraph = nx.DiGraph(selected_edges)

        # Find the first subclass that is in the graph
        for dtype in origin.__mro__:
            if dtype in subgraph:
                upcast_origin = dtype
                break
        else:
            raise TypeError(
                f"'{origin.__qualname__}' could not be resolved as '{target.__qualname__}'"
            )

        try:
            path = find_shortest_path(
                subgraph,
                upcast_origin,
                target,
                intermediate,
                weight="weight",
                container_builder=partial(
                    build_default_container, local_scope_name=local_scope_name
                ),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
            raise TypeError(
                f"'{origin.__qualname__}' could not be resolved as '{target.__qualname__}'"
            )

        steps = [
            _make_step((u, v), subgraph.get_edge_data(u, v))
            for u, v in zip(path, path[1:])
        ]

        return ComposedResolver(origin, target, steps=steps)

    def _resolve_namespaces(self, value: Any) -> Tuple[Set[str], str, Type]:
        def _find_module(val: Any) -> Optional[str]:
            try:
                if inspect.ismodule(val):
                    return val.__name__
                else:
                    return val.__module__
            except AttributeError:
                return None

        def _expand_namespace(namespace: str, expanded: List[str]) -> Set[str]:
            if not namespace:
                return set(expanded)
            else:
                path = namespace.split(".")
                head, tail = path[0], path[1:]

                next_el = f"{expanded[-1]}.{head}" if expanded else head

                return _expand_namespace(".".join(tail), expanded + [next_el])

        # Find imported modules to construct namespace
        frame = _find_calling_frame()
        calling_module: str = frame.f_globals["__name__"]

        calling_namespace = Registry._namespace_from_symbol(
            module=calling_module, qualname=_get_qualname(frame.f_code)
        )

        origin_type = value if inspect.isclass(value) else type(value)
        associated_namespace = Registry._namespace_from_symbol(
            module=origin_type.__module__, qualname=origin_type.__qualname__
        )
        associated_namespace = _expand_namespace(associated_namespace, [])

        visible_modules = {
            v for v in map(_find_module, frame.f_globals.values()) if v is not None
        }
        del frame

        visible_modules.add(calling_namespace)

        if __package__:
            visible_modules.add(__package__)

        visible_modules = visible_modules.union(associated_namespace)

        # Expand namespaces: "foo.bar.func" -> {foo, foo.bar, foo.bar.func}
        visible_modules = {m for n in visible_modules for m in _expand_namespace(n, [])}

        # Get the namespaces registered in the graph and expand
        registered_modules = self.get_registered_modules()
        registered_modules = {
            m for n in registered_modules for m in _expand_namespace(n, [])
        }

        # The union of the visible modules and registered modules is the available namespace
        namespaces = visible_modules & registered_modules
        return namespaces, calling_namespace, origin_type

    def _lookup_resolver(
        self,
        value: Any,
        astype: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
    ) -> ComposedResolver[Any, Target]:
        namespaces, calling_namespace, origin_type = self._resolve_namespaces(value)

        with resolution_context(self, namespaces, calling_namespace) as context:
            assert context is not None

            intermediate_desc = (
                f" (via '{intermediate.__qualname__}')"
                if intermediate is not None
                else ""
            )

            namespaces_ctx = [f"'{n}'" for n in context.namespaces]
            namespaces_desc = ", ".join(namespaces_ctx).rstrip()

            indent = " " * 4 * context.recursion_level

            logging.info(
                f"{indent}Resolve '{origin_type.__qualname__}' -> '{astype.__qualname__}'"
                f"{intermediate_desc} (caller in '{context.caller_namespace}')",
            )

            logging.debug(f"{indent}Namespace: {namespaces_desc}")

            return self.find_resolve_func(
                context.namespaces,
                origin_type,
                astype,
                intermediate,
                context.caller_namespace,
            )

    def find_resolver(
        self,
        origin: Union[Type[Origin], Origin],
        target: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
    ) -> ComposedResolver[Origin, Target]:
        """Precompute and return a ComposedResolver for the given origin and target types."""
        return self._lookup_resolver(origin, target, intermediate)

    def resolve(
        self,
        value,
        astype: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
        metadata: Optional[Any] = None,
        slice: Optional[builtins.slice] = None,
    ) -> Target:
        start = time.time()
        namespaces, calling_namespace, origin_type = self._resolve_namespaces(value)

        with resolution_context(self, namespaces, calling_namespace) as context:
            assert context is not None

            intermediate_desc = (
                f" (via '{intermediate.__qualname__}')"
                if intermediate is not None
                else ""
            )

            namespaces_ctx = [f"'{n}'" for n in context.namespaces]
            namespaces_desc = ", ".join(namespaces_ctx).rstrip()

            indent = " " * 4 * context.recursion_level

            logging.info(
                f"{indent}Resolve '{origin_type.__qualname__}' -> '{astype.__qualname__}'"
                f"{intermediate_desc} (caller in '{context.caller_namespace}')",
            )

            logging.debug(f"{indent}Namespace: {namespaces_desc}")

            resolve_fn = self.find_resolve_func(
                context.namespaces,
                origin_type,
                astype,
                intermediate,
                context.caller_namespace,
            )

            end_search = time.time()

            search_time = end_search - start
            search_time_ms = search_time * 1_000

            resolved = resolve_fn(value, metadata=metadata, slice=slice)
            end = time.time()

        total = end - start
        total_ms = total * 1_000
        search_percentage = int((search_time / total) * 100) if total > 0 else 0

        logging.info(
            f"Resolving from '{type(value).__qualname__}' to '{astype.__qualname__}' took {total_ms:2f}ms "
            f"({search_time_ms:2f}ms ({search_percentage}%) path finding)"
        )

        return resolved

    def resolve_iter(
        self,
        value,
        astype: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
        chunk_size: Optional[int] = None,
        target_bytes: Optional[int] = None,
        metadata: Optional[Any] = None,
    ) -> Iterator[Target]:
        resolve_fn = self._lookup_resolver(value, astype, intermediate)
        return resolve_fn.iter(
            value,
            chunk_size=chunk_size,
            target_bytes=target_bytes,
            metadata=metadata,
        )

    def lift_resolvers(self, origin_module_name: str, target_module_name: str):
        # Assert that the target module is actually imports the module from which
        # we want to lift the resolvers from.
        #
        # For example, we can declare some resolvers in scipion_bridge.typed.common
        # and then lift those into scipion_bridge, but we cannot lift them into
        # scipion_bridge.proxy. This is important as the resolvers are still
        # available even if the user only imports scipion_bridge.typed.common as
        # the parent module is always visible when resolving types
        assert origin_module_name.startswith(target_module_name)

        for _, _, attr in self.graph.edges(data=True):  # type: ignore
            if attr["module"] == origin_module_name:
                attr["module"] = target_module_name

    def _plot_graph(self, G=None):  # pragma: no cover
        import networkx as nx
        import matplotlib.pyplot as plt

        if G is None:
            G = self.graph

        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=250)
        nx.draw_networkx_edges(G, pos, width=1)

        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        edge_weights = nx.get_edge_attributes(G, "weight")
        edge_modules = nx.get_edge_attributes(G, "module")

        edge_labels = {}
        for k in edge_weights.keys():
            edge_labels[k] = f"{edge_modules[k]} ({edge_weights[k]})"

        nx.draw_networkx_edge_labels(G, pos, edge_weights)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


DEFAULT_REGISTRY = Registry()
_current_ctx: ContextVar[Optional[ResolveContext]] = ContextVar(
    "_current_ctx", default=None
)


def current_registry() -> Registry:
    ctx = _current_ctx.get()
    return ctx.registry if ctx is not None else DEFAULT_REGISTRY


def resolver(target: Any) -> Any:
    resolver_cls, method, requires_metadata, requires_slice = _normalize_resolver(
        target
    )

    hints = typing.get_type_hints(method)
    in_dtype = hints["value"]
    out_dtype = hints["return"]

    qualname = (
        getattr(target, "__qualname__", target.__name__)
        if inspect.isclass(target)
        else _get_qualname(getattr(target, "__code__", None))
        or getattr(target, "__qualname__", getattr(target, "__name__", None))
    )

    namespace = Registry._namespace_from_symbol(
        module=target.__module__,
        qualname=qualname,
        strip_last=True,
    )

    current_registry().add_resolver(
        in_dtype,
        out_dtype,
        resolver_cls,
        namespace=namespace,
        requires_metadata=requires_metadata,
        requires_slice=requires_slice,
    )

    return target if inspect.isclass(target) else resolver_cls()


def resolve(
    value,
    astype: Type[Target],
    intermediate: Optional[Type[Intermediate]] = None,
    metadata: Optional[Any] = None,
    slice: Optional[builtins.slice] = None,
) -> Target:
    return current_registry().resolve(
        value,
        astype=astype,
        intermediate=intermediate,
        metadata=metadata,
        slice=slice,
    )


def resolve_iter(
    value,
    astype: Type[Target],
    intermediate: Optional[Type[Intermediate]] = None,
    chunk_size: Optional[int] = None,
    target_bytes: Optional[int] = None,
    metadata: Optional[Any] = None,
) -> Iterator[Target]:
    return current_registry().resolve_iter(
        value,
        astype=astype,
        intermediate=intermediate,
        chunk_size=chunk_size,
        target_bytes=target_bytes,
        metadata=metadata,
    )


def find_resolver(
    origin: Union[Type[Origin], Origin],
    target: Type[Target],
    intermediate: Optional[Type[Intermediate]] = None,
) -> ComposedResolver[Origin, Target]:
    """Precompute and return a ComposedResolver from the current registry."""
    return current_registry().find_resolver(
        origin,
        target,
        intermediate=intermediate,
    )


def lift_resolvers(
    *modules: types.ModuleType, target: Optional[types.ModuleType] = None
):
    if target is None:
        # Get the calling module
        frame = inspect.currentframe()
        assert frame is not None
        caller_frame = frame.f_back
        assert caller_frame is not None
        target_module_name = caller_frame.f_globals["__name__"]
        assert isinstance(target_module_name, str)
    else:
        target_module_name = target.__name__

    reg = current_registry()
    for module in modules:
        reg.lift_resolvers(module.__name__, target_module_name)


def resolve_params(f: Callable):

    signature = inspect.signature(f)

    def _resolve_arg(arg: Tuple[inspect.Parameter, Any]):
        param, value = arg
        if param.annotation is not None and get_origin(param.annotation) == Resolve:
            args = get_args(param.annotation)
            if len(args) == 1:
                args = tuple([args[0], Any])

            target, constraint = args

            constraint = None if constraint == Any else constraint
            value = current_registry().resolve(
                value, astype=target, intermediate=constraint
            )

        return param, value

    @wraps(f)
    def wrapper(*args, **kwargs):
        func_params = extract_func_params(args, kwargs, signature)

        n_positional = len(args)
        positional = list(func_params.items())[:n_positional]
        keyword = list(func_params.items())[n_positional:]

        positional = [_resolve_arg(a) for a in positional]
        args = tuple(v for _, v in positional)

        keyword = [_resolve_arg(a) for a in keyword]
        kwargs = {k.name: v for k, v in keyword}

        return f(*args, **kwargs)

    return wrapper
