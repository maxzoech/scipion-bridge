import sys
import inspect
import textwrap
import types
import networkx as nx
import logging
import warnings
import time
from collections import namedtuple
from functools import wraps, partial

from ..utils.func_params import extract_func_params
from .dijkstra import find_shortest_path, PathfindingContainer

from typing import (
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
    TYPE_CHECKING,
)

from typing_extensions import TypeVar, get_args

if sys.version_info < (3, 11):
    warnings.warn(
        "Local scopes are not supported below Python 3.11; Type resolution behavior might be different.",
        RuntimeWarning,
    )

ResolveStep = namedtuple("ResolveStep", ("func", "description"))
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


def _downcast(x):
    return x


def _passthrough(x):
    return x


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

    ResolverNode = namedtuple("ResolverNode", ["resolver_fn", "module"])

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
    def resolution_priority(self):
        if self.is_local_scope:
            return 0  # Higher priority
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, ScopedPathfindingContainer)

        assert self.edge_attributes is not None
        assert other.edge_attributes is not None

        if not self.weight == other.weight:
            return self.weight < other.weight
        else:
            if not self.resolution_priority == other.resolution_priority:
                return self.resolution_priority < other.resolution_priority
            else:

                symbol_path = f"{self.edge_attributes.module}.{self.edge_attributes.resolver_fn.__qualname__}"
                other_symbol_path = f"{other.edge_attributes.module}.{other.edge_attributes.resolver_fn.__qualname__}"

                path_length = len(symbol_path.split("."))
                other_path_length = len(other_symbol_path.split("."))

                if (
                    other_path_length == path_length
                    and self.edge_attributes.resolver_fn
                    is not other.edge_attributes.resolver_fn
                ):
                    pass

                    # warnings.warn(
                    #     f"Found ambiguous resolvers {symbol_path} and {other_symbol_path} during type resolution.",
                    # )

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
            attrs["resolver"], attrs["module"]
        )
    return ScopedPathfindingContainer(
        value,
        previous,
        weight,
        edge_attributes,
        local_scope_name,
    )


class resolution_context:

    def __init__(
        self, registry: "Registry", namespace: Set[str], caller_namespace: str
    ):

        global CURRENT_CTX
        self._old_context: Optional[ResolveContext] = CURRENT_CTX

        if self._old_context is None:
            CURRENT_CTX = ResolveContext(
                registry, namespace, caller_namespace, recursion_level=0
            )
        else:
            CURRENT_CTX = ResolveContext(
                self._old_context.registry,
                self._old_context.namespaces,
                self._old_context.caller_namespace,
                recursion_level=self._old_context.recursion_level + 1,
            )

    def __enter__(self):
        global CURRENT_CTX

        return CURRENT_CTX

    def __exit__(self, *args, **kws):
        global CURRENT_CTX
        CURRENT_CTX = self._old_context


class Registry:

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

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
            path = path.split(".")[:-1]
            if path[-1] == "<locals>":
                path = path[:-1]
            path = ".".join(path)

        return path  #

    def add_resolver(
        self,
        origin: Type[Origin],
        target: Type[Origin],
        resolver: Callable,
        namespace: Optional[str] = None,
    ):

        if namespace is None:
            frame = _find_calling_frame()
            module = frame.f_globals["__name__"]

            namespace = Registry._namespace_from_symbol(
                module=module, qualname=_get_qualname(frame.f_code), strip_last=True
            )

        if self.graph.has_edge(origin, target):
            edge = self.graph.edges[(origin, target)]

            if edge["module"] == namespace and resolver is not edge["resolver"]:
                warnings.warn(
                    f"Attempted register a resolver for existing transform '{origin.__qualname__}' -> '{target.__qualname__}' ('{edge['resolver'].__qualname__}' vs '{resolver.__qualname__}')",
                    UserWarning,
                )
                return

        def _add_downcasts(subclass: Type):
            for weight, dtype in enumerate(subclass.__mro__):
                if subclass == dtype:
                    continue

                self.graph.add_edge(
                    subclass,
                    dtype,
                    resolver=_downcast,
                    weight=weight,
                    module=__package__,
                )

                # print(f"Add downcast: {subclass} -> {dtype} in {__package__}, {weight}")

        self.graph.add_edge(
            origin, target, resolver=resolver, weight=0, module=namespace
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
    ):
        assert local_scope_name is not None

        def _make_step(edge, data):
            u, v = edge
            fn = data["resolver"]
            mod = data["module"]

            return ResolveStep(
                fn,
                f"{u.__qualname__} -> {v.__qualname__}: {fn.__qualname__} ({mod})",
            )

        if origin == target:
            return _passthrough

        selected_edges = [
            (u, v, e)
            for u, v, e in self.graph.edges(data=True)
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

        def resolver_fn(value: Origin) -> Target:
            if not isinstance(value, origin):
                raise TypeError("The input value for did not match origin data type")

            x = value
            for step in steps:
                logging.debug(step.description)

                x = step.func(x)  # type: ignore

            if not isinstance(x, target):
                resolve_desc = "\n".join([step.description for step in steps])

                raise TypeError(
                    f"The resolved output with type '{type(x).__qualname__}' did not match target data type '{target.__qualname__}'; this is most likely a bug in a resolver function. Set log level to INFO debug resolver calls.\nResolvers used:\n{resolve_desc}"
                )

            return x

        return resolver_fn

    def resolve(
        self,
        value,
        astype: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
    ) -> Target:

        def _find_module(value: Any) -> Optional[str]:
            try:
                if inspect.ismodule(value):
                    return value.__name__
                else:
                    return value.__module__
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

        start = time.time()

        # Find imported modules to construct namespace
        frame = _find_calling_frame()
        calling_module: str = frame.f_globals["__name__"]

        calling_namespace = Registry._namespace_from_symbol(
            module=calling_module, qualname=_get_qualname(frame.f_code)
        )

        # The associated namespace is the namespace where the value we want to
        # resolve.
        # This useful when we use a symbol without directly importing the module
        # where it was declared, e.g.
        # import scipion_bridge
        # ...
        # value.typed(scipion_bridge.typed.volume.SpiderFile) <- we never imported Spider file
        associated_namespace = Registry._namespace_from_symbol(
            module=type(value).__module__, qualname=type(value).__qualname__
        )
        associated_namespace = _expand_namespace(associated_namespace, [])

        visible_modules = {
            v for v in map(_find_module, frame.f_globals.values()) if v is not None
        }

        visible_modules.add(calling_namespace)
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
                f"{indent}Resolve '{type(value).__qualname__}' -> '{astype.__qualname__}'{intermediate_desc} (caller in '{context.caller_namespace}')",
            )

            logging.debug(f"{indent}Namespace: {namespaces_desc}")

            resolve_fn = self.find_resolve_func(
                context.namespaces,
                type(value),
                astype,
                intermediate,
                context.caller_namespace,
            )

            end_search = time.time()

            search_time = end_search - start
            search_time_ms = search_time * 1_000

            resolved = resolve_fn(value)
            end = time.time()

        total = end - start
        total_ms = total * 1_000
        search_percentage = int((search_time / total) * 100)

        logging.info(
            f"Resolving from '{type(value).__qualname__}' to '{astype.__qualname__}' took {total_ms:2f}ms ({search_time_ms:2f}ms ({search_percentage}%) path finding)"
        )

        return resolved
        
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

        for _, _, attr in self.graph.edges(data=True):
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
CURRENT_CTX = None


def current_registry() -> Registry:
    global CURRENT_CTX

    if CURRENT_CTX:
        return CURRENT_CTX.registry
    else:
        return DEFAULT_REGISTRY


def resolver(f):

    # TODO: Input validation
    in_dtype = f.__annotations__["value"]
    out_dtype = f.__annotations__["return"]

    namespace = Registry._namespace_from_symbol(
        module=f.__module__,
        qualname=_get_qualname(f.__code__),  # f.__qualname__,
        strip_last=True,
    )

    current_registry().add_resolver(in_dtype, out_dtype, f, namespace)

    return f

def resolve(
        value,
        astype: Type[Target],
        intermediate: Optional[Type[Intermediate]] = None,
    ) -> Target:
    return current_registry().resolve(value, astype=astype, intermediate=intermediate)

def lift_resolvers(*modules: types.ModuleType, target: Optional[types.ModuleType] = None):
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

        args = list(func_params.items())[: len(args)]
        kwargs = list(func_params.items())[len(args) :]

        args = [_resolve_arg(a) for a in args]
        args = [v for _, v in args]

        kwargs = [_resolve_arg(a) for a in kwargs]
        kwargs = {k.name: v for k, v in kwargs}

        return f(*args, **kwargs)

    return wrapper
