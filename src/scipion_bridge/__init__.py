"""scipion-bridge: columnar types, protocols and streaming pipelines for Scipion.

The user API, imported as::

    import scipion_bridge as B

Domain types and proxies of single particle analysis are in
``scipion_bridge.single_particle``; backends are in ``scipion_bridge.backend``.
"""

# Structs
from .core.struct import Arg, Array, Collection, Dim, Set, Struct, concat

# Protocols and their compute resources
from .core.protocol import (
    ChainedProtocol,
    Field,
    Input,
    Protocol,
    Resource,
    SealedProtocolError,
    ValidationError,
    resources,
)
from .core.environment import ComputeResources, Domain, ResourceScope, TaskType

# Streaming
from .core.streaming import (
    FLUSH,
    CallbackSinkWriter,
    FlushSignal,
    Op,
    Pipeline,
    Sink,
    SinkWriter,
    Source,
)

# Type resolution and proxies
from .core.typed import (
    ComposedResolver,
    Output,
    Proxy,
    ProxyGroup,
    Resolve,
    ResolveProxy,
    estimate_optimal_chunk_size,
    find_resolver,
    lift_resolvers,
    namedproxy,
    proxify,
    resolve_iter,
    resolve_params,
    resolver,
)
from .core.typed.resolve import resolve

# Shell commands
from .core.utils import shell_command

# Registration: the resolvers of the core types, of proxies and of single
# particle analysis resolve from `scipion_bridge`.
from .core.typed import core_resolvers, proxy
from . import single_particle

lift_resolvers(core_resolvers, proxy, single_particle)

__all__ = [
    # Structs
    "Struct",
    "Set",
    "Collection",
    "Array",
    "Arg",
    "Dim",
    "concat",
    # Protocols and their compute resources
    "Protocol",
    "ChainedProtocol",
    "Input",
    "Field",
    "Resource",
    "ResourceScope",
    "resources",
    "ComputeResources",
    "TaskType",
    "Domain",
    "ValidationError",
    "SealedProtocolError",
    # Streaming
    "Op",
    "Source",
    "Sink",
    "Pipeline",
    "SinkWriter",
    "CallbackSinkWriter",
    "FlushSignal",
    "FLUSH",
    # Type resolution and proxies
    "resolver",
    "resolve",
    "resolve_iter",
    "resolve_params",
    "find_resolver",
    "lift_resolvers",
    "Resolve",
    "ComposedResolver",
    "estimate_optimal_chunk_size",
    "Proxy",
    "ProxyGroup",
    "Output",
    "ResolveProxy",
    "namedproxy",
    "proxify",
    # Shell commands
    "shell_command",
]
