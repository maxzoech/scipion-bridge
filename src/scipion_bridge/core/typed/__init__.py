"""Type resolution between data types, and proxies passing data as files.

The function ``resolve`` is not exported here: it would hide the module
``scipion_bridge.core.typed.resolve``. Use ``scipion_bridge.resolve``.
"""

from .proxy import Output, Proxy, ProxyGroup, ResolveProxy, namedproxy, proxify
from .resolve import (
    ComposedResolver,
    Registry,
    Resolve,
    current_registry,
    estimate_optimal_chunk_size,
    find_resolver,
    lift_resolvers,
    resolve_iter,
    resolve_params,
    resolver,
)

# Registration: the resolvers between core types.
from . import core_resolvers  # noqa: F401

__all__ = [
    # Resolution
    "resolver",
    "resolve_iter",
    "resolve_params",
    "find_resolver",
    "lift_resolvers",
    "Resolve",
    "ComposedResolver",
    "estimate_optimal_chunk_size",
    "current_registry",
    "Registry",
    # Proxies
    "Proxy",
    "ProxyGroup",
    "Output",
    "ResolveProxy",
    "namedproxy",
    "proxify",
]
