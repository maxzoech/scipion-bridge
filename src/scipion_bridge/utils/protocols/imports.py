"""Dynamic import protocol synthesis for ``@canonical`` types.

For every type ``T`` registered with :func:`scipion_bridge.canonical`, this
module inspects the global type-resolution graph
(:func:`scipion_bridge.core.typed.resolve.current_registry`) to find which
targets among ``(T, Set[T], Collection[T])`` are reachable from :class:`Path`.
For each reachable target, :func:`synthesize_import_protocols` builds (and
caches) a concrete :class:`Protocol` subclass with deterministic module,
qualname, allowed file extensions, and fingerprint.

Each synthesized protocol accepts ``path: Input[Path]`` (either a single file
path or a ``glob.glob`` pattern) and a ``mode: Field[ImportMode]`` parameter:

* :attr:`ImportMode.ONCE` (default) discovers all currently matching files,
  resolves and streams their data (chunked when a slice-aware resolver exists),
  and terminates normally with a ``FLUSH`` signal.
* :attr:`ImportMode.WATCH` watches the target directory with an OS file watcher
  (``watchfiles``) and pushes newly arriving matching files into the pipeline.
  **Watch-mode streams never emit a ``FLUSH`` signal** until the pipeline is
  closed or cancelled.
"""

from __future__ import annotations

import fnmatch
import glob
import inspect
import os
import re
import sys
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Set as PySet,
    Tuple,
    Type,
    cast,
)

import networkx as nx
import watchfiles

from ...core.protocol.fields import Field, Input
from ...core.protocol.fingerprint import class_identity, digest, fingerprint_value
from ...core.protocol.protocol_base import (
    Protocol,
    ValidationError,
    _UNFINGERPRINTED,
)
from ...core.streaming.ops import Op
from ...core.struct.collection import Collection
from ...core.struct.set import Set
from ...core.typed.canonical_types import get_canonical_types
from ...core.typed.proxy import (
    PathToProxyGroupResolver,
    PathToProxyResolver,
    Proxy,
)
from ...core.typed.resolve import ComposedResolver, Registry, current_registry
from ...core.utils.hashing import join as _join

# Sealed resolver attributes hashed deterministically via _resolver_step_identity
# rather than cloudpickle.
_RESOLVER_STATE_ATTRS = frozenset({"_default_resolver", "_extension_resolvers"})


class ImportMode(Enum):
    """Execution mode for canonical import protocols.

    * ``ONCE``: discover matching files once, stream their resolved data, and
      finish normally (emitting a ``FLUSH`` signal downstream).
    * ``WATCH``: watch the directory with an OS file watcher (``watchfiles``)
      and push newly arriving files into the pipeline as they appear. Watch
      streams **never** send a ``FLUSH`` signal until the pipeline is closed.
    """

    ONCE = "once"
    WATCH = "watch"


def _expand_namespace(namespace: str) -> PySet[str]:
    parts = [p for p in namespace.split(".") if p]
    result: PySet[str] = set()
    prefix = ""
    for part in parts:
        prefix = f"{prefix}.{part}" if prefix else part
        result.add(prefix)
    return result


def _all_registered_namespaces(registry: Registry) -> PySet[str]:
    """All module namespaces present in ``registry.graph``, expanded."""
    namespaces: PySet[str] = set()
    for mod in registry.get_registered_modules():
        if isinstance(mod, str):
            namespaces.update(_expand_namespace(mod))
    return namespaces


def _find_resolve_func(
    registry: Registry,
    origin: Type[Any],
    target: Type[Any],
    *,
    intermediate: Optional[Type[Any]] = None,
) -> ComposedResolver[Any, Any]:
    """Find a resolver across all registered modules (including plugins)."""
    namespaces = _all_registered_namespaces(registry)
    local_scope = getattr(target, "__module__", __name__)
    return registry.find_resolve_func(
        namespaces,
        origin,
        target,
        intermediate=intermediate,
        local_scope_name=local_scope,
    )


def _is_reachable_from_path(registry: Registry, target: Type[Any]) -> bool:
    """Whether ``Path`` has a valid resolution path to ``target``."""
    try:
        _find_resolve_func(registry, Path, target)
    except TypeError:
        return False
    else:
        return True


def _proxy_nodes_for_target(
    registry: Registry,
    target: Type[Any],
) -> List[Type[Proxy]]:
    """All ``Proxy`` subclasses reachable from ``Path`` that can reach ``target``."""
    graph: Any = registry.graph
    if Path not in graph or target not in graph:
        return []

    proxies: List[Type[Proxy]] = []
    for node in graph.nodes:
        if (
            isinstance(node, type)
            and issubclass(node, Proxy)
            and node is not target
            and nx.has_path(graph, Path, node)
            and nx.has_path(graph, node, target)
        ):
            proxies.append(node)
    return proxies


def _derive_allowed_extensions(
    registry: Registry,
    target: Type[Any],
) -> Tuple[str, ...]:
    """Sorted lowercase file extensions accepted by proxies resolving to ``target``."""
    extensions: PySet[str] = set()
    for proxy_cls in _proxy_nodes_for_target(registry, target):
        proxy_exts = proxy_cls.extensions()
        if proxy_exts is not None:
            for ext in proxy_exts:
                extensions.add(ext.lower())
    return tuple(sorted(extensions))


def _precompute_target_resolvers(
    registry: Registry,
    target: Type[Any],
) -> Tuple[
    Optional[ComposedResolver[Any, Any]],
    Tuple[Tuple[str, ComposedResolver[Any, Any]], ...],
]:
    """Precompute default and per-extension ``ComposedResolver`` instances for ``target``."""
    default_resolver = _find_resolve_func(registry, Path, target)
    proxies = _proxy_nodes_for_target(registry, target)

    by_ext: Dict[str, List[Type[Proxy]]] = {}
    for proxy_cls in proxies:
        proxy_exts = proxy_cls.extensions()
        if proxy_exts is None:
            continue
        for ext in proxy_exts:
            by_ext.setdefault(ext.lower(), []).append(proxy_cls)

    ext_resolvers: List[Tuple[str, ComposedResolver[Any, Any]]] = []
    for ext in sorted(by_ext):
        best_resolver: Optional[ComposedResolver[Any, Any]] = None
        best_key: Optional[Tuple[int, str]] = None
        for proxy_cls in by_ext[ext]:
            try:
                candidate = _find_resolve_func(
                    registry,
                    Path,
                    target,
                    intermediate=proxy_cls,
                )
            except TypeError:
                continue

            # Prefer fewer steps, breaking ties deterministically by proxy name.
            key = (
                len(candidate.steps),
                f"{proxy_cls.__module__}.{proxy_cls.__qualname__}",
            )
            if best_key is None or key < best_key:
                best_key = key
                best_resolver = candidate

        if best_resolver is not None:
            ext_resolvers.append((ext, best_resolver))

    return default_resolver, tuple(ext_resolvers)


def _resolver_step_identity(step_cls: type) -> bytes:
    """Deterministic identity for a resolver step class.

    Handles dynamically generated ``PathToProxyResolver`` subclasses and
    ``DynamicFunctionProjector`` wrappers without falling back to
    non-deterministic ``cloudpickle.dumps``.
    """
    if issubclass(step_cls, PathToProxyResolver) and hasattr(step_cls, "target_cls"):
        base = (
            PathToProxyGroupResolver
            if issubclass(step_cls, PathToProxyGroupResolver)
            else PathToProxyResolver
        )
        return _join(
            b"proxy_resolver",
            class_identity(base),
            class_identity(step_cls.target_cls),
        )

    forward = getattr(step_cls, "forward", None)
    wrapped = getattr(forward, "__wrapped__", None) if forward is not None else None
    if wrapped is not None:
        fn = inspect.unwrap(wrapped)
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            return _join(
                b"resolver_fn",
                f"{fn.__module__}.{fn.__qualname__}".encode(),
            )
        else:
            return _join(
                b"resolver_fn",
                f"{fn.__module__}.{fn.__qualname__}".encode(),
                source.encode(),
            )

    return class_identity(step_cls)


def _watch_root_and_recursive(pattern: str) -> Tuple[Path, bool]:
    """Static directory to watch and whether watching must be recursive."""
    if not glob.has_magic(pattern):
        p = Path(pattern)
        return (p if p.is_dir() else p.parent), False

    parts = Path(pattern).parts
    static_parts: List[str] = []
    magic_index = len(parts)
    for idx, part in enumerate(parts):
        if glob.has_magic(part):
            magic_index = idx
            break
        static_parts.append(part)

    root = Path(*static_parts) if static_parts else Path(".")
    remaining = parts[magic_index:]
    recursive = "**" in remaining or len(remaining) > 1
    return root, recursive


def _matches_pattern(candidate: Path, pattern: str) -> bool:
    """Whether ``candidate`` matches a single path or glob ``pattern``."""
    if not glob.has_magic(pattern):
        return candidate.resolve() == Path(pattern).resolve()

    resolved_candidate = str(candidate.resolve())
    abs_pattern = pattern if os.path.isabs(pattern) else str(Path.cwd() / pattern)
    return fnmatch.fnmatch(resolved_candidate, abs_pattern) or fnmatch.fnmatch(
        str(candidate),
        pattern,
    )


class _CanonicalImportProtocol(Protocol):
    """Static base class for dynamically synthesized canonical import protocols.

    Accepts ``path: Input[Path]`` (a file path or ``glob.glob`` pattern) and
    streams resolved objects as ``{"output": item}``.

    In :attr:`ImportMode.ONCE` mode, all currently matching files are loaded
    and streamed, followed by a normal ``FLUSH`` signal.

    In :attr:`ImportMode.WATCH` mode, an OS file watcher (``watchfiles``)
    monitors the target directory and streams newly arriving files as they
    appear. **Watch-mode streams never send a ``FLUSH`` signal** until the
    pipeline is closed or cancelled.
    """

    canonical_type: ClassVar[type]
    target_type: ClassVar[type]
    allowed_extensions: ClassVar[Tuple[str, ...]] = ()

    path: Input[Path] = Input(
        optional=False,
        label="Path",
        help="File path or glob pattern to import.",
    )
    mode: Field[ImportMode] = Field(
        default=ImportMode.ONCE,
        label="Mode",
        help=(
            "Import mode: 'once' loads currently matching files and flushes; "
            "'watch' monitors the directory with an OS file watcher and never "
            "emits a FLUSH signal."
        ),
    )

    _default_resolver: Optional[ComposedResolver[Any, Any]] = None
    _extension_resolvers: Tuple[Tuple[str, ComposedResolver[Any, Any]], ...] = ()

    @classmethod
    def outputs(cls) -> Dict[str, Type[Any]]:
        return {"output": cls.target_type}

    def validate_path(self, path: Path) -> None:
        """Validate a concrete (non-glob) path against ``allowed_extensions``."""
        if not self.allowed_extensions:
            return
        raw = str(path)
        if glob.has_magic(raw):
            return
        suffix = Path(raw).suffix.lower()
        if suffix not in self.allowed_extensions:
            accepted = ", ".join(self.allowed_extensions)
            raise ValidationError(
                f"File '{path}' has unsupported extension '{suffix}' for "
                f"{type(self).__name__}; accepted extensions: {accepted}.",
            )

    def validate_protocol_configuration(self) -> None:
        registry = current_registry()
        if not _is_reachable_from_path(registry, self.target_type):
            raise ValidationError(
                f"No resolver path from Path to {self.target_type} is registered.",
            )
        if self.path.default is not None:
            self.validate_path(Path(self.path.default))

    def setup(self) -> None:
        """Precompute and freeze resolvers before launching the pipeline."""
        registry = current_registry()
        default_resolver, ext_resolvers = _precompute_target_resolvers(
            registry,
            self.target_type,
        )
        self._default_resolver = default_resolver
        self._extension_resolvers = ext_resolvers

    def _extension_allowed(self, file_path: Path) -> bool:
        if not self.allowed_extensions:
            return True
        return file_path.suffix.lower() in self.allowed_extensions

    def _resolver_for(self, file_path: Path) -> ComposedResolver[Any, Any]:
        suffix = file_path.suffix.lower()
        for ext, resolved in self._extension_resolvers:
            if ext == suffix:
                return resolved
        assert (
            self._default_resolver is not None
        ), "Protocol setup() must run before loading files."
        return self._default_resolver

    def _load_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        resolver = self._resolver_for(file_path)
        env_chunk_size = (
            int(os.environ["SCIPION_CHUNK_SIZE"])
            if "SCIPION_CHUNK_SIZE" in os.environ
            else None
        )
        env_target_bytes = (
            int(os.environ["SCIPION_TARGET_BYTE_SIZE"])
            if "SCIPION_TARGET_BYTE_SIZE" in os.environ
            else None
        )
        if any(step.requires_slice for step in resolver.steps):
            for chunk in resolver.iter(
                file_path,
                chunk_size=env_chunk_size,
                target_bytes=env_target_bytes,
            ):
                yield {"output": chunk}
        else:
            yield {"output": resolver(file_path)}

    def _matching_files(self, raw_path: Path) -> List[Path]:
        pattern = str(raw_path)
        if glob.has_magic(pattern):
            candidates = sorted(
                Path(match)
                for match in glob.glob(pattern, recursive=True)
                if Path(match).is_file()
            )
            return [p for p in candidates if self._extension_allowed(p)]
        else:
            self.validate_path(raw_path)
            return [raw_path] if raw_path.exists() else []

    def _discover_and_load(self, raw_path: Path) -> Iterator[Dict[str, Any]]:
        path = Path(raw_path)
        match self.mode.value:
            case ImportMode.ONCE:
                if not glob.has_magic(str(path)):
                    self.validate_path(path)
                    yield from self._load_file(path)
                else:
                    for file_path in self._matching_files(path):
                        yield from self._load_file(file_path)

            case ImportMode.WATCH:
                pattern = str(path)
                if not glob.has_magic(pattern):
                    self.validate_path(path)

                seen: PySet[Path] = set()
                for file_path in self._matching_files(path):
                    resolved_file = file_path.resolve()
                    seen.add(resolved_file)
                    yield from self._load_file(file_path)

                watch_root, recursive = _watch_root_and_recursive(pattern)
                for changes in watchfiles.watch(watch_root, recursive=recursive):
                    new_paths = sorted(
                        Path(changed_str)
                        for change, changed_str in changes
                        if change
                        in (watchfiles.Change.added, watchfiles.Change.modified)
                    )
                    for candidate in new_paths:
                        if not candidate.is_file():
                            continue
                        if not _matches_pattern(candidate, pattern):
                            continue
                        if not self._extension_allowed(candidate):
                            continue
                        resolved_candidate = candidate.resolve()
                        if resolved_candidate in seen:
                            continue
                        seen.add(resolved_candidate)
                        yield from self._load_file(candidate)

    def steps(self) -> Op:
        return self.path.flat_map(self._discover_and_load)

    def fingerprint(self) -> str:
        """Deterministic fingerprint for dynamically synthesized import protocols."""
        if not self.sealed:
            raise RuntimeError(
                f"The fingerprint of {type(self).__qualname__} is only defined "
                "once its pipeline is built.",
            )

        resolver_parts: List[bytes] = []
        if self._default_resolver is not None:
            for step in self._default_resolver.steps:
                resolver_parts.append(_resolver_step_identity(step.cls))
        for ext, ext_resolver in self._extension_resolvers:
            resolver_parts.append(ext.encode())
            for step in ext_resolver.steps:
                resolver_parts.append(_resolver_step_identity(step.cls))

        return digest(
            class_identity(_CanonicalImportProtocol),
            class_identity(self.canonical_type),
            f"{self.target_type.__module__}.{self.target_type.__qualname__}".encode(),
            *resolver_parts,
            *(
                fingerprint_value((name, value))
                for name, value in self._parameter_values()
            ),
            *(
                fingerprint_value((name, value))
                for name, value in sorted(self.__dict__.items())
                if name not in _UNFINGERPRINTED and name not in _RESOLVER_STATE_ATTRS
            ),
        )


def _sanitized_module_prefix(module_name: str) -> str:
    return "".join(
        part.capitalize() for part in re.split(r"[._]+", module_name) if part
    )


def _stem_for_canonical_type(cls: type, name_counts: Counter[str]) -> str:
    if name_counts[cls.__name__] > 1:
        return f"{_sanitized_module_prefix(cls.__module__)}_{cls.__name__}"
    else:
        return cls.__name__


def _format_path_help(target_label: str, extensions: Tuple[str, ...]) -> str:
    if extensions:
        ext_list = ", ".join(extensions)
        return (
            f"Path or glob pattern of {target_label} file(s) to import "
            f"(accepted extensions: {ext_list})."
        )
    else:
        return f"Path or glob pattern of {target_label} file(s) to import."


# Cache keyed by (canonical_type, target_type, class_name, allowed_extensions)
_SYNTHESIZED_CACHE: Dict[
    Tuple[type, type, str, Tuple[str, ...]],
    Type[Protocol],
] = {}


def synthesize_import_protocols() -> List[Type[Protocol]]:
    """Synthesize import protocol classes for all reachable ``@canonical`` types.

    Inspects :func:`get_canonical_types` and :func:`current_registry` to find
    every target among ``(T, Set[T], Collection[T])`` reachable from ``Path``.
    Synthesized classes are cached, attached to
    ``scipion_bridge.utils.protocols.imports``, and returned in deterministic
    order.
    """
    registry = current_registry()
    registered = get_canonical_types()
    name_counts = Counter(cls.__name__ for cls in registered)
    module = sys.modules[__name__]
    protocols: List[Type[Protocol]] = []

    for canonical_cls in registered:
        stem = _stem_for_canonical_type(canonical_cls, name_counts)
        candidates: Tuple[Tuple[Type[Any], str, str], ...] = (
            (canonical_cls, f"Import{stem}", canonical_cls.__name__),
            (
                cast(Type[Any], Set.__class_getitem__(canonical_cls)),
                f"ImportSetOf{stem}",
                f"Set[{canonical_cls.__name__}]",
            ),
            (
                cast(Type[Any], Collection.__class_getitem__(canonical_cls)),
                f"ImportCollectionOf{stem}",
                f"Collection[{canonical_cls.__name__}]",
            ),
        )

        for target_type, class_name, target_label in candidates:
            if not _is_reachable_from_path(registry, target_type):
                continue

            allowed_extensions = _derive_allowed_extensions(registry, target_type)
            cache_key = (canonical_cls, target_type, class_name, allowed_extensions)
            cached = _SYNTHESIZED_CACHE.get(cache_key)
            if cached is not None:
                setattr(module, class_name, cached)
                protocols.append(cached)
                continue

            path_help = _format_path_help(target_label, allowed_extensions)
            ext_doc = (
                f" Accepted extensions: {', '.join(allowed_extensions)}."
                if allowed_extensions
                else ""
            )
            doc = (
                f"Import {target_label} from a file path or glob pattern.{ext_doc}\n\n"
                "Supports ImportMode.ONCE (loads currently matching files and flushes) "
                "and ImportMode.WATCH (watches the directory with an OS file watcher "
                "and never emits a FLUSH signal)."
            )

            namespace: Dict[str, Any] = {
                "__module__": __name__,
                "__qualname__": class_name,
                "__doc__": doc,
                "__annotations__": {
                    "canonical_type": ClassVar[type],
                    "target_type": ClassVar[type],
                    "allowed_extensions": ClassVar[Tuple[str, ...]],
                    "path": Input[Path],
                    "mode": Field[ImportMode],
                    "_default_resolver": Optional[ComposedResolver[Any, Any]],
                    "_extension_resolvers": Tuple[
                        Tuple[str, ComposedResolver[Any, Any]], ...
                    ],
                },
                "canonical_type": canonical_cls,
                "target_type": target_type,
                "allowed_extensions": allowed_extensions,
                "path": Input(
                    optional=False,
                    label="Path",
                    help=path_help,
                    extensions=allowed_extensions or None,
                ),
                "mode": Field(
                    default=ImportMode.ONCE,
                    label="Mode",
                    help=(
                        "Import mode: 'once' loads currently matching files and "
                        "flushes; 'watch' monitors the directory with an OS file "
                        "watcher and never emits a FLUSH signal."
                    ),
                ),
                "_default_resolver": None,
                "_extension_resolvers": (),
            }

            proto_cls = type(class_name, (_CanonicalImportProtocol,), namespace)
            _SYNTHESIZED_CACHE[cache_key] = proto_cls
            setattr(module, class_name, proto_cls)
            protocols.append(proto_cls)

    return protocols


def __getattr__(name: str) -> Any:
    """Resolve dynamically synthesized import protocols by name (PEP 562)."""
    if name.startswith("Import"):
        synthesize_import_protocols()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ImportMode",
    "synthesize_import_protocols",
]
