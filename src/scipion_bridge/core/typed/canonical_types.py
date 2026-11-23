"""Registry of canonical data types (the Canonical Data Model of scipion-bridge).

Canonical types are the shared hub types into and from which external formats
and tools convert (e.g. ``CryoSPARC -> Particle -> RELION``). Cross-cutting
features — such as dynamically synthesized import protocols and visualizers —
target these canonical types rather than format-specific representations.
"""

from typing import Dict, Tuple, TypeVar

T = TypeVar("T", bound=type)

_CANONICAL_TYPES: Dict[Tuple[str, str], type] = {}


def canonical(cls: T) -> T:
    """Register ``cls`` as a canonical type and return it unchanged.

    A canonical type is a shared hub type in the type-resolution graph into and
    from which format-specific representations convert (for example
    ``CryoSPARC -> Particle -> RELION``), and against which synthesized import
    protocols and visualizers are built.

    Registration is keyed by ``(cls.__module__, cls.__qualname__)``, so
    reloading a module updates the entry in place without creating duplicates.
    """
    if not isinstance(cls, type):
        raise TypeError(
            f"@canonical can only decorate classes, got {type(cls).__name__}.",
        )
    _CANONICAL_TYPES[(cls.__module__, cls.__qualname__)] = cls
    return cls


def get_canonical_types() -> Tuple[type, ...]:
    """Return all registered canonical types in registration order."""
    return tuple(_CANONICAL_TYPES.values())


def unregister_canonical(cls: type) -> None:
    """Remove ``cls`` from the canonical type registry if present."""
    _CANONICAL_TYPES.pop((cls.__module__, cls.__qualname__), None)

