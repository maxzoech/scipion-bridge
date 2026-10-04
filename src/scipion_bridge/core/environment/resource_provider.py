from abc import ABC, abstractmethod
from enum import Enum
import threading
from typing import Any, Callable, Dict, Optional


class ResourceScope(str, Enum):
    """Lifecycle and sharing scope for protocol resources."""

    PROCESS = "process"
    SHARED = "shared"


class ResourceProvider(ABC):
    """Abstract provider for managing lifecycle and caching of Protocol resources."""

    @abstractmethod
    def get_resource(
        self,
        name: str,
        builder: Callable[[Any], Any],
        instance: Any,
        scope: ResourceScope = ResourceScope.PROCESS,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Retrieve or build a resource for a given protocol instance."""
        pass


class DefaultResourceProvider(ResourceProvider):
    """Process-local provider that lazily constructs resources and caches them."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_resource(
        self,
        name: str,
        builder: Callable[[Any], Any],
        instance: Any,
        scope: ResourceScope = ResourceScope.PROCESS,
        dtype: Optional[Any] = None,
    ) -> Any:
        key = f"{instance.protocol_id}:{name}"

        if key not in self._cache:
            with self._lock:
                if key not in self._cache:
                    res = builder(instance)

                    if (
                        dtype is not None
                        and isinstance(dtype, type)
                        and not isinstance(res, dtype)
                    ):
                        raise TypeError(
                            f"Resource '{name}' expected type {dtype}, got {type(res)}."
                        )

                    self._cache[key] = res

        return self._cache[key]
