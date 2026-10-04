from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class ResourceProvider(ABC):
    """Abstract provider for managing lifecycle and caching of Protocol resources."""

    @abstractmethod
    def get_resource(
        self,
        name: str,
        builder: Callable[[Any], Any],
        instance: Any,
    ) -> Any:
        """Retrieve or build a resource for a given protocol instance."""
        pass


class DefaultResourceProvider(ResourceProvider):
    """Process-local provider that lazily constructs resources and caches them."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def get_resource(
        self,
        name: str,
        builder: Callable[[Any], Any],
        instance: Any,
    ) -> Any:
        
        if name not in self._cache:
            self._cache[name] = builder(instance)

        return self._cache[name]
