from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Type, Dict
from enum import Enum


class ProtocolConfigurationProvider(ABC):
    """Abstract provider for managing field values of Protocol instances."""

    @abstractmethod
    def get_value(self, name: str, default: Any = None) -> Any:
        """Retrieve the value of a field for a given protocol instance."""
        pass


class StaticProtocolConfigurationProvider(ProtocolConfigurationProvider):
    """Provider serving field values from a fixed mapping.

    Fields without an entry in the mapping resolve to their declared default.
    """

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def get_value(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)
