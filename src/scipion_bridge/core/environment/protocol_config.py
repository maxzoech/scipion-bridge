from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Dict
from enum import Enum


class ProtocolConfigurationProvider(ABC):
    """Abstract provider for managing field values of Protocol instances."""

    @abstractmethod
    def get_value(self, name: str, default: Any = None) -> Any:
        """Retrieve the value of a field for a given protocol instance."""
        pass

