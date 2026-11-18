from abc import ABC, abstractmethod
from typing import Any, Mapping


class ProtocolConfigurationProvider(ABC):
    """Abstract provider for the parameter values of protocols.

    Values are scoped by the instance name of a protocol (``Protocol.name``),
    so that the protocols of a chain declaring a parameter of the same name
    get values of their own.
    """

    @abstractmethod
    def get_value(self, scope: str, name: str, default: Any = None) -> Any:
        """The value of parameter ``name`` of the protocol named ``scope``."""
        pass


class StaticProtocolConfigurationProvider(ProtocolConfigurationProvider):
    """Provider serving parameter values from a fixed mapping,
    ``{protocol name: {parameter: value}}``.

    Parameters without an entry resolve to ``default``: the value given to
    the constructor of the protocol, else the declared default.
    """

    def __init__(self, values: Mapping[str, Mapping[str, Any]]) -> None:
        self.values = {scope: dict(scoped) for scope, scoped in values.items()}

    def get_value(self, scope: str, name: str, default: Any = None) -> Any:
        return self.values.get(scope, {}).get(name, default)
