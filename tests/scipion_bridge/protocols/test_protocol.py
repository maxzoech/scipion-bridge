import pytest

import scipion_bridge as B
from scipion_bridge import Protocol

from typing import Any


def test_protocol_fields():
    # These should pass
    f1 = B.Field(default=42)
    assert f1.default == 42
    assert f1.optional is False

    f2 = B.Field(optional=True)
    assert f2.default is None
    assert f2.optional is True

    f3 = B.Field(default=42, optional=True)
    assert f3.default == 42
    assert f3.optional is True

    f4 = B.Field()
    assert f4.default is None
    assert f4.optional is True


class BasicProtocol(Protocol):

    # Inputs
    path: B.Input[str]
    magic_number: B.Input[int] = B.Input(default=42, optional=True, label="Magic Number")

    # Parameters
    param: B.Field[float]
    param_default: B.Field[int] = B.Field(default=42)

    # Fields
    state: int = 42

    def steps(self):
        pass

def test_create_protocol():
    proto = BasicProtocol()

    config = proto.configuration
    assert config.inputs["path"] == B.Input(optional=False)
    assert config.inputs["magic_number"] == B.Input(default=42, optional=True, label="Magic Number")

    assert config.parameters["param"] == B.Field(optional=False)
    assert config.parameters["param_default"] == B.Field(default=42)


def test_protocol_untyped_state():
    with pytest.raises(TypeError, match="The protocol .* has declared attributes without type annotation."):
        class UntypedStateProtocol(Protocol):
            state = 42

            def run(self, inputs: int):
                pass


def test_convert_scipion_to_python_enum_with_protocol_configuration():
    from enum import Enum
    from scipion_bridge.backend.pyworkflow.workflow_container import (
        _PyWorkflowProtocolConfigurationProvider,
    )

    class Color(Enum):
        RED = "red"
        GREEN = "green"

    class EnumProtocol(Protocol):
        color: B.Field[Color] = B.Field(default=Color.RED)

        def steps(self):
            pass

    class MockPyWorkflowParam:
        def __init__(self, val):
            self._val = val

        def get(self):
            return self._val

    class MockBackend:
        def __init__(self):
            self.color = MockPyWorkflowParam(1)

    proto = EnumProtocol()
    backend = MockBackend()
    provider = _PyWorkflowProtocolConfigurationProvider(
        backend, configuration=proto._configuration
    )

    assert provider.get_value("color") == Color.GREEN


if __name__ == "__main__":
    test_create_protocol()