import pytest

import scipion_bridge as B


def test_protocol_fields():
    # These should pass
    f1 = B.Field(42)
    assert f1.default == 42
    assert f1.optional is False

    f2 = B.Field(optional=True)
    assert f2.default is None
    assert f2.optional is True

    f3 = B.Field(42, optional=True)
    assert f3.default == 42
    assert f3.optional is True

    f4 = B.Field()
    assert f4.default is None
    assert f4.optional is True


from scipion_bridge.core.protocol.base import Protocol

class BasicProtocol(Protocol):

    # Parameters    
    path: B.Field[int]

    # Fields
    state: int = 42

    def run(self, inputs: int):
        pass

from typing import Any
def test_create_protocol():
    proto = BasicProtocol()
    desc = proto._exec_info

    assert proto.path.optional is False
    assert "inputs" in desc.inputs
    assert "path" in desc.configuration
    assert "state" in desc.states


def test_protocol_variational_args():
    with pytest.raises(RuntimeError, match="Method run\\(\\) in .* has variational arguments"):
        class VariationalProtocol(Protocol):
            def run(self, *args: Any):
                pass

    with pytest.raises(RuntimeError, match="Method run\\(\\) in .* has variational arguments"):
        class KwargsProtocol(Protocol):
            def run(self, **kwargs: Any):
                pass


def test_protocol_missing_annotation():
    with pytest.raises(RuntimeError, match="Protocol inputs inputs .* either do not have type annotations or a default value"):
        class MissingAnnotationProtocol(Protocol):
            def run(self, inputs):
                pass


def test_protocol_input_default_value():
    with pytest.raises(RuntimeError, match="Protocol inputs inputs .* either do not have type annotations or a default value"):
        class DefaultValueProtocol(Protocol):
            def run(self, inputs: int = 42):
                pass


def test_protocol_untyped_state():
    with pytest.raises(TypeError, match="The protocol states 'state' .* do not have type annotations"):
        class UntypedStateProtocol(Protocol):
            state = 42

            def run(self, inputs: int):
                pass


if __name__ == "__main__":
    test_create_protocol()