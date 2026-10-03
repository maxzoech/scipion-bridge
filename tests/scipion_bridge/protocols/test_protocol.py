import pytest

import scipion_bridge as B
from scipion_bridge import Protocol


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
    magic_number: B.Input[int] = B.Input(
        default=42, optional=True, label="Magic Number"
    )

    # Parameters
    param: B.Field[float]
    param_default: B.Field[int] = B.Field(default=42)

    # Fields
    state: int = 42

    def outputs(self):
        return {}

    def steps(self):
        pass


def test_create_protocol():
    proto = BasicProtocol()

    config = proto.configuration
    assert config.inputs["path"] == B.Input(optional=False)
    assert config.inputs["magic_number"] == B.Input(
        default=42, optional=True, label="Magic Number"
    )

    assert config.parameters["param"] == B.Field(optional=False)
    assert config.parameters["param_default"] == B.Field(default=42)

    assert config.inputs["path"].dtype == str
    assert config.inputs["magic_number"].dtype == int
    assert config.parameters["param"].dtype == float
    assert config.parameters["param_default"].dtype == int

    assert proto.magic_number.dtype == int
    assert proto.param_default.dtype == int


def test_protocol_untyped_state():
    with pytest.raises(
        TypeError,
        match="The protocol .* has declared attributes without type annotation.",
    ):

        class UntypedStateProtocol(Protocol):
            state = 42

            def outputs(self):
                return {}

            def steps(self):
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

        def outputs(self):
            return {}

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


class CloudpickleTestProtocol(Protocol):
    particles: B.Input[str] = B.Input(label="Input Particles", optional=False)
    batch_size: B.Field[int] = B.Field(default=100)
    extra_state: int = 10

    def _compute_latents(self, item: str) -> str:
        return f"transformed_{item}"

    def outputs(self):
        return {"output": str}

    def steps(self):
        return None


def test_extract_declaration_order_from_ast():
    from scipion_bridge.core.protocol.protocol_base import (
        _extract_declaration_order_from_ast,
    )

    order = _extract_declaration_order_from_ast(BasicProtocol)
    assert order == ["path", "magic_number", "param", "param_default", "state"]

    # Class with no source on disk (e.g. built via type)
    dyn_cls = type("DynamicClass", (object,), {})
    assert _extract_declaration_order_from_ast(dyn_cls) is None


def test_protocol_class_cloudpickle_serialization_main_module():
    import cloudpickle

    CloudpickleTestProtocol.__module__ = "__main__"
    data = cloudpickle.dumps(CloudpickleTestProtocol)
    unpickled_cls = cloudpickle.loads(data)

    assert unpickled_cls is not None
    assert list(unpickled_cls._configuration.inputs.keys()) == ["particles"]
    assert list(unpickled_cls._configuration.parameters.keys()) == ["batch_size"]
    assert list(unpickled_cls._configuration.states.keys()) == ["extra_state"]


def test_protocol_instance_and_bound_method_cloudpickle_serialization():
    import cloudpickle

    CloudpickleTestProtocol.__module__ = "__main__"
    proto = CloudpickleTestProtocol()
    proto.extra_state = 999

    inst_data = cloudpickle.dumps(proto)
    unpickled_inst = cloudpickle.loads(inst_data)

    assert unpickled_inst.extra_state == 999
    assert list(unpickled_inst.configuration.inputs.keys()) == ["particles"]
    assert list(unpickled_inst.configuration.parameters.keys()) == ["batch_size"]

    method_data = cloudpickle.dumps(proto._compute_latents)
    unpickled_method = cloudpickle.loads(method_data)

    assert callable(unpickled_method)
    assert unpickled_method("sample") == "transformed_sample"
    assert unpickled_method.__self__.extra_state == 999


if __name__ == "__main__":
    test_create_protocol()
