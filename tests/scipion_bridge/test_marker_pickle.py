import pickle
import subprocess
import sys

import cloudpickle
import pytest
import ray.cloudpickle

import scipion_bridge as B
from scipion_bridge.core.utils.marker import Marker
from scipion_bridge.core.utils.marker import Marker
from scipion_bridge.single_particle.particle import Particle

_SERIALIZERS = {
    "pickle": pickle.dumps,
    "cloudpickle": cloudpickle.dumps,
    "ray.cloudpickle": ray.cloudpickle.dumps,
}

_LOAD_AND_COMPARE = """
import pickle
import sys
import scipion_bridge as B
from scipion_bridge.core.utils.marker import Marker
from scipion_bridge.single_particle.particle import Particle

loaded = pickle.loads(sys.stdin.buffer.read())
expected = {expected}
print(loaded is expected and isinstance(loaded, type))
"""


def _roundtrip_in_subprocess(payload: bytes, expected_expr: str) -> bool:
    """Unpickle ``payload`` in a fresh interpreter and compare with ``expected_expr``."""
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_AND_COMPARE.format(expected=expected_expr)],
        input=payload,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode().strip() == "True"


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
@pytest.mark.parametrize(
    ("make_type", "expected_expr"),
    [
        (lambda: B.Set[Particle], "B.Set[Particle]"),
        (lambda: Marker[B.Set[Particle]], "Marker[B.Set[Particle]]"),
    ],
    ids=["flat", "nested"],
)
def test_marker_specialization_keeps_identity_across_processes(
    serializer, make_type, expected_expr
):
    payload = _SERIALIZERS[serializer](make_type())
    assert _roundtrip_in_subprocess(payload, expected_expr)


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
def test_marker_instance_class_keeps_identity_across_processes(serializer):
    payload = _SERIALIZERS[serializer](B.Set[Particle])
    assert pickle.loads(payload) is B.Set[Particle]


def test_marker_specialization_is_resolved_by_name():
    specialization = B.Set[Particle]
    module = sys.modules[specialization.__module__]

    assert "." not in specialization.__qualname__
    assert getattr(module, specialization.__qualname__) is specialization


def test_marker_specialization_repr_is_readable():
    assert repr(B.Set[Particle]) == "<class 'scipion_bridge.core.struct.set.Set[Particle]'>"


def test_marker_specialization_with_local_argument_falls_back_to_by_value():
    class LocalItem:
        pass

    specialization = Marker[LocalItem]

    assert specialization.__qualname__ == "Marker[LocalItem]"
    assert cloudpickle.loads(cloudpickle.dumps(specialization)).__name__ == "Marker[LocalItem]"


def test_module_getattr_raises_attribute_error_for_unknown_names():
    module = sys.modules[B.Set.__module__]

    with pytest.raises(AttributeError):
        getattr(module, "does_not_exist")

