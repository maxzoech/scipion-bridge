import json
import pickle
import subprocess
import sys

import cloudpickle
import pytest
import ray.cloudpickle

import scipion_bridge as B
from scipion_bridge.core.utils.marker import Marker
from scipion_bridge.single_particle.particle import Particle

_SERIALIZERS = {
    "pickle": pickle.dumps,
    "cloudpickle": cloudpickle.dumps,
    "ray.cloudpickle": ray.cloudpickle.dumps,
}

_TYPES = {
    "flat": (lambda: B.Set[Particle], "B.Set[Particle]"),
    "nested": (lambda: Marker[B.Set[Particle]], "Marker[B.Set[Particle]]"),
}

# Unpickles every case in one fresh interpreter: starting one per case costs
# about two seconds each for importing scipion_bridge.
_LOAD_AND_COMPARE = """
import json
import pickle
import sys
import scipion_bridge as B
from scipion_bridge.core.utils.marker import Marker
from scipion_bridge.single_particle.particle import Particle

cases = pickle.loads(sys.stdin.buffer.read())
print(json.dumps({
    case: (loaded := pickle.loads(payload)) is eval(expected) and isinstance(loaded, type)
    for case, (payload, expected) in cases.items()
}))
"""


@pytest.fixture(scope="module")
def roundtrips_in_subprocess() -> dict[str, bool]:
    """Whether each pickled specialization is identical when unpickled in a fresh interpreter."""
    cases = {
        f"{kind}-{serializer}": (dumps(make_type()), expected)
        for kind, (make_type, expected) in _TYPES.items()
        for serializer, dumps in _SERIALIZERS.items()
    }
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_AND_COMPARE],
        input=pickle.dumps(cases),
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("serializer", list(_SERIALIZERS))
@pytest.mark.parametrize("kind", list(_TYPES))
def test_marker_specialization_keeps_identity_across_processes(
    roundtrips_in_subprocess, kind, serializer
):
    assert roundtrips_in_subprocess[f"{kind}-{serializer}"]


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
    assert (
        repr(B.Set[Particle])
        == "<class 'scipion_bridge.core.struct.set.Set[Particle]'>"
    )


def test_marker_specialization_with_local_argument_falls_back_to_by_value():
    class LocalItem:
        pass

    specialization = Marker[LocalItem]

    assert specialization.__qualname__ == "Marker[LocalItem]"
    assert (
        cloudpickle.loads(cloudpickle.dumps(specialization)).__name__
        == "Marker[LocalItem]"
    )


def test_module_getattr_raises_attribute_error_for_unknown_names():
    module = sys.modules[B.Set.__module__]

    with pytest.raises(AttributeError):
        getattr(module, "does_not_exist")
