"""The public namespaces of scipion_bridge and its packages."""

import importlib
import json
import subprocess
import sys

import pytest

import scipion_bridge as B
from scipion_bridge import core

_PACKAGES = [
    "scipion_bridge",
    "scipion_bridge.core.environment",
    "scipion_bridge.core.utils",
    "scipion_bridge.core.typed",
    "scipion_bridge.core.struct",
    "scipion_bridge.core.streaming",
    "scipion_bridge.core.protocol",
    "scipion_bridge.single_particle",
    "scipion_bridge.backend",
    "scipion_bridge.backend.ray",
]

# Imports scipion_bridge in a fresh interpreter, in which nothing else has
# loaded a backend yet.
_LOADED_BY_IMPORT = """
import json
import sys
import scipion_bridge
print(json.dumps(sorted(sys.modules)))
"""


@pytest.fixture(scope="module")
def loaded_by_import() -> set[str]:
    """The modules that ``import scipion_bridge`` loads."""
    result = subprocess.run(
        [sys.executable, "-c", _LOADED_BY_IMPORT],
        capture_output=True,
        check=True,
    )
    return set(json.loads(result.stdout))


@pytest.mark.parametrize(
    "module",
    ["ray", "pyworkflow", "scipion_bridge.backend"],
)
def test_importing_the_library_loads_no_backend(loaded_by_import, module):
    assert module not in loaded_by_import


@pytest.mark.parametrize("package", _PACKAGES)
def test_every_exported_name_exists_and_is_public(package):
    module = importlib.import_module(package)

    assert [name for name in module.__all__ if not hasattr(module, name)] == []
    assert [name for name in module.__all__ if name.startswith("_")] == []


@pytest.mark.parametrize("package", _PACKAGES)
def test_no_exported_name_hides_a_submodule(package):
    module = importlib.import_module(package)

    hidden = [
        name
        for name in module.__all__
        if f"{package}.{name}" in sys.modules
        and not isinstance(getattr(module, name), type(sys))
    ]
    assert hidden == []


def test_the_streaming_api_is_available_from_the_top_level():
    assert (B.Op, B.Source, B.Sink, B.Pipeline) == (
        core.streaming.Op,
        core.streaming.Source,
        core.streaming.Sink,
        core.streaming.Pipeline,
    )
    assert (B.SinkWriter, B.CallbackSinkWriter) == (
        core.streaming.SinkWriter,
        core.streaming.CallbackSinkWriter,
    )


@pytest.mark.parametrize(
    "name",
    [
        "ParticleStackProxy",
        "StarfileProxy",
        "MRCStackProxy",
        "ResourceProvider",
        "DefaultResourceProvider",
    ],
)
def test_names_of_other_packages_are_not_in_the_top_level(name):
    assert name not in B.__all__
    assert not hasattr(B, name)
