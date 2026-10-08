import os

# Read by Ray on import: a test that needs Ray but does not request the
# ray_cluster fixture fails instead of starting an unconfigured cluster.
os.environ.setdefault("RAY_ENABLE_AUTO_CONNECT", "0")

import sys  # noqa: E402

import pytest  # noqa: E402
import ray  # noqa: E402

from scipion_bridge.backend.standalone.container import (  # noqa: E402
    configure_default_env,
)

# Eagerly wire container so top-level Struct definitions in test modules collect cleanly
_test_container = configure_default_env()


@pytest.fixture(autouse=True)
def setup_test_container():
    """Ensure default standalone container is active for all tests."""
    yield _test_container


def pytest_collection_modifyitems(items):
    """Mark the tests running on the Ray cluster, to select them with ``-m ray``."""
    for item in items:
        if "ray_cluster" in item.fixturenames:
            item.add_marker(pytest.mark.ray)


@pytest.fixture(scope="session")
def ray_cluster():
    """Ray cluster with 2 CPUs and 2 logical GPUs, shared by the session.

    The PYTHONPATH of the test modules and Struct definitions is exported to
    the environment the raylet inherits rather than passed as a runtime_env:
    workers with a runtime_env cannot reuse the processes Ray prestarts, which
    costs about a second per actor. Every actor of a pipeline takes a worker
    process of its own, and Ray starts only ``num_cpus`` of them at a time
    unless ``worker_maximum_startup_concurrency`` raises the limit.
    """
    tests_dir = os.path.abspath("tests")
    paths = [
        os.getcwd(),
        os.path.abspath("src"),
        *(root for root, _, _ in os.walk(tests_dir)),
        *(os.path.abspath(path) for path in sys.path if path),
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("PYTHONPATH", ":".join(dict.fromkeys(paths)))
        ray.init(
            num_cpus=2,
            # Logical GPUs: Ray schedules them without a GPU driver.
            num_gpus=2,
            _system_config={"worker_maximum_startup_concurrency": 8},
        )
        yield
        ray.shutdown()


@pytest.fixture(
    params=["staging", "arrow_from_batch"]
)  # TODO: add "arrow_frozen" when the Arrow storage engine is implemented
def engine_mode(request):
    """Parametrizes tests across the three engine states."""
    return request.param


@pytest.fixture
def as_engine(engine_mode):
    """Transforms a populated Set into the active test engine state."""
    import scipion_bridge as B

    def _transform(set_instance: B.Set):
        match engine_mode:
            case "staging":
                return set_instance
            case "arrow_frozen":
                set_instance.to_arrow()
                return set_instance
            case "arrow_from_batch":
                batch = set_instance.to_arrow()
                return type(set_instance).from_arrow(batch)

    return _transform
