import pytest
from scipion_bridge.backend.standalone.container import configure_default_env

# Eagerly wire container so top-level Struct definitions in test modules collect cleanly
_test_container = configure_default_env()


@pytest.fixture(autouse=True)
def setup_test_container():
    """Ensure default standalone container is active for all tests."""
    yield _test_container


@pytest.fixture(scope="session", autouse=True)
def init_ray_session():
    """Initialize Ray session with comprehensive PYTHONPATH for test modules and Struct definitions."""
    import os
    import sys
    import ray

    if ray.is_initialized():
        ray.shutdown()

    extra_paths = [os.getcwd(), os.path.abspath("src")]
    tests_dir = os.path.abspath("tests")
    if os.path.exists(tests_dir):
        for root, _, _ in os.walk(tests_dir):
            extra_paths.append(root)

    all_paths = list(
        dict.fromkeys(extra_paths + [os.path.abspath(p) for p in sys.path if p])
    )
    python_path = ":".join(all_paths)
    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        runtime_env={"env_vars": {"PYTHONPATH": python_path}},
    )
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(
    params=["staging"]
)  # TODO: add "arrow_frozen", "arrow_from_batch" when implemented
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
