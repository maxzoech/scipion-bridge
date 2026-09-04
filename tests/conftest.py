import pytest
from scipion_bridge.backend.standalone.container import configure_default_env

# Eagerly wire container so top-level Struct definitions in test modules collect cleanly
_test_container = configure_default_env()


@pytest.fixture(autouse=True)
def setup_test_container():
    """Ensure default standalone container is active for all tests."""
    yield _test_container


@pytest.fixture(params=["staging", "arrow_frozen", "arrow_from_batch"])
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
                return type(set_instance).from_arrow(set_instance.dtype, batch)

    return _transform

