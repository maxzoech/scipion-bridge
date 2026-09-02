import pytest
from scipion_bridge.backend.standalone.container import configure_default_env

# Eagerly wire container so top-level Struct definitions in test modules collect cleanly
_test_container = configure_default_env()


@pytest.fixture(autouse=True)
def setup_test_container():
    """Ensure default standalone container is active for all tests."""
    yield _test_container

