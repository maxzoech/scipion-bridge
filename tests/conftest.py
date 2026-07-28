import pytest
from scipion_bridge.backend.standalone.container import configure_default_env


@pytest.fixture(autouse=True)
def setup_test_container():
    """Automatically wire the default standalone container for all tests."""
    container = configure_default_env()
    yield container
    container.unwire()
