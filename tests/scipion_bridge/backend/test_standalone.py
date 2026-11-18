"""The standalone container (in-process)."""

import pytest

from scipion_bridge.core.streaming import Pipeline, Source


def test_the_standalone_container_does_not_implement_streaming():
    with pytest.raises(NotImplementedError, match="RayBackend"):
        Pipeline.from_sink(Source("items").sink(print))
