import pytest
from typing import TypeVar
from scipion_bridge.core.utils.marker import Marker

T = TypeVar("T")

class ArrayMarker(Marker[T]):
    pass # Maker does nothing for now

def test_maker():

    class Foo:
        marker: ArrayMarker[float] = ArrayMarker()

    assert Foo.marker.dtype == float

    with pytest.raises(TypeError, match="Forward declarations using type strings are not supported yet"):
        class Bar:
            marker_typevar: ArrayMarker["str"] = ArrayMarker()


if __name__ == "__main__":
    test_maker()