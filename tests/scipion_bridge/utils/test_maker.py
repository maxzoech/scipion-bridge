from typing import TypeVar
from scipion_bridge.core.utils.marker import Marker

T = TypeVar("T")

class ArrayMarker(Marker[T]):
    pass # Maker does nothing for now

def test_maker():

    class Foo:
        marker: ArrayMarker[float] = ArrayMarker()
        marker_typevar: ArrayMarker["str"] = ArrayMarker()

    assert Foo.marker.dtype == float
    assert Foo.marker_typevar.dtype == str


if __name__ == "__main__":
    test_maker()