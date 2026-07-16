import pytest

import scipion_bridge as B
from scipion_bridge.core.protocol import create_protocol

def test_protocol_fields():
    # These should pass
    f1 = B.Field(42)
    assert f1.default == 42
    assert f1.optional is False

    f2 = B.Field(optional=True)
    assert f2.default is None
    assert f2.optional is True

    f3 = B.Field(42, optional=True)
    assert f3.default == 42
    assert f3.optional is True

    f4 = B.Field()
    assert f4.default is None
    assert f4.optional is True


class BasicProtocol(B.Protocol):

    # Parameters    
    path: B.Field[int] = B.Field(42)

    # Fields
    state: int = 42

    def run(self, inputs: int):
        pass

def test_create_protocol():
    
    desc = create_protocol(BasicProtocol())

    print(BasicProtocol().path.optional)



if __name__ == "__main__":
    test_create_protocol()