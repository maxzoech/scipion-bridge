
import numpy as np
import scipion_bridge as B

def test_set_of_classes_2d_struct():

    class CTF(B.Struct):
        bar: np.float32

    class Particle(B.Struct):
        H = B.Arg()
        W = B.Arg()

        pixels = B.Array[np.float32](shape=(H, W))
        ctf: CTF

    class Class2D(B.Struct):
        H = B.Arg()
        W = B.Arg()

        particles: B.Set[Particle]
        average = B.Array[np.float32](shape=(H, W))

    classes_set = B.Set[Class2D.static(H=128, W=128)](capacity=None)
    classes_set.print_schema()


if __name__ == "__main__":
    from scipion_bridge.backend.standalone.container import configure_default_env
    
    # Wire the container for 'scipion_bridge'
    container = configure_default_env()

    test_set_of_classes_2d_struct()