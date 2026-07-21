import scipion_bridge as B

import zarr

class CTF(B.Struct):
    voltage_kv: float
    amplitude_contrast: float

class Particle(B.Struct):
    foo: int
    bar: float

    ctf: CTF


def test_create_dataclass_schema():

    Particle.print_schema()

    particle = Particle()
    print(particle)

    # root = zarr.group()
    # foo = root.create_group('foo')

    # _ = foo.zeros(name='baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
    # _ = foo.zeros(name='baz2', shape=(500, 500), chunks=(1000, 1000), dtype='i4')

    # print(root.tree())


if __name__ == "__main__":
    test_create_dataclass_schema()