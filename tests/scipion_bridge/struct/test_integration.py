
import numpy as np
import pytest

import scipion_bridge as B

def test_set_of_classes_2d_struct():

    class Particle(B.Struct):
        H = B.Arg()
        W = B.Arg()

        pixels = B.Array[np.float32](shape=(H, W))

    class Class2D(B.Struct):
        H = B.Arg()
        W = B.Arg()

        particles: B.Set[Particle]
        average = B.Array[np.float32](shape=(H, W))

    classes_set = B.Set[Class2D](capacity=20)

    particles_1 = B.Set[Particle]([
        Particle(pixels=np.random.uniform(size=(128, 128,))) for _ in range(10)
    ])

    cls_1 = Class2D(
        particles=particles_1,
        average=np.random.uniform(size=(128, 128)),
    )

    particles_2 = B.Set[Particle]([
        Particle(pixels=np.random.uniform(size=(128, 128,))) for _ in range(15)
    ])
    
    cls_2 = Class2D(
        particles=particles_2,
        average=np.random.uniform(size=(128, 128)),
    )

    classes_set[0] = cls_1
    classes_set[1] = cls_2

    classes_set = classes_set.freeze()  # Freeze the storage to the arrow engine

    print(classes_set._storage.root_storage._engine)

    # Verify element 0
    assert isinstance(classes_set[0].average, np.ndarray)
    assert np.allclose(classes_set[0].average, cls_1.average)
    assert np.allclose(classes_set[0].particles["pixels"][0], particles_1[0].pixels)
    assert np.allclose(classes_set[1].particles["pixels"][1], particles_2[1].pixels)

    # Verify unpopulated slot access raises ValueError
    with pytest.raises(ValueError, match="Cannot read unpopulated or null value"):
        _ = classes_set[5].average

    # Verify arrow compilation retains explicit capacity
    batch_full = classes_set.to_arrow()
    assert len(batch_full) == 20

    # Verify set construction from sequence
    classes_from_seq = B.Set[Class2D]([cls_1])
    assert np.allclose(classes_from_seq[0].average, cls_1.average)
    assert np.allclose(classes_from_seq[0].particles["pixels"][0], particles_1[0].pixels)

    # Verify arrow compilation
    batch = classes_from_seq.to_arrow()
    assert len(batch) == 1
    assert "particles" in batch.schema.names
    assert "average" in batch.schema.names

if __name__ == "__main__":
    from scipion_bridge.backend.standalone.container import configure_default_env
    
    # Wire the container for 'scipion_bridge'
    container = configure_default_env()

    test_set_of_classes_2d_struct()