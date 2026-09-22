"""Tests for pyworkflow backend resolvers (requires pwem / Scipion 3)."""

import pytest
import numpy as np

pwem = pytest.importorskip("pwem", reason="pwem (Scipion 3) not available")

import pwem.objects as emobj
from pwem.objects import (
    Particle,
    SetOfParticles,
    SetOfClasses2D,
)

import scipion_bridge as B
from scipion_bridge.single_particle import (
    Particle as BParticle,
    Class2D as BClass2D,
)
from scipion_bridge.core.struct.storage import UninitializedFieldError
from scipion_bridge.backend.pyworkflow import resolvers as res
from scipion_bridge.backend.pyworkflow.utils.resolve_graph import find_pointer_class

# ---------------------------------------------------------------------------
# Resolver graph discovery
# ---------------------------------------------------------------------------


def test_find_pointer_class_set_of_particles():
    # Import resolvers to ensure they are registered
    import scipion_bridge.backend.pyworkflow.resolvers  # noqa: F401

    assert find_pointer_class(B.Set[BParticle]) == "SetOfParticles"


def test_find_pointer_class_set_of_classes2d():
    import scipion_bridge.backend.pyworkflow.resolvers  # noqa: F401

    assert find_pointer_class(B.Set[BClass2D]) == "SetOfClasses2D"


# ---------------------------------------------------------------------------
# Single-particle resolvers
# ---------------------------------------------------------------------------


def _make_scipion_particle(
    filename: str,
    index: int = 1,
    sampling_rate: float = 1.5,
    add_ctf: bool = False,
    add_coord: bool = False,
) -> Particle:
    p = emobj.Particle()
    p.setLocation(index, filename)
    p.setSamplingRate(sampling_rate)

    if add_ctf:
        ctf = emobj.CTFModel()
        ctf.setDefocusU(15000.0)
        ctf.setDefocusV(14500.0)
        ctf.setDefocusAngle(45.0)
        ctf.setPhaseShift(0.0)
        p.setCTF(ctf)

    if add_coord:
        coord = emobj.Coordinate()
        coord.setX(120)
        coord.setY(240)
        p.setCoordinate(coord)

    return p


@pytest.fixture()
def tmp_mrcs(tmp_path):
    """Write a tiny 3-particle MRC stack and return its path."""
    import mrcfile

    stack_path = str(tmp_path / "particles.mrcs")
    data = np.zeros((3, 16, 16), dtype=np.float32)
    mrcfile.write(stack_path, data, overwrite=True)
    return stack_path


def test_resolve_scipion_particle_to_bridge_particle(tmp_mrcs):
    p_sci = _make_scipion_particle(
        filename=tmp_mrcs,
        index=1,
        sampling_rate=2.0,
        add_ctf=True,
        add_coord=True,
    )
    bridge_p = res.resolve_scipion_particle_to_bridge_particle(p_sci)

    assert isinstance(bridge_p, BParticle)
    assert bridge_p.pixels.shape == (16, 16)
    assert bridge_p.sampling_rate == pytest.approx(2.0)
    assert bridge_p.ctf.defocus_u == pytest.approx(15000.0)
    assert bridge_p.ctf.defocus_v == pytest.approx(14500.0)
    assert bridge_p.coordinate.x == pytest.approx(120.0)
    assert bridge_p.coordinate.y == pytest.approx(240.0)


def test_resolve_scipion_particle_to_bridge_particle_no_ctf(tmp_mrcs):
    p_sci = _make_scipion_particle(filename=tmp_mrcs, index=1)
    bridge_p = res.resolve_scipion_particle_to_bridge_particle(p_sci)

    with pytest.raises(UninitializedFieldError):
        _ = bridge_p.ctf.defocus_u


def test_bridge_particle_roundtrip(tmp_mrcs):
    """spa.Particle with CTF → Scipion Particle → back (CTF values preserved)."""
    p_sci_original = _make_scipion_particle(
        filename=tmp_mrcs,
        sampling_rate=1.2,
        add_ctf=True,
        add_coord=True,
    )
    bridge_p = res.resolve_scipion_particle_to_bridge_particle(p_sci_original)
    p_sci_roundtrip = res.resolve_bridge_particle_to_scipion_particle(bridge_p)

    assert p_sci_roundtrip.hasCTF()
    assert p_sci_roundtrip.getCTF().getDefocusU() == pytest.approx(15000.0)
    assert p_sci_roundtrip.getCTF().getDefocusV() == pytest.approx(14500.0)
    assert p_sci_roundtrip.hasCoordinate()


# ---------------------------------------------------------------------------
# SetOfParticles → Set[Particle] with / without CTF
# ---------------------------------------------------------------------------


def _make_set_of_particles(
    tmp_path,
    n: int = 4,
    add_ctf: bool = True,
    add_coord: bool = True,
) -> SetOfParticles:
    """Create an in-memory Scipion SetOfParticles backed by a temp SQLite file."""
    import mrcfile

    mrcs_path = str(tmp_path / "particles.mrcs")
    db_path = str(tmp_path / "particles.sqlite")

    data = np.zeros((n, 16, 16), dtype=np.float32)
    mrcfile.write(mrcs_path, data, overwrite=True)

    sop = emobj.SetOfParticles(filename=db_path)
    sop.setSamplingRate(1.5)
    sop.enableAppend()

    for i in range(n):
        p = emobj.Particle()
        p.setLocation(i + 1, mrcs_path)
        p.setSamplingRate(1.5)

        if add_ctf:
            ctf = emobj.CTFModel()
            ctf.setDefocusU(10000.0 + i * 100)
            ctf.setDefocusV(9500.0 + i * 100)
            ctf.setDefocusAngle(30.0)
            ctf.setPhaseShift(0.0)
            p.setCTF(ctf)

        if add_coord:
            coord = emobj.Coordinate()
            coord.setX(10 * i)
            coord.setY(20 * i)
            p.setCoordinate(coord)

        sop.append(p)

    sop.write()
    sop._getMapper().commit()
    return sop


def test_set_of_particles_to_bridge_with_ctf(tmp_path):
    sop = _make_set_of_particles(tmp_path, n=4, add_ctf=True, add_coord=True)
    bridge = res.resolve_set_of_particles_to_bridge_particles(sop)

    assert len(bridge) == 4
    assert bridge[0].ctf.defocus_u == pytest.approx(10000.0, rel=1e-3)
    assert bridge[0].coordinate.x == pytest.approx(0.0, abs=1)


def test_set_of_particles_to_bridge_no_ctf_raises_on_access(tmp_path):
    sop = _make_set_of_particles(tmp_path, n=3, add_ctf=False, add_coord=False)
    bridge = res.resolve_set_of_particles_to_bridge_particles(sop)

    assert len(bridge) == 3
    with pytest.raises(UninitializedFieldError):
        _ = bridge[0].ctf.defocus_u

    with pytest.raises(UninitializedFieldError):
        _ = bridge[0].coordinate.x


# ---------------------------------------------------------------------------
# SetOfClasses2D → Set[Class2D] roundtrip
# ---------------------------------------------------------------------------


def _make_set_of_classes2d(
    tmp_path, n_classes: int = 2, particles_per_class: int = 3
) -> SetOfClasses2D:
    """Create a minimal in-memory Scipion SetOfClasses2D."""
    import mrcfile

    mrcs_path = str(tmp_path / "class_particles.mrcs")
    db_path = str(tmp_path / "classes.sqlite")

    total = n_classes * particles_per_class
    data = np.zeros((total, 16, 16), dtype=np.float32)
    mrcfile.write(mrcs_path, data, overwrite=True)

    soc = emobj.SetOfClasses2D(filename=db_path)
    soc.enableAppend()

    particle_idx = 1
    for c in range(n_classes):
        cls2d = emobj.Class2D()
        cls2d.setObjId(c + 1)
        soc.append(cls2d)
        for _ in range(particles_per_class):
            p = emobj.Particle()
            p.setLocation(particle_idx, mrcs_path)
            p.setSamplingRate(1.5)
            cls2d.append(p)
            particle_idx += 1
        soc.update(cls2d)

    soc.write()
    soc._getMapper().commit()
    return soc


def test_set_of_classes2d_to_bridge(tmp_path):
    soc = _make_set_of_classes2d(tmp_path, n_classes=2, particles_per_class=3)
    bridge = res.resolve_set_of_classes2d_to_bridge_classes2d(soc)

    assert len(bridge) == 2
    # class_id column should have been set
    ids = np.array(bridge["class_id"]).squeeze()
    assert set(ids.tolist()) == {1, 2}


def test_set_of_classes2d_variable_particles(tmp_path):
    """Classes can have different particle counts (ragged staging)."""
    import mrcfile

    mrcs_path = str(tmp_path / "var_particles.mrcs")
    db_path = str(tmp_path / "var_classes.sqlite")

    data = np.zeros((5 + 3, 16, 16), dtype=np.float32)  # 5 + 3 particles
    mrcfile.write(mrcs_path, data, overwrite=True)

    soc = emobj.SetOfClasses2D(filename=db_path)
    soc.enableAppend()

    counts = [5, 3]
    particle_idx = 1
    for c, count in enumerate(counts):
        cls2d = emobj.Class2D()
        cls2d.setObjId(c + 1)
        soc.append(cls2d)
        for _ in range(count):
            p = emobj.Particle()
            p.setLocation(particle_idx, mrcs_path)
            p.setSamplingRate(1.5)
            cls2d.append(p)
            particle_idx += 1
        soc.update(cls2d)

    soc.write()
    soc._getMapper().commit()

    bridge = res.resolve_set_of_classes2d_to_bridge_classes2d(soc)
    assert len(bridge) == 2
