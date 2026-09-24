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
from scipion_bridge import Protocol
from scipion_bridge.single_particle import (
    Particle as BParticle,
    Class2D as BClass2D,
    FlexParticle as BFlexParticle,
)
from scipion_bridge.core.struct.storage import UninitializedFieldError
from scipion_bridge.backend.pyworkflow import resolvers as res
from scipion_bridge.backend.pyworkflow.utils.resolve_graph import (
    find_pointer_class,
    find_output_pointer_class,
)
from scipion_bridge.backend.pyworkflow.scipion3_protocol import (
    reduce_minibatch_to_persistent_output,
    convert_protocol_to_scipion3_protocol,
)

# ---------------------------------------------------------------------------
# Resolver graph discovery
# ---------------------------------------------------------------------------


def test_find_pointer_class_set_of_particles():
    # Import resolvers to ensure they are registered
    import scipion_bridge.backend.pyworkflow.resolvers  # noqa: F401

    assert find_pointer_class(B.Set[BParticle]) == "SetOfParticles"


def test_find_pointer_class_collection_classes2d():
    import scipion_bridge.backend.pyworkflow.resolvers  # noqa: F401

    assert find_pointer_class(B.Collection[BClass2D]) == "SetOfClasses2D"
    assert find_output_pointer_class(B.Collection[BClass2D]) == emobj.SetOfClasses2D


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
    bridge = res.resolve_set_of_classes2d_to_collection_classes2d(soc)

    assert isinstance(bridge, B.Collection)
    assert len(bridge) == 2
    assert bridge.initialized_indices() == [0, 1]
    assert bridge[0].class_id == 0
    assert bridge[1].class_id == 1
    assert len(bridge[0].particles) == 3
    assert len(bridge[1].particles) == 3


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

    bridge = res.resolve_set_of_classes2d_to_collection_classes2d(soc)
    assert isinstance(bridge, B.Collection)
    assert len(bridge) == 2
    assert len(bridge[0].particles) == 5
    assert len(bridge[1].particles) == 3


def test_set_of_classes2d_sparse_ids(tmp_path):
    """1-based Scipion IDs map directly to 0-based collection rows (cid - 1)."""
    import mrcfile

    mrcs_path = str(tmp_path / "sparse_particles.mrcs")
    db_path = str(tmp_path / "sparse_classes.sqlite")

    data = np.zeros((3, 16, 16), dtype=np.float32)
    mrcfile.write(mrcs_path, data, overwrite=True)

    soc = emobj.SetOfClasses2D(filename=db_path)
    soc.enableAppend()

    sparse_ids = [1, 4, 8]
    for i, cid in enumerate(sparse_ids):
        cls2d = emobj.Class2D()
        cls2d.setObjId(cid)
        soc.append(cls2d)
        p = emobj.Particle()
        p.setLocation(i + 1, mrcs_path)
        cls2d.append(p)
        soc.update(cls2d)

    soc.write()
    soc._getMapper().commit()

    bridge = res.resolve_set_of_classes2d_to_collection_classes2d(soc)
    assert isinstance(bridge, B.Collection)
    assert len(bridge) == 8  # max(1, 4, 8) -> size 8
    assert bridge.initialized_indices() == [0, 3, 7]
    assert bridge[0].class_id == 0
    assert bridge[3].class_id == 3
    assert bridge[7].class_id == 7


def test_resolve_collection_classes2d_to_set_of_classes2d(tmp_path):
    class _MockProtocol:
        def __init__(self, p):
            self.p = p

        def _createSetOfClasses2D(self, suffix=""):
            db_path = str(self.p / f"classes{suffix}.sqlite")
            return emobj.SetOfClasses2D(filename=db_path)

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    p1 = BParticle(
        pixels=np.zeros((16, 16), dtype=np.float32),
        sampling_rate=1.5,
    )
    coll = B.Collection[BClass2D](size=5)
    coll[0] = BClass2D(
        class_id=0,
        particles=B.Set[BParticle]([p1]),
        representative=p1,
    )
    coll[3] = BClass2D(
        class_id=3,
        particles=B.Set[BParticle]([p1]),
        representative=p1,
    )

    ctx = res.PyWorkflowResolutionContext(
        protocol=_MockProtocol(tmp_path),
        output_name="output_classes",
        append=False,
    )
    soc = res.resolve_collection_classes2d_to_set_of_classes2d(coll, metadata=ctx)
    assert isinstance(soc, emobj.SetOfClasses2D)
    assert len(soc) == 2

    # Check IDs of classes in the generated SetOfClasses2D
    class_ids = [cls2d.getObjId() for cls2d in soc]
    assert set(class_ids) == {1, 4}
    for cls2d in soc:
        cid = cls2d.getObjId()
        assert cls2d.getRepresentative().getClassId() == cid
        for p in cls2d:
            assert p.getClassId() == cid


def test_collection_classes2d_roundtrip(tmp_path):
    class _MockProtocol:
        def __init__(self, p):
            self.p = p

        def _createSetOfClasses2D(self, suffix=""):
            db_path = str(self.p / f"roundtrip{suffix}.sqlite")
            return emobj.SetOfClasses2D(filename=db_path)

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    soc_orig = _make_set_of_classes2d(tmp_path, n_classes=2, particles_per_class=2)
    coll = res.resolve_set_of_classes2d_to_collection_classes2d(soc_orig)
    assert len(coll) == 2

    ctx = res.PyWorkflowResolutionContext(
        protocol=_MockProtocol(tmp_path),
        output_name="roundtrip",
        append=False,
    )
    soc_back = res.resolve_collection_classes2d_to_set_of_classes2d(coll, metadata=ctx)
    assert len(soc_back) == 2

    coll_back = res.resolve_set_of_classes2d_to_collection_classes2d(soc_back)
    assert len(coll_back) == 2
    assert coll_back.initialized_indices() == [0, 1]
    assert len(coll_back[0].particles) == 2
    assert len(coll_back[1].particles) == 2


def test_collection_classes2d_accumulation_output_handler(tmp_path):
    """Test reduction algebra on disk: representative replaced, particles concatenated."""

    class _MockProtocol:
        def __init__(self, p):
            self.p = p

        def _createSetOfClasses2D(self, suffix=""):
            db_path = str(self.p / f"classes{suffix}.sqlite")
            return emobj.SetOfClasses2D(filename=db_path)

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    proto = _MockProtocol(tmp_path)

    p1 = BParticle(
        pixels=np.zeros((16, 16), dtype=np.float32),
        sampling_rate=1.0,
    )
    p2 = BParticle(
        pixels=np.ones((16, 16), dtype=np.float32),
        sampling_rate=2.0,
    )

    # Batch 1: slots 0, 3
    b1 = B.Collection[BClass2D](size=5)
    b1[0] = BClass2D(
        class_id=0,
        representative=p1,
        particles=B.Set[BParticle]([p1]),
    )
    b1[3] = BClass2D(
        class_id=3,
        representative=p1,
        particles=B.Set[BParticle]([p1]),
    )

    ctx1 = res.PyWorkflowResolutionContext(
        protocol=proto,
        output_name="output_classes",
        append=False,
    )
    soc1 = res.resolve_collection_classes2d_to_set_of_classes2d(b1, metadata=ctx1)
    assert len(soc1) == 2
    reduce_minibatch_to_persistent_output(proto, "output_classes", soc1)
    assert len(proto.output_classes) == 2

    # Batch 2: slots 3, 7 (representative replaced with p2, new particle p2)
    b2 = B.Collection[BClass2D](size=8)
    b2[3] = BClass2D(
        class_id=3,
        representative=p2,
        particles=B.Set[BParticle]([p2]),
    )
    b2[7] = BClass2D(
        class_id=7,
        representative=p2,
        particles=B.Set[BParticle]([p2]),
    )

    ctx2 = res.PyWorkflowResolutionContext(
        protocol=proto,
        output_name="output_classes",
        append=True,
    )
    soc2 = res.resolve_collection_classes2d_to_set_of_classes2d(b2, metadata=ctx2)
    assert len(soc2) == 2  # Minibatch only contains 2 classes (stateless)

    reduce_minibatch_to_persistent_output(proto, "output_classes", soc2)
    assert len(proto.output_classes) == 3  # Reduced on disk

    # Close and reopen from disk to verify true on-disk persistence
    proto.output_classes.close()
    db_path = str(tmp_path / "classes_output_classes.sqlite")
    verify_soc = emobj.SetOfClasses2D(filename=db_path)

    reps = {}
    class_lens = {}
    for c in verify_soc:
        cid = c.getObjId()
        reps[cid] = c.getRepresentative().clone()
        class_lens[cid] = len(c)
        for p in c:
            assert p.getClassId() == cid

    assert set(reps.keys()) == {1, 4, 8}

    # In class 1, original representative preserved (sr=1.0) and 1 particle
    assert reps[1].getSamplingRate() == pytest.approx(1.0)
    assert reps[1].getClassId() == 1

    # In class 4, representative replaced (sr=2.0) and particles concatenated (len=2)
    assert reps[4].getSamplingRate() == pytest.approx(2.0)
    assert reps[4].getClassId() == 4

    # In class 8, newly added class (sr=2.0, len=1)
    assert reps[8].getSamplingRate() == pytest.approx(2.0)
    assert reps[8].getClassId() == 8

    # Verify particle counts on disk
    assert class_lens == {1: 1, 4: 2, 8: 1}


def test_set_of_particles_streaming_append(tmp_path):
    """Test streaming batches of Set[Particle] appended incrementally to disk."""

    class _MockProtocol:
        def __init__(self, p):
            self.p = p

        def _createSetOfParticles(self, suffix=""):
            db_path = str(self.p / f"particles{suffix}.sqlite")
            return emobj.SetOfParticles(filename=db_path)

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    proto = _MockProtocol(tmp_path)

    # Batch 1: 3 particles
    p1 = BParticle(pixels=np.zeros((16, 16), dtype=np.float32), sampling_rate=1.0)
    p2 = BParticle(pixels=np.ones((16, 16), dtype=np.float32), sampling_rate=1.0)
    p3 = BParticle(pixels=np.ones((16, 16), dtype=np.float32) * 2, sampling_rate=1.0)
    batch1 = B.Set[BParticle]([p1, p2, p3])

    ctx1 = res.PyWorkflowResolutionContext(
        protocol=proto, output_name="particles", append=False
    )
    sop1 = res.resolve_bridge_particles_to_set_of_particles(batch1, metadata=ctx1)
    assert len(sop1) == 3
    reduce_minibatch_to_persistent_output(proto, "particles", sop1)
    assert len(proto.particles) == 3

    # Batch 2: 2 particles (minibatch, not accumulated)
    p4 = BParticle(pixels=np.ones((16, 16), dtype=np.float32) * 3, sampling_rate=1.5)
    p5 = BParticle(pixels=np.ones((16, 16), dtype=np.float32) * 4, sampling_rate=1.5)
    batch2 = B.Set[BParticle]([p4, p5])

    ctx2 = res.PyWorkflowResolutionContext(
        protocol=proto, output_name="particles", append=True
    )
    sop2 = res.resolve_bridge_particles_to_set_of_particles(batch2, metadata=ctx2)
    assert len(sop2) == 2  # Stateless: minibatch only contains 2 particles

    reduce_minibatch_to_persistent_output(proto, "particles", sop2)
    assert len(proto.particles) == 5

    proto.particles.close()
    db_path = str(tmp_path / "particles_particles.sqlite")
    verify_sop = emobj.SetOfParticles(filename=db_path)
    assert len(verify_sop) == 5

    # Check that particles read back properly
    bridge_sop = res.resolve_set_of_particles_to_bridge_particles(verify_sop)
    assert len(bridge_sop) == 5
    assert bridge_sop[0].sampling_rate == pytest.approx(1.0)
    assert bridge_sop[3].sampling_rate == pytest.approx(1.5)
    assert bridge_sop[4].sampling_rate == pytest.approx(1.5)


def test_set_of_particles_flex_streaming_append(tmp_path):
    """Test streaming batches of Set[FlexParticle] appended incrementally to disk."""

    class _MockProtocol:
        def __init__(self, p):
            self.p = p

        def _createSetOfParticlesFlex(self, suffix="", progName=""):
            db_path = str(self.p / f"flex{suffix}.sqlite")
            s = emobj.SetOfParticlesFlex(filename=db_path)
            s.getFlexInfo().setProgName(progName)
            return s

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    proto = _MockProtocol(tmp_path)

    # Batch 1: 2 flex particles
    fp1 = BFlexParticle(
        pixels=np.zeros((16, 16), dtype=np.float32),
        embeddings=np.array([0.1, 0.2], dtype=np.float32),
    )
    fp2 = BFlexParticle(
        pixels=np.ones((16, 16), dtype=np.float32),
        embeddings=np.array([0.3, 0.4], dtype=np.float32),
    )
    batch1 = B.Set[BFlexParticle]([fp1, fp2])

    ctx1 = res.PyWorkflowResolutionContext(
        protocol=proto, output_name="flex_particles", append=False
    )
    flex1 = res.resolve_embeddings_to_flex_particles(batch1, metadata=ctx1)
    assert len(flex1) == 2
    reduce_minibatch_to_persistent_output(proto, "flex_particles", flex1)
    assert len(proto.flex_particles) == 2

    # Batch 2: 1 flex particle
    fp3 = BFlexParticle(
        pixels=np.ones((16, 16), dtype=np.float32) * 2,
        embeddings=np.array([0.5, 0.6], dtype=np.float32),
    )
    batch2 = B.Set[BFlexParticle]([fp3])

    ctx2 = res.PyWorkflowResolutionContext(
        protocol=proto, output_name="flex_particles", append=True
    )
    flex2 = res.resolve_embeddings_to_flex_particles(batch2, metadata=ctx2)
    assert len(flex2) == 1  # Stateless minibatch

    reduce_minibatch_to_persistent_output(proto, "flex_particles", flex2)
    assert len(proto.flex_particles) == 3

    proto.flex_particles.close()
    db_path = str(tmp_path / "flex_flex_particles.sqlite")
    verify_flex = emobj.SetOfParticlesFlex(filename=db_path)
    assert len(verify_flex) == 3

    # Check that particles read back properly
    bridge_flex = res.resolve_set_of_particles_flex_to_bridge_particles(verify_flex)
    assert len(bridge_flex) == 3
    assert np.allclose(bridge_flex[0].embeddings, [0.1, 0.2])
    assert np.allclose(bridge_flex[1].embeddings, [0.3, 0.4])
    assert np.allclose(bridge_flex[2].embeddings, [0.5, 0.6])


class _DummyParticleProtocol(Protocol):
    input_particles: B.Input[B.Set[BParticle]]

    def outputs(self):
        return {"output_particles": B.Set[BParticle]}

    def steps(self):
        pass


def test_scipion_protocol_wrapper_write_output_data_handler(tmp_path):
    """Test ScipionProtocolWrapper._writeOutputDataHandler end-to-end reduction."""
    Wrapper = convert_protocol_to_scipion3_protocol(
        _DummyParticleProtocol(), label="test_protocol", conda_env="test_env"
    )
    w = Wrapper()
    w._workingDir = str(tmp_path)

    p1 = BParticle(pixels=np.zeros((16, 16), dtype=np.float32), sampling_rate=1.0)
    batch1 = B.Set[BParticle]([p1])
    w._writeOutputDataHandler({"output_particles": batch1})
    assert len(w.output_particles) == 1

    p2 = BParticle(pixels=np.ones((16, 16), dtype=np.float32), sampling_rate=2.0)
    batch2 = B.Set[BParticle]([p2])
    w._writeOutputDataHandler({"output_particles": batch2})
    assert len(w.output_particles) == 2

    # Close and verify on disk
    db_path = w.output_particles.getFileName()
    w.output_particles.close()
    verify = emobj.SetOfParticles(filename=db_path)
    assert len(verify) == 2
    bridge_verify = res.resolve_set_of_particles_to_bridge_particles(verify)
    assert bridge_verify[0].sampling_rate == pytest.approx(1.0)
    assert bridge_verify[1].sampling_rate == pytest.approx(2.0)


def test_collection_classes2d_to_sqlite_full_fields(tmp_path):
    """Test that resolving Collection[Class2D] writes MRCs and populates all SQLite fields."""
    import sqlite3
    from pathlib import Path

    class _MockPointer:
        def __init__(self, val):
            self._val = val

        def get(self):
            return self._val

        def hasValue(self):
            return True

    class _MockProtocol:
        def __init__(self, p):
            self.p = p
            flex_set = emobj.SetOfParticlesFlex(filename=str(p / "input_parts.sqlite"))
            flex_set.setSamplingRate(1.23)
            self.inputTypes = ["inputParticles"]
            self.inputParticles = _MockPointer(flex_set)

        def _createSetOfClasses2D(self, suffix=""):
            db_path = str(self.p / f"classes{suffix}.sqlite")
            return emobj.SetOfClasses2D(filename=db_path)

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    proto = _MockProtocol(tmp_path)

    # 1. Create a representative particle with pixels
    rep_pixels = np.ones((16, 16), dtype=np.float32) * 5.0
    rep = BParticle(pixels=rep_pixels, sampling_rate=1.23)

    # 2. Create Particles with CTF, Coordinates, and pixels
    p1 = BParticle(
        pixels=np.ones((16, 16), dtype=np.float32) * 1.0,
        sampling_rate=1.23,
    )
    p1.ctf.defocus_u = 10000.0
    p1.ctf.defocus_v = 11000.0
    p1.ctf.defocus_angle = 45.0
    p1.coordinate.x = 100.0
    p1.coordinate.y = 200.0

    p2 = BParticle(
        pixels=np.ones((16, 16), dtype=np.float32) * 2.0,
        sampling_rate=1.23,
    )
    p2.ctf.defocus_u = 12000.0
    p2.ctf.defocus_v = 13000.0
    p2.ctf.defocus_angle = 30.0
    p2.coordinate.x = 150.0
    p2.coordinate.y = 250.0

    part_set = B.Set[BParticle]([p1, p2])
    cls1 = BClass2D(class_id=0, representative=rep, particles=part_set)

    coll = B.Collection[BClass2D](size=2)
    coll[0] = cls1

    ctx = res.PyWorkflowResolutionContext(
        protocol=proto,
        output_name="test_classes",
        append=False,
    )
    soc = res.resolve_collection_classes2d_to_set_of_classes2d(coll, metadata=ctx)
    assert soc.getSamplingRate() == pytest.approx(1.23)
    images_ref = soc.getImages()
    db_file = soc.getFileName()
    soc.close()

    # Verify SQLite tables directly
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Check Objects table (classes)
    cur.execute("SELECT * FROM Objects;")
    obj_rows = cur.fetchall()
    assert len(obj_rows) == 1

    # Check representative filename in Objects table
    cur.execute(
        "SELECT column_name FROM Classes WHERE label_property='_representative._filename';"
    )
    rep_fn_col = cur.fetchone()[0]
    cur.execute(f"SELECT {rep_fn_col} FROM Objects;")
    rep_fn_val = cur.fetchone()[0]
    assert rep_fn_val is not None
    assert rep_fn_val.endswith(".mrc")
    assert (tmp_path / rep_fn_val).exists() or Path(rep_fn_val).exists()

    # Check representative classId in Objects table
    cur.execute(
        "SELECT column_name FROM Classes WHERE label_property='_representative._classId';"
    )
    rep_cid_col = cur.fetchone()[0]
    cur.execute(f"SELECT {rep_cid_col} FROM Objects;")
    assert cur.fetchone()[0] == 1

    # Check Class001_Objects table (particles in class 1)
    cur.execute("SELECT * FROM Class001_Objects;")
    part_rows = cur.fetchall()
    assert len(part_rows) == 2

    # Check particle classId in Class001_Objects
    cur.execute(
        "SELECT column_name FROM Class001_Classes WHERE label_property='_classId';"
    )
    part_cid_col = cur.fetchone()[0]
    cur.execute(f"SELECT {part_cid_col} FROM Class001_Objects;")
    assert [r[0] for r in cur.fetchall()] == [1, 1]

    # Check particle filename
    cur.execute(
        "SELECT column_name FROM Class001_Classes WHERE label_property='_filename';"
    )
    part_fn_col = cur.fetchone()[0]
    cur.execute(f"SELECT {part_fn_col} FROM Class001_Objects;")
    part_fn_vals = [r[0] for r in cur.fetchall()]
    assert all(fn is not None and fn.endswith(".mrcs") for fn in part_fn_vals)
    assert Path(part_fn_vals[0]).exists()

    # Check Particle type in Class001_Classes
    cur.execute("SELECT class_name FROM Class001_Classes WHERE column_name='c00';")
    assert cur.fetchone()[0] == "Particle"

    # Check CTF defocusU
    cur.execute(
        "SELECT column_name FROM Class001_Classes WHERE label_property='_ctfModel._defocusU';"
    )
    defocus_u_col = cur.fetchone()[0]
    cur.execute(f"SELECT {defocus_u_col} FROM Class001_Objects;")
    defocus_vals = [r[0] for r in cur.fetchall()]
    assert defocus_vals == [10000.0, 12000.0]

    # Check coordinates
    cur.execute(
        "SELECT column_name FROM Class001_Classes WHERE label_property='_coordinate._x';"
    )
    coord_x_col = cur.fetchone()[0]
    cur.execute(f"SELECT {coord_x_col} FROM Class001_Objects;")
    coord_x_vals = [r[0] for r in cur.fetchall()]
    assert coord_x_vals == [100, 150]

    conn.close()

    # Reopen with Scipion and check sampling rate, dimensions, and classId
    reopened = emobj.SetOfClasses2D(filename=db_file)
    if images_ref is not None:
        reopened.setImages(images_ref)
        assert reopened.getSamplingRate() == pytest.approx(1.23)
    first_cls = reopened.getFirstItem()
    assert first_cls.getObjId() == 1
    assert first_cls.getSamplingRate() == pytest.approx(1.23)
    assert first_cls.getRepresentative().getSamplingRate() == pytest.approx(1.23)
    assert first_cls.getRepresentative().getClassId() == 1
    assert first_cls.getRepresentative().getFileName() == rep_fn_val
    for p in first_cls:
        assert p.getClassId() == 1
    reopened.close()


def test_collection_20_classes_to_scipion_no_collision(tmp_path):
    """Verify that 20 0-based classes (0..19) resolve to exactly 20 1-based Scipion classes (1..20)."""

    class _MockProtocol:
        def __init__(self, p):
            self.p = p

        def _createSetOfClasses2D(self, suffix=""):
            db_path = str(self.p / f"classes20{suffix}.sqlite")
            return emobj.SetOfClasses2D(filename=db_path)

        def _getExtraPath(self, path=""):
            return str(self.p / path)

    proto = _MockProtocol(tmp_path)
    coll = B.Collection[BClass2D](size=20)

    for i in range(20):
        rep = BParticle(
            pixels=np.full((16, 16), float(i), dtype=np.float32),
            sampling_rate=1.0,
        )
        p = BParticle(
            pixels=np.full((16, 16), float(i + 100), dtype=np.float32),
            sampling_rate=1.0,
        )
        coll[i] = BClass2D(
            class_id=i,
            representative=rep,
            particles=B.Set[BParticle]([p]),
        )

    ctx = res.PyWorkflowResolutionContext(
        protocol=proto,
        output_name="twenty_classes",
        append=False,
    )
    soc = res.resolve_collection_classes2d_to_set_of_classes2d(coll, metadata=ctx)
    assert len(soc) == 20

    class_ids = [c.getObjId() for c in soc]
    assert sorted(class_ids) == list(range(1, 21))

    for c in soc:
        cid = c.getObjId()
        assert c.getRepresentative().getClassId() == cid
        for p in c:
            assert p.getClassId() == cid

    # Roundtrip back to bridge collection
    coll_back = res.resolve_set_of_classes2d_to_collection_classes2d(soc)
    assert len(coll_back) == 20
    assert coll_back.initialized_indices() == list(range(20))
    for i in range(20):
        assert coll_back[i].class_id == i
        assert len(coll_back[i].particles) == 1
    soc.close()
