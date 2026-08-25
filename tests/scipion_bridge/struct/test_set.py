import pytest
import numpy as np

import scipion_bridge as B
from scipion_bridge.core.struct.schema import (
    Schema,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaSetEntry,
    _SchemaEntry,
    _ArrayEntryBase
)


class CTF(B.Struct):
    voltage_kv: float              # Accelerating voltage (typically 300.0 or 200.0 kV)
    amplitude_contrast: float      # Amplitude contrast fraction (typically 0.07 to 0.10)
    # spherical_aberration_mm: float # Spherical aberration (Cs) of the objective lens in mm (e.g., 2.7)
    # defocus_u: float               # Defocus along the major axis (usually in Angstroms)
    # defocus_v: float               # Defocus along the minor axis (usually in Angstroms)
    # defocus_angle: float           # Astigmatism angle between the U axis and X axis (degrees)
    # phase_shift: float             # Phase shift (in degrees), usually 0.0 unless using a Volta Phase Plate


class Particle(B.Struct):
    pixels: B.Array[float] = B.Array(shape=(None, None))
    ctf: CTF = CTF()


class TiltSeries(B.Struct):
    tilts: B.Set[Particle]


def test_unsubscripted_set_schema_raises():
    with pytest.raises(TypeError, match="Cannot convert unsubscripted Set"):
        B.Set().convert_to_entry()


def test_non_struct_set_schema_raises():
    with pytest.raises(TypeError, match="Element of a set has to be of type Struct"):
        B.Set[int]() # type: ignore


def test_set_item_type_and_caching():
    assert B.Set[CTF]._dtype == CTF
    assert B.Set[CTF]().dtype == CTF
    assert B.Set[CTF] is B.Set[CTF]


def test_static_struct_set_schema():
    schema = B.Set[CTF]().schema
    assert isinstance(schema, Schema)
    assert schema.is_static is True

    # Check that all fields in CTF set schema are _ArraySetEntry
    for field_name, entry in schema.fields.items():
        assert isinstance(entry, _ArraySetEntry), f"{field_name} should be _ArraySetEntry"
        assert entry.is_static is True
        assert entry.shape == (1,)


def test_ragged_struct_set_schema():
    schema = B.Set[Particle]().schema
    assert isinstance(schema, Schema)
    assert schema.is_static is False

    # pixels is B.Array -> dynamic -> _RaggedArraySetEntry
    pixels_entry = schema.fields["pixels"]
    assert isinstance(pixels_entry, _RaggedArraySetEntry)
    assert pixels_entry.is_static is False

    # ctf is nested CTF struct -> _SchemaEntry containing set-converted fields
    ctf_entry = schema.fields["ctf"]
    assert isinstance(ctf_entry, _SchemaEntry)
    assert ctf_entry.is_static is True
    for _, entry in ctf_entry.schema.fields.items():
        assert isinstance(entry, _ArraySetEntry)


def test_tiltseries_set_schema_integration():
    schema = TiltSeries().schema
    assert isinstance(schema, Schema)
    assert "tilts" in schema.fields
    tilts_entry = schema.fields["tilts"]
    assert isinstance(tilts_entry, _SchemaSetEntry)
    assert tilts_entry.is_static is False


def test_set_of_tiltseries():
    schema = B.Set[TiltSeries]().schema
    assert isinstance(schema, Schema)
    assert "tilts" in schema.fields
    tilts_entry = schema.fields["tilts"]
    assert isinstance(tilts_entry, _SchemaSetEntry)
    assert tilts_entry.is_static is False


def test_set_of_classes2d_schema():
    class StaticParticle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(128, 128))
        voltage_kv: float

    class Class2D(B.Struct):
        average: B.Array[float] = B.Array(shape=(128, 128))
        particles: B.Set[StaticParticle] = B.Set[StaticParticle]()

    set_schema = B.Set[Class2D]().schema
    assert isinstance(set_schema, Schema)
    assert set_schema.is_static is False

    # average is static batch array (128, 128)
    avg_entry = set_schema.fields["average"]
    assert isinstance(avg_entry, _ArraySetEntry)
    assert avg_entry.is_static is True
    assert avg_entry.shape == (128, 128)

    # particles is nested ragged collection
    particles_entry = set_schema.fields["particles"]
    assert isinstance(particles_entry, _SchemaSetEntry)
    assert particles_entry.is_static is False

    # child schema of particles has pixels (ArraySet) and voltage_kv (ArraySet)
    p_schema = particles_entry.schema
    assert isinstance(p_schema.fields["pixels"], _ArraySetEntry)
    assert p_schema.fields["pixels"].shape == (128, 128)
    assert isinstance(p_schema.fields["voltage_kv"], _ArraySetEntry)
    assert p_schema.fields["voltage_kv"].shape == (1,)


def test_set_specialize_struct_dimensions_and_capacity():
    """Test specializing struct dimensions within an inner Set and specializing outer Set capacity."""
    class DynamicParticle(B.Struct):
        H: B.Dim = B.Dim()

        pixels: B.Array[float] = B.Array(shape=(H, H))
        voltage_kv: float

    class Class2D(B.Struct):
        H: B.Dim = B.Dim()
        num_particles: B.Dim = B.Dim()

        average: B.Array[float] = B.Array(shape=(H, H))
        particles: B.Set[DynamicParticle] = B.Set[DynamicParticle](
            capacity=num_particles,
            H=H,
        )

    # 1. Specialize a single Class2D instance
    class2d_specialized = Class2D(H=64)
    assert class2d_specialized.is_static()
    assert isinstance(class2d_specialized.schema.fields["average"], _ArrayEntryBase)
    assert class2d_specialized.schema.fields["average"].shape == (64, 64)

    # Inner particles set has dynamic capacity, but its elements have specialized shape (64, 64)
    particles_entry = class2d_specialized.schema.fields["particles"]
    assert isinstance(particles_entry, _SchemaSetEntry)
    assert particles_entry.capacity is None
    assert isinstance(particles_entry.schema.fields["pixels"], _ArrayEntryBase)
    assert particles_entry.schema.fields["pixels"].shape == (64, 64)

    # 2. Specialize outer Set capacity with a dynamic Dim
    classes_set = B.Set[Class2D](capacity=10, H=128, num_particles=100)
    classes_set.print_schema()


@pytest.mark.xfail(
    reason="Limitation: Unsubscripted B.Set() default field on Struct requires lazy schema creation via __set_name__"
)
def test_unsubscripted_set_default_field_in_struct():
    class LocalParticle(B.Struct):
        voltage_kv: float

    class TiltSeriesWithUnsubscriptedDefault(B.Struct):
        tilts: B.Set[LocalParticle] = B.Set()

    ts = TiltSeriesWithUnsubscriptedDefault()
    assert isinstance(ts.schema, Schema)
    assert "tilts" in ts.schema.fields


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_storage_simple_set():
    ctfs = B.Set[CTF](capacity=10)
    
    ctf1 = CTF(voltage_kv=300.0, amplitude_contrast=0.07)
    ctf2 = CTF(voltage_kv=200.0, amplitude_contrast=0.10)

    # Assign elements to specific indices
    ctfs[0] = ctf1
    ctfs[2] = ctf2
    
    # Assert values for assigned index 0
    assert ctfs[0].voltage_kv == 300.0
    assert ctfs[0].amplitude_contrast == 0.07
    
    # Assert values for unassigned index 1 (should default to 0.0 for float)
    assert ctfs[1].voltage_kv == 0.0
    assert ctfs[1].amplitude_contrast == 0.0
    
    # Assert values for assigned index 2
    assert ctfs[2].voltage_kv == 200.0
    assert ctfs[2].amplitude_contrast == 0.10


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_get_slice_basic():
    ctfs = B.Set[CTF](capacity=5)
    for i in range(5):
        ctfs[i] = CTF(voltage_kv=100.0 + i * 10, amplitude_contrast=0.01 * (i + 1))

    sliced = ctfs[1:4]
    assert isinstance(sliced, B.Set)
    assert len(sliced) == 3
    assert sliced[0].voltage_kv == 110.0
    assert sliced[1].voltage_kv == 120.0
    assert sliced[2].voltage_kv == 130.0
    assert pytest.approx(sliced[0].amplitude_contrast) == 0.02
    assert pytest.approx(sliced[2].amplitude_contrast) == 0.04


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_get_slice_defaults_and_negative_indices():
    ctfs = B.Set[CTF](capacity=5)
    for i in range(5):
        ctfs[i] = CTF(voltage_kv=200.0 + i, amplitude_contrast=0.1)

    # Implicit start
    start_slice = ctfs[:2]
    assert len(start_slice) == 2
    assert start_slice[0].voltage_kv == 200.0
    assert start_slice[1].voltage_kv == 201.0

    # Implicit stop
    stop_slice = ctfs[3:]
    assert len(stop_slice) == 2
    assert stop_slice[0].voltage_kv == 203.0
    assert stop_slice[1].voltage_kv == 204.0

    # Negative indices (-4 to -1 -> indices 1 to 4)
    neg_slice = ctfs[-4:-1]
    assert len(neg_slice) == 3
    assert neg_slice[0].voltage_kv == 201.0
    assert neg_slice[1].voltage_kv == 202.0
    assert neg_slice[2].voltage_kv == 203.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_slice_basic():
    ctfs = B.Set[CTF](capacity=5)
    for i in range(5):
        ctfs[i] = CTF(voltage_kv=100.0, amplitude_contrast=0.05)

    replacement = B.Set[CTF](capacity=2)
    replacement[0] = CTF(voltage_kv=300.0, amplitude_contrast=0.07)
    replacement[1] = CTF(voltage_kv=400.0, amplitude_contrast=0.08)

    ctfs[1:3] = replacement

    assert ctfs[0].voltage_kv == 100.0
    assert ctfs[1].voltage_kv == 300.0
    assert ctfs[2].voltage_kv == 400.0
    assert ctfs[3].voltage_kv == 100.0
    assert ctfs[4].voltage_kv == 100.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_slice_from_get_slice():
    source = B.Set[CTF](capacity=5)
    for i in range(5):
        source[i] = CTF(voltage_kv=10.0 * i, amplitude_contrast=0.01 * i)

    target = B.Set[CTF](capacity=5)

    # Assign a slice of source to a slice of target
    target[1:4] = source[2:5]

    assert target[1].voltage_kv == 20.0
    assert target[2].voltage_kv == 30.0
    assert target[3].voltage_kv == 40.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_slice_type_and_capacity_mismatch_errors():
    ctfs = B.Set[CTF](capacity=5)

    # Capacity mismatch error
    replacement_wrong_cap = B.Set[CTF](capacity=3)
    with pytest.raises(ValueError, match="Cannot assign a Set of capacity 3 to a slice of length 2"):
        ctfs[1:3] = replacement_wrong_cap

    # Type mismatch error
    with pytest.raises(TypeError, match="Cannot assign 'list' to a slice of Set"):
        ctfs[1:3] = [1, 2]  # type: ignore


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_iter():
    ctfs = B.Set[CTF](capacity=3)
    ctfs[0] = CTF(voltage_kv=100.0, amplitude_contrast=0.01)
    ctfs[1] = CTF(voltage_kv=200.0, amplitude_contrast=0.02)
    ctfs[2] = CTF(voltage_kv=300.0, amplitude_contrast=0.03)

    items = list(ctfs)
    assert len(items) == 3
    assert [item.voltage_kv for item in items] == [100.0, 200.0, 300.0]


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_indexing_out_of_bounds():
    ctfs = B.Set[CTF](capacity=3)

    assert ctfs[-1].voltage_kv == 0.0

    with pytest.raises(IndexError, match="out of range"):
        _ = ctfs[3]

    with pytest.raises(IndexError, match="out of range"):
        _ = ctfs[-4]

    with pytest.raises(IndexError, match="out of range"):
        ctfs[3] = CTF(voltage_kv=100.0, amplitude_contrast=0.01)


class Camera(B.Struct):
    gain: float
    pixel_size_A: float


class StaticParticle(B.Struct):
    ctf: CTF = CTF()
    camera: Camera = Camera()


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_nested_struct_get_set_element():
    particles = B.Set[StaticParticle](capacity=3)
    p0 = StaticParticle(
        ctf=CTF(voltage_kv=300.0, amplitude_contrast=0.07),
        camera=Camera(gain=1.5, pixel_size_A=0.85),
    )
    p1 = StaticParticle(
        ctf=CTF(voltage_kv=200.0, amplitude_contrast=0.10),
        camera=Camera(gain=2.0, pixel_size_A=1.05),
    )

    particles[0] = p0
    particles[1] = p1

    res0 = particles[0]
    assert isinstance(res0, StaticParticle)
    assert isinstance(res0.ctf, CTF)
    assert res0.ctf.voltage_kv == 300.0
    assert pytest.approx(res0.ctf.amplitude_contrast) == 0.07
    assert res0.camera.gain == 1.5
    assert res0.camera.pixel_size_A == 0.85

    res1 = particles[1]
    assert res1.ctf.voltage_kv == 200.0
    assert res1.camera.gain == 2.0


class InnerStruct(B.Struct):
    val: float


class MiddleStruct(B.Struct):
    inner: InnerStruct = InnerStruct()


class OuterStruct(B.Struct):
    middle: MiddleStruct = MiddleStruct()


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_multilevel_nested_struct_get_set():
    outer_set = B.Set[OuterStruct](capacity=2)
    elem = OuterStruct(middle=MiddleStruct(inner=InnerStruct(val=42.0)))

    outer_set[0] = elem

    res = outer_set[0]
    assert isinstance(res, OuterStruct)
    assert isinstance(res.middle, MiddleStruct)
    assert isinstance(res.middle.inner, InnerStruct)
    assert res.middle.inner.val == 42.0


class SimpleParticle(B.Struct):
    voltage_kv: float


class TiltSeriesStatic(B.Struct):
    tilts: B.Set[SimpleParticle, 10]


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_nested_struct_slicing():
    particles = B.Set[StaticParticle](capacity=4)
    for i in range(4):
        particles[i] = StaticParticle(
            ctf=CTF(voltage_kv=100.0 * (i + 1), amplitude_contrast=0.01 * (i + 1)),
            camera=Camera(gain=1.0 + i, pixel_size_A=0.5 + i),
        )

    # 1. Getting a slice of nested struct Set
    sliced = particles[1:3]
    assert len(sliced) == 2
    assert isinstance(sliced[0], StaticParticle)
    assert sliced[0].ctf.voltage_kv == 200.0
    assert pytest.approx(sliced[0].ctf.amplitude_contrast) == 0.02
    assert sliced[0].camera.gain == 2.0

    assert sliced[1].ctf.voltage_kv == 300.0
    assert sliced[1].camera.gain == 3.0

    # 2. Setting a slice of nested struct Set
    replacement = B.Set[StaticParticle](capacity=2)
    replacement[0] = StaticParticle(
        ctf=CTF(voltage_kv=500.0, amplitude_contrast=0.05),
        camera=Camera(gain=5.0, pixel_size_A=5.5),
    )
    replacement[1] = StaticParticle(
        ctf=CTF(voltage_kv=600.0, amplitude_contrast=0.06),
        camera=Camera(gain=6.0, pixel_size_A=6.5),
    )

    particles[1:3] = replacement
    assert particles[0].ctf.voltage_kv == 100.0
    assert particles[1].ctf.voltage_kv == 500.0
    assert particles[2].ctf.voltage_kv == 600.0
    assert particles[3].ctf.voltage_kv == 400.0

    # 3. Assigning sliced source to sliced target
    target = B.Set[StaticParticle](capacity=4)
    target[0:2] = particles[1:3]
    assert target[0].ctf.voltage_kv == 500.0
    assert target[1].ctf.voltage_kv == 600.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_nested_set_field_get_set_element():
    series_set = B.Set[TiltSeriesStatic](capacity=2)

    p_set0 = B.Set[SimpleParticle](capacity=3)
    p_set0[0] = SimpleParticle(voltage_kv=300.0)
    p_set0[1] = SimpleParticle(voltage_kv=200.0)
    p_set0[2] = SimpleParticle(voltage_kv=100.0)

    ts0 = TiltSeriesStatic(tilts=p_set0)
    series_set[0] = ts0

    # Retrieve single element ts0 from series_set
    retrieved_ts0 = series_set[0]
    assert isinstance(retrieved_ts0, TiltSeriesStatic)
    assert isinstance(retrieved_ts0.tilts, B.Set)
    assert len(retrieved_ts0.tilts) == 10
    assert retrieved_ts0.tilts[0].voltage_kv == 300.0
    assert retrieved_ts0.tilts[1].voltage_kv == 200.0
    assert retrieved_ts0.tilts[2].voltage_kv == 100.0


# class LeafStruct(B.Struct):
#     val: float


# class NodeStruct(B.Struct):
#     leaves: B.Set[LeafStruct, 3]
#     bar: int


# class RootStruct(B.Struct):
#     nodes: B.Set[NodeStruct, 5]
#     foo: float


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_deeply_nested_sets():
    # 3 levels of sets: Set[RootStruct] -> NodeStruct (Set[LeafStruct])
    root_set = B.Set[RootStruct](capacity=10)
    assert len(root_set) == 10

    # Create source set to assign into slice
    source_set = B.Set[RootStruct](capacity=4)
    assert len(source_set) == 4

    # Perform slice assignment and retrieval using public API
    root_set[2:6] = source_set

    sliced = root_set[2:6]
    assert isinstance(sliced, B.Set)
    assert len(sliced) == 4

    # Verify element indexing
    first_elem = root_set[2]
    assert isinstance(first_elem, RootStruct)

    # Verify iteration over sliced set
    sliced_items = list(sliced)
    assert len(sliced_items) == 4

    # Verify out of bounds check
    with pytest.raises(IndexError):
        _ = root_set[10]


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_nested_struct_attribute_mutation():
    particles = B.Set[StaticParticle](capacity=3)
    p0 = StaticParticle(
        ctf=CTF(voltage_kv=300.0, amplitude_contrast=0.07),
        camera=Camera(gain=1.5, pixel_size_A=0.85),
    )
    particles[0] = p0

    # Retrieve element and verify initial attribute value
    res0 = particles[0]
    assert res0.ctf.voltage_kv == 300.0

    # Mutate attribute on nested child struct
    res0.ctf.voltage_kv = 400.0
    assert res0.ctf.voltage_kv == 400.0

    # Re-assign mutated struct back into Set and verify updated storage
    particles[0] = res0
    retrieved = particles[0]
    assert retrieved.ctf.voltage_kv == 400.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_nested_set_attribute_mutation():
    series_set = B.Set[TiltSeriesStatic](capacity=2)

    p_set0 = B.Set[SimpleParticle](capacity=3)
    p_set0[0] = SimpleParticle(voltage_kv=300.0)
    p_set0[1] = SimpleParticle(voltage_kv=200.0)
    p_set0[2] = SimpleParticle(voltage_kv=100.0)

    ts0 = TiltSeriesStatic(tilts=p_set0)
    series_set[0] = ts0

    # Retrieve ts0 from series_set and verify initial element value in nested Set
    retrieved_ts0 = series_set[0]
    assert retrieved_ts0.tilts[0].voltage_kv == 300.0

    # Mutate element inside nested Set
    retrieved_ts0.tilts[0] = SimpleParticle(voltage_kv=450.0)
    assert retrieved_ts0.tilts[0].voltage_kv == 450.0

    # Re-assign mutated struct back into series_set and verify updated storage
    series_set[0] = retrieved_ts0
    updated_ts0 = series_set[0]
    assert updated_ts0.tilts[0].voltage_kv == 450.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_unsubscripted_set_raises_type_error():
    with pytest.raises(TypeError, match="subscript Set"):
        _ = B.Set.item_type()

    with pytest.raises(TypeError, match="subscript Set"):
        _ = B.Set.capacity()


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_stride_slicing_raises_not_implemented():
    ctfs = B.Set[CTF](capacity=5)
    with pytest.raises(NotImplementedError, match="stride"):
        _ = ctfs[0:5:2]


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_element_assignment_type_errors():
    ctfs = B.Set[CTF](capacity=3)

    # 1. Assigning a Set object to an element
    with pytest.raises(TypeError, match="Cannot assign Set of"):
        ctfs[0] = B.Set[CTF](capacity=2)

    # 2. Assigning an incompatible struct type to an element
    with pytest.raises(TypeError, match="Cannot assign 'Camera'"):
        ctfs[0] = Camera(gain=1.5, pixel_size_A=0.85)


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_slice_assignment_type_errors():
    ctfs = B.Set[CTF](capacity=3)

    # 1. Assigning a non-Set object to slice
    with pytest.raises(TypeError, match="Cannot assign"):
        ctfs[0:2] = "invalid_value"

    # 2. Assigning incompatible Set type to slice
    with pytest.raises(TypeError, match="Cannot assign"):
        ctfs[0:2] = B.Set[Camera](capacity=2)


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_nested_set_missing_capacity_raises_value_error():
    class InvalidNestedStruct(B.Struct):
        nodes: B.Set[SimpleParticle]  # Missing capacity parameter

    with pytest.raises(ValueError, match="requires an explicit capacity"):
        _ = B.Set[InvalidNestedStruct](capacity=5)


class StaticParticle2D(B.Struct):
    pixels: B.Array[float] = B.Array(shape=(256, 256))
    ctf: CTF = CTF()


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_struct_instance_static_arrays():

    particles = B.Set[StaticParticle2D](capacity=10)
    data_1 = np.random.uniform(size=[256, 256])
    data_2 = np.random.uniform(size=[256, 256])

    # particles.print_storage_info()

    particles[0].pixels = data_1
    particles[1].pixels = data_2

    assert particles[0].pixels.shape == (256, 256)
    assert np.allclose(particles[0].pixels, data_1)
    assert np.allclose(particles[1].pixels, data_2)

    subset = particles[:2]
    assert np.allclose(subset[-1].pixels, data_2)


# class DeepTiltSeries(B.Struct):
#     tilts: B.Set[StaticParticle, 10]


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_deeply_nested_multi_level_attribute_mutation():
    # Set -> DeepTiltSeries -> Set[StaticParticle, 10] -> StaticParticle -> CTF -> voltage_kv
    series_set = B.Set[DeepTiltSeries](capacity=2)

    p_set = B.Set[StaticParticle](capacity=3)
    p_set[0] = StaticParticle(
        ctf=CTF(voltage_kv=300.0, amplitude_contrast=0.07),
        camera=Camera(gain=1.5, pixel_size_A=0.85),
    )
    p_set[1] = StaticParticle(
        ctf=CTF(voltage_kv=200.0, amplitude_contrast=0.10),
        camera=Camera(gain=2.0, pixel_size_A=1.00),
    )

    ts0 = DeepTiltSeries(tilts=p_set)
    series_set[0] = ts0

    # 1. Verify initial deep nested values
    assert series_set[0].tilts[1].ctf.voltage_kv == 200.0
    assert series_set[0].tilts[1].camera.gain == 2.0

    # 2. Mutate deeply nested attribute directly on retrieved proxy view
    series_set[0].tilts[1].ctf.voltage_kv = 400.0
    series_set[0].tilts[1].camera.gain = 3.5

    assert series_set[0].tilts[1].ctf.voltage_kv == 400.0
    assert series_set[0].tilts[1].camera.gain == 3.5


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_init_from_elements():
    ctf1 = CTF(voltage_kv=300.0, amplitude_contrast=0.07)
    ctf2 = CTF(voltage_kv=200.0, amplitude_contrast=0.10)

    # Positional list initialization
    s1 = B.Set[CTF]([ctf1, ctf2])
    assert len(s1) == 2
    assert s1[0].voltage_kv == 300.0
    assert s1[1].voltage_kv == 200.0

    # Keyword capacity + elements
    s2 = B.Set[CTF]([ctf1], capacity=5)
    assert len(s2) == 5
    assert s2[0].voltage_kv == 300.0
    assert s2[1].voltage_kv == 0.0

    # Errors
    with pytest.raises(ValueError, match="Must specify capacity"):
        B.Set[CTF]()

    with pytest.raises(ValueError, match="smaller than number of elements"):
        B.Set[CTF]([ctf1, ctf2], capacity=1)


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_concat():
    ctf1 = CTF(voltage_kv=300.0, amplitude_contrast=0.07)
    ctf2 = CTF(voltage_kv=200.0, amplitude_contrast=0.10)
    ctf3 = CTF(voltage_kv=100.0, amplitude_contrast=0.05)

    s1 = B.Set[CTF]([ctf1, ctf2])
    s2 = B.Set[CTF]([ctf3])

    combined = B.Set[CTF].concat(s1, s2)
    assert isinstance(combined, B.Set)
    assert len(combined) == 3
    assert combined[0].voltage_kv == 300.0
    assert combined[1].voltage_kv == 200.0
    assert combined[2].voltage_kv == 100.0


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_string_slicing_getitem():
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(16, 16))

    s = B.Set[Particle](capacity=3)
    p0_pixels = np.ones((16, 16), dtype=np.float32) * 1.0
    p1_pixels = np.ones((16, 16), dtype=np.float32) * 2.0
    p2_pixels = np.ones((16, 16), dtype=np.float32) * 3.0

    s[0].pixels = p0_pixels
    s[1].pixels = p1_pixels
    s[2].pixels = p2_pixels

    batched_pixels = s["pixels"]
    
    assert batched_pixels is not None
    assert batched_pixels.shape == (3, 16, 16)
    assert np.allclose(batched_pixels[0], p0_pixels)
    assert np.allclose(batched_pixels[1], p1_pixels)
    assert np.allclose(batched_pixels[2], p2_pixels)

    # Test string slicing on a Set slice view
    sub_slice = s[1:3]
    sub_batched_pixels = sub_slice["pixels"]
    assert sub_batched_pixels is not None
    assert sub_batched_pixels.shape == (2, 16, 16)
    assert np.allclose(sub_batched_pixels[0], p1_pixels)
    assert np.allclose(sub_batched_pixels[1], p2_pixels)


@pytest.mark.skip(reason="Data storage and container operations not yet implemented")
def test_set_string_slicing_setitem():
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(16, 16))

    s = B.Set[Particle](capacity=3)

    new_pixels = np.zeros((3, 16, 16), dtype=np.float32)
    new_pixels[0] = 10.0
    new_pixels[1] = 20.0
    new_pixels[2] = 30.0

    # Set batched array via string indexing on Set
    s["pixels"] = new_pixels

    # Check individual element access
    assert np.allclose(s[0].pixels, new_pixels[0])
    assert np.allclose(s[1].pixels, new_pixels[1])
    assert np.allclose(s[2].pixels, new_pixels[2])

    # Test setting via string slicing on a Set slice view
    sub_pixels = np.zeros((2, 16, 16), dtype=np.float32)
    sub_pixels[0] = 50.0
    sub_pixels[1] = 60.0

    s[1:3]["pixels"] = sub_pixels

    assert np.allclose(s[0].pixels, new_pixels[0])
    assert np.allclose(s[1].pixels, sub_pixels[0])
    assert np.allclose(s[2].pixels, sub_pixels[1])


if __name__ == "__main__":
    test_set_specialize_struct_dimensions_and_capacity()