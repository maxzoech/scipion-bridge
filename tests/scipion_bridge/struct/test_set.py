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
    set_schema.print_tree()

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
        num_particles: B.Arg = B.Arg()

        average: B.Array[float] = B.Array(shape=(H, H))
        particles: B.Set[DynamicParticle] = B.Set[DynamicParticle](
            num_particles,
            H=H,
        )

    # 1. Specialize a single Class2D instance
    class2d_specialized = Class2D(H=64)
    assert not class2d_specialized.schema.is_static
    assert isinstance(class2d_specialized.schema.fields["average"], _ArrayEntryBase)
    assert class2d_specialized.schema.fields["average"].shape == (64, 64)


    # Inner particles set has dynamic capacity, but its elements have specialized shape (64, 64)
    particles_entry = class2d_specialized.schema.fields["particles"]
    assert isinstance(particles_entry, _SchemaSetEntry)
    assert particles_entry.capacity is None
    assert isinstance(particles_entry.schema.fields["pixels"], _ArrayEntryBase)
    assert particles_entry.schema.fields["pixels"].shape == (64, 64)

    # 2. Specialize outer Set capacity with a dynamic Dim and size
    classes_set = B.Set[Class2D](H=128, num_particles=100)
    assert classes_set.schema.is_static
    assert not classes_set.convert_to_entry().is_static

    classes_set.print_schema()
