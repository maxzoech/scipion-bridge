import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.entries import (
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaSetEntry,
    _StructEntry,
)
from scipion_bridge.core.struct.schema import Schema
from scipion_bridge.core.struct.set import generate_set_schema


class CTF(B.Struct):
    voltage_kv: float              # Accelerating voltage (typically 300.0 or 200.0 kV)
    amplitude_contrast: float      # Amplitude contrast fraction (typically 0.07 to 0.10)
    spherical_aberration_mm: float # Spherical aberration (Cs) of the objective lens in mm (e.g., 2.7)
    defocus_u: float               # Defocus along the major axis (usually in Angstroms)
    defocus_v: float               # Defocus along the minor axis (usually in Angstroms)
    defocus_angle: float           # Astigmatism angle between the U axis and X axis (degrees)
    phase_shift: float             # Phase shift (in degrees), usually 0.0 unless using a Volta Phase Plate


class Particle(B.Struct):
    pixels: B.Array
    ctf: CTF


class TiltSeries(B.Struct):
    tilts: B.Set[Particle]


def test_unsubscripted_set_schema_raises():
    with pytest.raises(TypeError, match="You must subscript Set"):
        B.Set.schema()

    with pytest.raises(TypeError, match="You must subscript Set"):
        B.Set.item_type()


def test_non_struct_set_schema_raises():
    with pytest.raises(TypeError, match="Element of a set has to be of type Struct"):
        B.Set[int].schema()

    with pytest.raises(TypeError, match="Element of a set has to be of type Struct"):
        generate_set_schema(int)  # type: ignore


def test_set_item_type_and_caching():
    assert B.Set[CTF].item_type() == CTF
    assert B.Set[CTF] is B.Set[CTF]


def test_static_struct_set_schema():
    schema = B.Set[CTF].schema()
    assert isinstance(schema, Schema)
    assert schema.is_static is True

    # Check that all fields in CTF set schema are _ArraySetEntry
    for field_name, entry in schema.fields.items():
        assert isinstance(entry, _ArraySetEntry), f"{field_name} should be _ArraySetEntry"
        assert entry.is_static is True
        assert entry.shape == (1,)
        assert entry.min_shape == (1,)
        assert entry.max_shape == (1,)


def test_ragged_struct_set_schema():
    schema = B.Set[Particle].schema()
    assert isinstance(schema, Schema)
    assert schema.is_static is False

    # pixels is B.Array -> dynamic -> _RaggedArraySetEntry
    pixels_entry = schema.fields["pixels"]
    assert isinstance(pixels_entry, _RaggedArraySetEntry)
    assert pixels_entry.is_static is False
    assert pixels_entry.min_shape is None
    assert pixels_entry.max_shape is None

    # ctf is nested CTF struct -> _StructEntry containing set-converted fields
    ctf_entry = schema.fields["ctf"]
    assert isinstance(ctf_entry, _StructEntry)
    assert ctf_entry.is_static is True
    for _, entry in ctf_entry.schema.fields.items():
        assert isinstance(entry, _ArraySetEntry)


def test_tiltseries_set_schema_integration():
    schema = TiltSeries.schema()
    assert isinstance(schema, Schema)
    assert "tilts" in schema.fields
    tilts_entry = schema.fields["tilts"]
    assert isinstance(tilts_entry, _SchemaSetEntry)
    assert tilts_entry.is_static is False

def test_set_of_tiltseries():
    schema = B.Set[TiltSeries].schema()
    assert isinstance(schema, Schema)
    assert "tilts" in schema.fields
    tilts_entry = schema.fields["tilts"]
    assert isinstance(tilts_entry, _SchemaSetEntry)
    assert tilts_entry.is_static is False

    B.Set[TiltSeries]().print_schema()

if __name__ == "__main__":
    test_set_of_tiltseries()