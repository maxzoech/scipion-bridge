import pytest
import numpy as np

import scipion_bridge as B
from scipion_bridge.core.struct.schema import (
    Schema,
    _ArrayEntry,
    _ArraySetEntry,
    _RaggedArraySetEntry,
    _SchemaSetEntry,
    _SchemaEntry,
    _ArrayEntryBase
)


def test_basic_set():
    
    class Metadata(B.Struct):
        val_float: float
        val_int: int
        val_complex: np.complex128

    class Foo(B.Struct):
        bar: int
        metadata: Metadata

    set_schema = B.Set[Foo].schema()
    assert isinstance(set_schema, Schema)
    assert set_schema.is_static is True
    assert set(set_schema.fields.keys()) == {"bar", "metadata"}

    # bar field in Set context converts to _ArraySetEntry
    bar_entry = set_schema.fields["bar"]
    assert isinstance(bar_entry, _ArraySetEntry)
    assert bar_entry.dtype == np.dtype(int)
    assert bar_entry.shape == (1,)
    assert bar_entry.is_static is True

    # metadata field in Set context converts to _SchemaEntry wrapping set schema
    metadata_entry = set_schema.fields["metadata"]
    assert isinstance(metadata_entry, _SchemaEntry)
    assert metadata_entry.is_static is True
    
    meta_schema = metadata_entry.schema
    assert isinstance(meta_schema, Schema)
    assert meta_schema.is_static is True
    assert set(meta_schema.fields.keys()) == {"val_float", "val_int", "val_complex"}

    val_float = meta_schema.fields["val_float"]
    assert isinstance(val_float, _ArraySetEntry)
    assert val_float.dtype == np.dtype(float)
    assert val_float.shape == (1,)
    assert val_float.is_static is True

    val_int = meta_schema.fields["val_int"]
    assert isinstance(val_int, _ArraySetEntry)
    assert val_int.dtype == np.dtype(int)
    assert val_int.shape == (1,)
    assert val_int.is_static is True

    val_complex = meta_schema.fields["val_complex"]
    assert isinstance(val_complex, _ArraySetEntry)
    assert val_complex.dtype == np.dtype(np.complex128)
    assert val_complex.shape == (1,)
    assert val_complex.is_static is True


def test_nested_sets():

    class Element(B.Struct):
        foo: float

    class Foo(B.Struct):
        bar: int
        data = B.Set[Element](capacity=10)

    schema = Foo.schema()
    assert isinstance(schema, Schema)
    assert schema.is_static is True
    assert set(schema.fields.keys()) == {"bar", "data"}

    # bar field is a standard scalar array field on Foo
    bar_entry = schema.fields["bar"]
    assert isinstance(bar_entry, _ArrayEntry)
    assert bar_entry.dtype == np.dtype(int)
    assert bar_entry.shape == (1,)
    assert bar_entry.is_static is True

    # data field is a nested Set entry with capacity 10
    data_entry = schema.fields["data"]
    assert isinstance(data_entry, _SchemaSetEntry)
    assert data_entry.capacity == 10
    assert data_entry.is_static is True

    # Element inner schema fields are converted to _ArraySetEntry
    elem_schema = data_entry.schema
    assert isinstance(elem_schema, Schema)
    assert elem_schema.is_static is True
    assert set(elem_schema.fields.keys()) == {"foo"}

    foo_entry = elem_schema.fields["foo"]
    assert isinstance(foo_entry, _ArraySetEntry)
    assert foo_entry.dtype == np.dtype(float)
    assert foo_entry.shape == (1,)
    assert foo_entry.is_static is True



def test_non_struct_set_schema_raises():
    with pytest.raises(TypeError, match="Element of a set has to be a Struct"):
        B.Set[int]  # type: ignore


def test_set_item_type_and_caching():
    class CTF(B.Struct):
        voltage_kv: float

    assert B.Set[CTF]._dtype == CTF
    assert B.Set[CTF]().dtype == CTF
    assert B.Set[CTF] is B.Set[CTF]


def test_static_struct_set_schema():
    class CTF(B.Struct):
        voltage_kv: float
        amplitude_contrast: float

    schema = B.Set[CTF].schema()
    assert isinstance(schema, Schema)
    assert schema.is_static is True

    # Check that all fields in CTF set schema are _ArraySetEntry
    for field_name, entry in schema.fields.items():
        assert isinstance(entry, _ArraySetEntry), f"{field_name} should be _ArraySetEntry"
        assert entry.is_static is True
        assert entry.shape == (1,)


def test_ragged_struct_set_schema():
    class CTF(B.Struct):
        voltage_kv: float

    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None, None))
        ctf: CTF

    schema = B.Set[Particle].schema()
    assert isinstance(schema, Schema)
    assert schema.is_static is False

    # pixels is dynamic B.Array -> _RaggedArraySetEntry
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
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None, None))

    class TiltSeries(B.Struct):
        tilts: B.Set[Particle]

    schema = TiltSeries.schema()
    assert isinstance(schema, Schema)
    assert "tilts" in schema.fields
    tilts_entry = schema.fields["tilts"]
    assert isinstance(tilts_entry, _SchemaSetEntry)
    assert tilts_entry.is_static is False


def test_set_of_tiltseries():
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None, None))

    class TiltSeries(B.Struct):
        tilts: B.Set[Particle]

    schema = B.Set[TiltSeries].schema()
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

    set_schema = B.Set[Class2D].schema()
    assert isinstance(set_schema, Schema)
    assert set_schema.is_static is False

    # average is static batch array (128, 128)
    avg_entry = set_schema.fields["average"]
    assert isinstance(avg_entry, _ArraySetEntry)
    assert avg_entry.is_static is True
    assert avg_entry.shape == (128, 128)

    # particles is nested collection
    particles_entry = set_schema.fields["particles"]
    assert isinstance(particles_entry, _SchemaSetEntry)
    assert particles_entry.is_static is False

    # child schema of particles has pixels (ArraySet) and voltage_kv (ArraySet)
    p_schema = particles_entry.schema
    assert isinstance(p_schema.fields["pixels"], _ArraySetEntry)
    assert p_schema.fields["pixels"].shape == (128, 128)
    assert isinstance(p_schema.fields["voltage_kv"], _ArraySetEntry)
    assert p_schema.fields["voltage_kv"].shape == (1,)


def test_set_of_specialized_struct():
    class DynamicParticle(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, H))

    SpecParticle = DynamicParticle.static(H=64)
    spec_set_schema = B.Set[SpecParticle].schema()
    assert spec_set_schema.is_static is True
    assert isinstance(spec_set_schema.fields["pixels"], _ArraySetEntry)
    assert spec_set_schema.fields["pixels"].shape == (64, 64)


def test_set_capacity_static_vs_dynamic():
    class Item(B.Struct):
        val: float

    dyn_entry = B.Set[Item]().convert_to_entry()
    assert isinstance(dyn_entry, _SchemaSetEntry)
    assert dyn_entry.capacity is None
    assert dyn_entry.is_static is False

    fixed_entry = B.Set[Item](capacity=10).convert_to_entry()
    assert isinstance(fixed_entry, _SchemaSetEntry)
    assert fixed_entry.capacity == 10
    assert fixed_entry.is_static is True


def test_set_capacity_with_fixed_arg():
    class Element(B.Struct):
        val: float

    class Container(B.Struct):
        N = B.Arg(10)
        items = B.Set[Element](capacity=N)

    schema = Container.schema()
    assert schema.is_static is True
    items_entry = schema.fields["items"]
    assert isinstance(items_entry, _SchemaSetEntry)
    assert items_entry.capacity == 10
    assert items_entry.is_static is True


def test_set_capacity_specialization_with_dynamic_arg():
    class Element(B.Struct):
        val: float

    class Container(B.Struct):
        N = B.Arg()
        items = B.Set[Element](capacity=N)

    # 1. Base struct has dynamic capacity (None)
    items_entry_base = Container.schema().fields["items"]
    assert isinstance(items_entry_base, _SchemaSetEntry)
    assert items_entry_base.capacity is None
    assert not items_entry_base.is_static
    assert not Container.schema().is_static

    # 2. Specializing N on Container via .static() propagates to Set capacity
    Container10 = Container.static(N=10)
    items_entry_10 = Container10.schema().fields["items"]
    assert isinstance(items_entry_10, _SchemaSetEntry)
    assert items_entry_10.capacity == 10
    assert items_entry_10.is_static is True
    assert Container10.schema().is_static is True

    # 3. Specializing N on Container via subclass specializations={"N": 20}
    class Container20(Container, specializations={"N": 20}):
        pass

    items_entry_20 = Container20.schema().fields["items"]
    assert isinstance(items_entry_20, _SchemaSetEntry)
    assert items_entry_20.capacity == 20
    assert items_entry_20.is_static is True
    assert Container20.schema().is_static is True


# Storage Tests

def test_basic_set_storage():

    class Foo(B.Struct):
        bar: int

    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))
        foo: float
        bar: float

    noise = np.random.uniform(size=[5, 128, 128])

    data = B.Set[Data](capacity=64)
    data["pixels"] = noise

    assert np.allclose(noise, data["pixels"]) # type: ignore


def test_set_storage_shape_mismatch_raises():
    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))

    data = B.Set[Data](capacity=64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        data["pixels"] = np.random.uniform(size=[5, 64, 64])


@pytest.mark.xfail(reason="Triage verification for now")
def test_set_storage_capacity_exceeded_raises():
    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))

    data = B.Set[Data](capacity=64)
    with pytest.raises(ValueError, match="exceeds capacity 64"):
        data["pixels"] = np.random.uniform(size=[100, 128, 128])


def test_set_storage_incompatible_dtype_raises():
    class Data(B.Struct):
        foo: float

    data = B.Set[Data](capacity=64)
    with pytest.raises(TypeError, match="Cannot cast data of dtype"):
        data["foo"] = np.array([1.0 + 2.0j], dtype=np.complex128)


@pytest.mark.xfail(reason="Triage verification for now")
def test_set_storage_unspecified_capacity_raises():
    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))

    data = B.Set[Data]()
    with pytest.raises(ValueError, match="Capacity must be explicitly specified"):
        data["pixels"] = np.random.uniform(size=[5, 128, 128])


if __name__ == "__main__":
    from scipion_bridge.backend.standalone.container import configure_default_env
    
    # Wire the container for 'scipion_bridge'
    container = configure_default_env()
    
    test_basic_set_storage()
    # test_set_storage_shape_mismatch_raises()
    # test_set_storage_capacity_exceeded_raises()
    # test_set_storage_incompatible_dtype_raises()
    # test_set_storage_unspecified_capacity_raises()