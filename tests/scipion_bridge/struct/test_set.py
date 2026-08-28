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



# def test_unsubscripted_set_schema_raises():
#     with pytest.raises(TypeError, match="Cannot convert unsubscripted Set"):
#         B.Set().convert_to_entry()


# def test_non_struct_set_schema_raises():
#     with pytest.raises(TypeError, match="Element of a set has to be of type Struct"):
#         B.Set[int]() # type: ignore


# def test_set_item_type_and_caching():
#     assert B.Set[CTF]._dtype == CTF
#     assert B.Set[CTF]().dtype == CTF
#     assert B.Set[CTF] is B.Set[CTF]


# def test_static_struct_set_schema():
#     schema = B.Set[CTF]().schema
#     assert isinstance(schema, Schema)
#     assert schema.is_static is True

#     # Check that all fields in CTF set schema are _ArraySetEntry
#     for field_name, entry in schema.fields.items():
#         assert isinstance(entry, _ArraySetEntry), f"{field_name} should be _ArraySetEntry"
#         assert entry.is_static is True
#         assert entry.shape == (1,)


# def test_ragged_struct_set_schema():
#     schema = B.Set[Particle]().schema
#     assert isinstance(schema, Schema)
#     assert schema.is_static is False

#     # pixels is B.Array -> dynamic -> _RaggedArraySetEntry
#     pixels_entry = schema.fields["pixels"]
#     assert isinstance(pixels_entry, _RaggedArraySetEntry)
#     assert pixels_entry.is_static is False

#     # ctf is nested CTF struct -> _SchemaEntry containing set-converted fields
#     ctf_entry = schema.fields["ctf"]
#     assert isinstance(ctf_entry, _SchemaEntry)
#     assert ctf_entry.is_static is True
#     for _, entry in ctf_entry.schema.fields.items():
#         assert isinstance(entry, _ArraySetEntry)


# def test_tiltseries_set_schema_integration():
#     schema = TiltSeries().schema
#     assert isinstance(schema, Schema)
#     assert "tilts" in schema.fields
#     tilts_entry = schema.fields["tilts"]
#     assert isinstance(tilts_entry, _SchemaSetEntry)
#     assert tilts_entry.is_static is False


# def test_set_of_tiltseries():
#     schema = B.Set[TiltSeries]().schema
#     assert isinstance(schema, Schema)
#     assert "tilts" in schema.fields
#     tilts_entry = schema.fields["tilts"]
#     assert isinstance(tilts_entry, _SchemaSetEntry)
#     assert tilts_entry.is_static is False


# def test_set_of_classes2d_schema():
#     class StaticParticle(B.Struct):
#         pixels: B.Array[float] = B.Array(shape=(128, 128))
#         voltage_kv: float

#     class Class2D(B.Struct):
#         average: B.Array[float] = B.Array(shape=(128, 128))
#         particles: B.Set[StaticParticle] = B.Set[StaticParticle]()

#     set_schema = B.Set[Class2D]().schema
#     set_schema.print_tree()

#     assert isinstance(set_schema, Schema)
#     assert set_schema.is_static is False

#     # average is static batch array (128, 128)
#     avg_entry = set_schema.fields["average"]
#     assert isinstance(avg_entry, _ArraySetEntry)
#     assert avg_entry.is_static is True
#     assert avg_entry.shape == (128, 128)

#     # particles is nested ragged collection
#     particles_entry = set_schema.fields["particles"]
#     assert isinstance(particles_entry, _SchemaSetEntry)
#     assert particles_entry.is_static is False

#     # child schema of particles has pixels (ArraySet) and voltage_kv (ArraySet)
#     p_schema = particles_entry.schema
#     assert isinstance(p_schema.fields["pixels"], _ArraySetEntry)
#     assert p_schema.fields["pixels"].shape == (128, 128)
#     assert isinstance(p_schema.fields["voltage_kv"], _ArraySetEntry)
#     assert p_schema.fields["voltage_kv"].shape == (1,)


# def test_set_specialize_struct_dimensions_and_capacity():
#     """Test specializing struct dimensions within an inner Set and specializing outer Set capacity."""
#     class DynamicParticle(B.Struct):
#         H: B.Dim = B.Dim()

#         pixels: B.Array[float] = B.Array(shape=(H, H))
#         voltage_kv: float

#     class Class2D(B.Struct):
#         H: B.Dim = B.Dim()
#         num_particles: B.Arg = B.Arg()

#         average: B.Array[float] = B.Array(shape=(H, H))
#         particles: B.Set[DynamicParticle] = B.Set[DynamicParticle](
#             num_particles,
#             H=H,
#         )

#     # 1. Specialize a single Class2D instance
#     class2d_specialized = Class2D(H=64)
#     assert not class2d_specialized.schema.is_static
#     assert isinstance(class2d_specialized.schema.fields["average"], _ArrayEntryBase)
#     assert class2d_specialized.schema.fields["average"].shape == (64, 64)


#     # Inner particles set has dynamic capacity, but its elements have specialized shape (64, 64)
#     particles_entry = class2d_specialized.schema.fields["particles"]
#     assert isinstance(particles_entry, _SchemaSetEntry)
#     assert particles_entry.capacity is None
#     assert isinstance(particles_entry.schema.fields["pixels"], _ArrayEntryBase)
#     assert particles_entry.schema.fields["pixels"].shape == (64, 64)

#     # 2. Specialize outer Set capacity with a dynamic Dim and size
#     classes_set = B.Set[Class2D](H=128, num_particles=100)
#     assert classes_set.schema.is_static
#     assert not classes_set.convert_to_entry().is_static

#     classes_set.print_schema()


if __name__ == "__main__":
    test_nested_sets()