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
    spec_set_schema = B.Set[SpecParticle].schema()  # type: ignore[valid-type]
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

    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))
        foo: float

    noise = np.random.uniform(size=[5, 128, 128])
    noise_foo = np.random.uniform(size=[5, 1])

    data = B.Set[Data](capacity=64)

    data["pixels"] = noise
    data["foo"] = noise_foo

    assert np.allclose(noise, data["pixels"])
    assert np.allclose(noise_foo, data["foo"])


def test_basic_set_slicing():

    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))
        foo: float

    data_pixels = np.random.uniform(size=[32, 128, 128])
    data_foo = np.random.uniform(size=[32, 1])

    # Set up buffer
    buffer = B.Set[Data](capacity=32)
    buffer["pixels"] = data_pixels
    buffer["foo"] = data_foo

    assert np.allclose(buffer[1:5]["pixels"], data_pixels[1:5])
    assert np.allclose(buffer[1:5]["foo"], data_foo[1:5])

    assert np.allclose(buffer[:5]["pixels"], data_pixels[:5])
    assert np.allclose(buffer[:5]["foo"], data_foo[:5])

    assert np.allclose(buffer[5:]["pixels"], data_pixels[5:])
    assert np.allclose(buffer[5:]["foo"], data_foo[5:])


def test_set_double_slicing():

    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))

    data_pixels = np.random.uniform(size=[32, 128, 128])

    # Set up buffer
    buffer = B.Set[Data](capacity=32)
    buffer["pixels"] = data_pixels

    buffer_slice = buffer[16:]
    assert buffer_slice.capacity == 16
    assert np.allclose(buffer_slice["pixels"], data_pixels[16:])

    # Second slice relative to the first: indices [4:9] -> root indices [20:25] (length 5)
    buffer_subslice = buffer_slice[4:9]
    assert buffer_subslice.capacity == 5
    assert buffer_subslice["pixels"].shape == (5, 128, 128) # type: ignore
    assert np.allclose(buffer_subslice["pixels"], data_pixels[20:25])


def test_set_index_reading():

    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))
        bar: int
    
    data_pixels = np.random.uniform(size=[32, 128, 128])
    buffer = B.Set[Data](capacity=32)
    buffer["pixels"] = data_pixels
    buffer["bar"] = np.arange(32)[..., None]

    sample = buffer[5]
    assert np.allclose(sample.pixels, data_pixels[5])
    assert sample.bar == 5


def test_nested_struct_set_slicing():

    class Metadata(B.Struct):
        latent = B.Array[np.float32](shape=(128,))
        bar: int

    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))
        metadata = Metadata()

    buffer = B.Set[Data](capacity=64)
    metadata_buffer: B.Set[Metadata] = buffer["metadata"]
    metadata_buffer["bar"] = np.arange(64)[..., None]
    latent_data = np.random.randn(64, 128).astype(np.float32)
    metadata_buffer["latent"] = latent_data

    # Direct integer indexing
    sample = buffer[42]
    assert sample.metadata.bar == 42
    assert np.allclose(sample.metadata.latent, latent_data[42])

    # Sliced subset indexing
    sub_buffer = buffer[10:30]
    assert sub_buffer[5].metadata.bar == 15
    assert np.allclose(sub_buffer[5].metadata.latent, latent_data[15])


@pytest.mark.skip("Multidimensional set slicing not implemented yet")
def test_multidimensional_nested_set_slicing_and_indexing():
    class Frame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(64, 64))
        frame_id: int

    class Movie(B.Struct):
        frames: B.Set[Frame] = B.Set[Frame](capacity=10)
        movie_id: int

    # 20 movies, each with 10 frames -> total (20, 10, ...)
    dataset = B.Set[Movie](capacity=20)
    dataset["movie_id"] = np.arange(20)[..., None]
    
    # Fill frame data
    frames_buffer: B.Set[Frame] = dataset["frames"]
    frames_buffer["frame_id"] = np.tile(np.arange(10)[None, :, None], (20, 1, 1))
    noise_frames = np.random.randn(20, 10, 64, 64)
    frames_buffer["pixels"] = noise_frames

    # 1. Index movie, then index frame: dataset[5].frames[3]
    movie_5 = dataset[5]
    assert movie_5.movie_id == 5
    assert movie_5.frames[3].frame_id == 3
    assert np.allclose(movie_5.frames[3].pixels, noise_frames[5, 3])

    # 2. Index movie, then slice frames: dataset[5].frames[2:8]
    sub_frames = movie_5.frames[2:8]
    assert sub_frames.capacity == 6
    assert sub_frames[0].frame_id == 2
    assert sub_frames[3].frame_id == 5
    assert np.allclose(sub_frames[0].pixels, noise_frames[5, 2])
    assert np.allclose(sub_frames[3].pixels, noise_frames[5, 5])

    # 3. Double-slice frames: movie_5.frames[2:8][1:4]
    double_sub_frames = sub_frames[1:4]
    assert double_sub_frames.capacity == 3
    assert double_sub_frames[0].frame_id == 3
    assert double_sub_frames[2].frame_id == 5
    assert np.allclose(double_sub_frames[0].pixels, noise_frames[5, 3])
    assert np.allclose(double_sub_frames[2].pixels, noise_frames[5, 5])

    # 4. Sliced movies, then indexing: dataset[10:15][2].frames[4]
    sliced_movies = dataset[10:15]
    assert sliced_movies[2].movie_id == 12
    assert sliced_movies[2].frames[4].frame_id == 4
    assert np.allclose(sliced_movies[2].frames[4].pixels, noise_frames[12, 4])
    


def test_set_storage_shape_mismatch_raises():
    class Data(B.Struct):
        pixels = B.Array[float](shape=(128, 128))

    data = B.Set[Data](capacity=64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        data["pixels"] = np.random.uniform(size=[5, 64, 64])


@pytest.mark.skip(reason="Triage verification for now")
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


def test_ragged_set_operations_raise_not_implemented():
    class DynamicItem(B.Struct):
        dim = B.Dim()
        pixels = B.Array[float](shape=(dim, dim))

    # Set of dynamic items has is_static == False
    ragged_set = B.Set[DynamicItem](capacity=10)
    assert not ragged_set.schema().is_static

    # 1. Reading ragged column raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Ragged array storage reading"):
        _ = ragged_set["pixels"]

    # 2. Writing ragged column raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Ragged array storage writing"):
        ragged_set["pixels"] = np.ones((5, 10, 10))

    # 3. Slicing ragged set raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Slicing a Set with dynamic/ragged schema"):
        _ = ragged_set[0:5]

    # 4. Indexing element from ragged set raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Indexing elements from a Set with dynamic/ragged schema"):
        _ = ragged_set[0]


def test_basic_ragged_set_assign():

    class Sample(B.Struct):
        latent = B.Array[np.float32](shape=(None,))

    sample_1 = Sample(latent=np.random.uniform(size=(128)))
    sample_2 = Sample(latent=np.random.uniform(size=(256)))

    samples = B.Set[Sample](capacity=10)

    samples[0] = sample_1
    samples[1] = sample_2


if __name__ == "__main__":
    from scipion_bridge.backend.standalone.container import configure_default_env
    
    # Wire the container for 'scipion_bridge'
    container = configure_default_env()
    
    test_basic_ragged_set_assign()