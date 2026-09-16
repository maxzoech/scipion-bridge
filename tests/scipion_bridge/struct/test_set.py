from typing import Any

import pytest
import numpy as np
import pyarrow as pa

import scipion_bridge as B
from scipion_bridge.core.struct.schema import (
    Entry,
    Schema,
    ArrayEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
    SchemaSetEntry,
    SchemaEntry,
    ArrayEntryBase,
)
from scipion_bridge.core.struct.key_path import KeyPath
from scipion_bridge.core.struct.storage import _BaseStorage

class Data(B.Struct):
    pixels = B.Array[float](shape=(128, 128))
    foo: float


class IndexedData(B.Struct):
    pixels = B.Array[float](shape=(128, 128))
    bar: int


class Frame(B.Struct):
    pixels: B.Array[float] = B.Array(shape=(64, 64))
    frame_id: int


class Movie(B.Struct):
    frames: B.Set[Frame] = B.Set[Frame](capacity=10)
    movie_id: int


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

    # bar field in Set context converts to ArraySetEntry
    bar_entry = set_schema.fields["bar"]
    assert isinstance(bar_entry, ArraySetEntry)
    assert bar_entry.dtype == np.dtype(int)
    assert bar_entry.shape == (1,)
    assert bar_entry.is_static is True

    # metadata field in Set context converts to SchemaEntry wrapping set schema
    metadata_entry = set_schema.fields["metadata"]
    assert isinstance(metadata_entry, SchemaEntry)
    assert metadata_entry.is_static is True
    
    meta_schema = metadata_entry.schema
    assert isinstance(meta_schema, Schema)
    assert meta_schema.is_static is True
    assert set(meta_schema.fields.keys()) == {"val_float", "val_int", "val_complex"}

    val_float = meta_schema.fields["val_float"]
    assert isinstance(val_float, ArraySetEntry)
    assert val_float.dtype == np.dtype(float)
    assert val_float.shape == (1,)
    assert val_float.is_static is True

    val_int = meta_schema.fields["val_int"]
    assert isinstance(val_int, ArraySetEntry)
    assert val_int.dtype == np.dtype(int)
    assert val_int.shape == (1,)
    assert val_int.is_static is True

    val_complex = meta_schema.fields["val_complex"]
    assert isinstance(val_complex, ArraySetEntry)
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
    assert isinstance(bar_entry, ArrayEntry)
    assert bar_entry.dtype == np.dtype(int)
    assert bar_entry.shape == (1,)
    assert bar_entry.is_static is True

    # data field is a nested Set entry with capacity 10
    data_entry = schema.fields["data"]
    assert isinstance(data_entry, SchemaSetEntry)
    assert data_entry.capacity == 10
    assert data_entry.is_static is True

    # Element inner schema fields are converted to ArraySetEntry
    elem_schema = data_entry.schema
    assert isinstance(elem_schema, Schema)
    assert elem_schema.is_static is True
    assert set(elem_schema.fields.keys()) == {"foo"}

    foo_entry = elem_schema.fields["foo"]
    assert isinstance(foo_entry, ArraySetEntry)
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

    # Check that all fields in CTF set schema are ArraySetEntry
    for field_name, entry in schema.fields.items():
        assert isinstance(entry, ArraySetEntry), f"{field_name} should be ArraySetEntry"
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

    # pixels is dynamic B.Array -> RaggedArraySetEntry
    pixels_entry = schema.fields["pixels"]
    assert isinstance(pixels_entry, RaggedArraySetEntry)
    assert pixels_entry.is_static is False

    # ctf is nested CTF struct -> SchemaEntry containing set-converted fields
    ctf_entry = schema.fields["ctf"]
    assert isinstance(ctf_entry, SchemaEntry)
    assert ctf_entry.is_static is True
    for _, entry in ctf_entry.schema.fields.items():
        assert isinstance(entry, ArraySetEntry)


def test_tiltseries_set_schema_integration():
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None, None))

    class TiltSeries(B.Struct):
        tilts: B.Set[Particle]

    schema = TiltSeries.schema()
    assert isinstance(schema, Schema)
    assert "tilts" in schema.fields
    tilts_entry = schema.fields["tilts"]
    assert isinstance(tilts_entry, SchemaSetEntry)
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
    assert isinstance(tilts_entry, SchemaSetEntry)
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
    assert isinstance(avg_entry, ArraySetEntry)
    assert avg_entry.is_static is True
    assert avg_entry.shape == (128, 128)

    # particles is nested collection
    particles_entry = set_schema.fields["particles"]
    assert isinstance(particles_entry, SchemaSetEntry)
    assert particles_entry.is_static is False

    # child schema of particles has pixels (ArraySet) and voltage_kv (ArraySet)
    p_schema = particles_entry.schema
    assert isinstance(p_schema.fields["pixels"], ArraySetEntry)
    assert p_schema.fields["pixels"].shape == (128, 128)
    assert isinstance(p_schema.fields["voltage_kv"], ArraySetEntry)
    assert p_schema.fields["voltage_kv"].shape == (1,)


def test_set_of_specialized_struct():
    class DynamicParticle(B.Struct):
        H = B.Dim(None)
        pixels = B.Array[float](shape=(H, H))

    SpecParticle = DynamicParticle.static(H=64)
    spec_set_schema = B.Set[SpecParticle].schema()  # type: ignore[valid-type]
    assert spec_set_schema.is_static is True
    assert isinstance(spec_set_schema.fields["pixels"], ArraySetEntry)
    assert spec_set_schema.fields["pixels"].shape == (64, 64)


def test_set_capacity_static_vs_dynamic():
    class Item(B.Struct):
        val: float

    dyn_entry = B.Set[Item]().convert_to_entry()
    assert isinstance(dyn_entry, SchemaSetEntry)
    assert dyn_entry.capacity is None
    assert dyn_entry.is_static is False

    fixed_entry = B.Set[Item](capacity=10).convert_to_entry()
    assert isinstance(fixed_entry, SchemaSetEntry)
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
    assert isinstance(items_entry, SchemaSetEntry)
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
    assert isinstance(items_entry_base, SchemaSetEntry)
    assert items_entry_base.capacity is None
    assert not items_entry_base.is_static
    assert not Container.schema().is_static

    # 2. Specializing N on Container via .static() propagates to Set capacity
    Container10 = Container.static(N=10)
    items_entry_10 = Container10.schema().fields["items"]
    assert isinstance(items_entry_10, SchemaSetEntry)
    assert items_entry_10.capacity == 10
    assert items_entry_10.is_static is True
    assert Container10.schema().is_static is True

    # 3. Specializing N on Container via subclass specializations={"N": 20}
    class Container20(Container, specializations={"N": 20}):
        pass

    items_entry_20 = Container20.schema().fields["items"]
    assert isinstance(items_entry_20, SchemaSetEntry)
    assert items_entry_20.capacity == 20
    assert items_entry_20.is_static is True
    assert Container20.schema().is_static is True


# Storage Tests

def test_basic_set_storage(as_engine):
    noise = np.random.uniform(size=[5, 128, 128])
    noise_foo = np.random.uniform(size=[5, 1])

    data = B.Set[Data](capacity=64)

    data["pixels"] = noise
    data["foo"] = noise_foo

    data = as_engine(data)

    assert np.allclose(noise, data["pixels"])
    assert np.allclose(noise_foo, data["foo"])


def test_basic_set_slicing(as_engine):
    data_pixels = np.random.uniform(size=[32, 128, 128])
    data_foo = np.random.uniform(size=[32, 1])

    # Set up buffer
    buffer = B.Set[Data](capacity=32)
    buffer["pixels"] = data_pixels
    buffer["foo"] = data_foo

    buffer = as_engine(buffer)

    assert np.allclose(buffer[1:5]["pixels"], data_pixels[1:5]) # type: ignore
    assert np.allclose(buffer[1:5]["foo"], data_foo[1:5]) # type: ignore

    assert np.allclose(buffer[:5]["pixels"], data_pixels[:5]) # type: ignore
    assert np.allclose(buffer[:5]["foo"], data_foo[:5]) # type: ignore

    assert np.allclose(buffer[5:]["pixels"], data_pixels[5:]) # type: ignore
    assert np.allclose(buffer[5:]["foo"], data_foo[5:]) # type: ignore


def test_set_double_slicing(as_engine):
    data_pixels = np.random.uniform(size=[32, 128, 128])

    # Set up buffer
    buffer = B.Set[Data](capacity=32)
    buffer["pixels"] = data_pixels

    buffer = as_engine(buffer)

    buffer_slice = buffer[16:]
    assert buffer_slice.capacity == 16
    assert np.allclose(buffer_slice["pixels"], data_pixels[16:]) # type: ignore

    # Second slice relative to the first: indices [4:9] -> root indices [20:25] (length 5)
    buffer_subslice = buffer_slice[4:9]
    assert buffer_subslice.capacity == 5
    assert buffer_subslice["pixels"].shape == (5, 128, 128) # type: ignore
    assert np.allclose(buffer_subslice["pixels"], data_pixels[20:25]) # type: ignore


def test_set_index_reading(as_engine):
    data_pixels = np.random.uniform(size=[32, 128, 128])
    buffer = B.Set[IndexedData](capacity=32)
    buffer["pixels"] = data_pixels
    buffer["bar"] = np.arange(32)[..., None]

    buffer = as_engine(buffer)

    sample = buffer[5]
    assert np.allclose(sample.pixels, data_pixels[5])
    assert sample.bar == 5


def test_nested_struct_set_slicing(as_engine):

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

    buffer = as_engine(buffer)

    # Direct integer indexing
    sample = buffer[42]
    assert sample.metadata.bar == 42
    assert np.allclose(sample.metadata.latent, latent_data[42])

    # Sliced subset indexing
    sub_buffer = buffer[10:30]
    assert sub_buffer[5].metadata.bar == 15
    assert np.allclose(sub_buffer[5].metadata.latent, latent_data[15])


def test_multidimensional_nested_set_slicing_and_indexing(as_engine):
    # 20 movies, each with 10 frames -> total (20, 10, ...)
    dataset = B.Set[Movie](capacity=20)
    dataset["movie_id"] = np.arange(20)[..., None]
    
    # Fill frame data
    frames_buffer: B.Set[Frame] = dataset["frames"]
    frames_buffer["frame_id"] = np.tile(np.arange(10)[None, :, None], (20, 1, 1))
    noise_frames = np.random.randn(20, 10, 64, 64)
    frames_buffer["pixels"] = noise_frames

    dataset = as_engine(dataset)

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


def test_nested_set_indexed_write_on_uninitialized_buffer():
    """Verify that writing to an uninitialized nested Set via element indexing allocates

    the full multi-level batch dimensions (20, 10, 64, 64).
    """
    dataset = B.Set[Movie](capacity=20)
    frame_pixels = np.ones((64, 64))

    dataset[0].frames[0].pixels = frame_pixels
    assert np.allclose(dataset[0].frames[0].pixels, frame_pixels)
    staged = dataset._storage._engine._static_staging[("frames", "pixels")]
    assert staged.shape == (20, 10, 64, 64)


def test_nested_set_3d_indexed_write(as_engine):
    class Chunk(B.Struct):
        pixels = B.Array[float](shape=(16, 16))

    class Frame3D(B.Struct):
        chunks: B.Set[Chunk] = B.Set[Chunk](capacity=3)

    class Movie3D(B.Struct):
        frames: B.Set[Frame3D] = B.Set[Frame3D](capacity=4)

    dataset = B.Set[Movie3D](capacity=5)

    data = np.ones((16, 16), dtype=np.float64) * 42.0
    dataset[2].frames[1].chunks[0].pixels = data

    staged = dataset._storage._engine._static_staging[("frames", "chunks", "pixels")]
    assert staged.shape == (5, 4, 3, 16, 16)
    assert np.allclose(dataset[2].frames[1].chunks[0].pixels, data)
    assert np.allclose(dataset[0].frames[0].chunks[0].pixels, np.zeros((16, 16)))

    # Verify Arrow compilation and engine conversion
    compiled = as_engine(dataset)
    assert np.allclose(compiled[2].frames[1].chunks[0].pixels, data)
    assert np.allclose(compiled[0].frames[0].chunks[0].pixels, np.zeros((16, 16)))


def test_struct_containing_set_indexed_write():
    class FrameLocal(B.Struct):
        frame_id: int
        pixels = B.Array[float](shape=(64, 64))

    class MovieLocal(B.Struct):
        movie_id: int
        frames = B.Set[FrameLocal](capacity=10)

    movie = MovieLocal()
    pixels_val = np.ones((64, 64)) * 99.0
    movie.frames[0].pixels = pixels_val

    staged = movie._storage._engine._static_staging[("frames", "pixels")]
    assert staged.shape == (10, 64, 64)
    assert np.allclose(movie.frames[0].pixels, pixels_val)


def test_nested_set_out_of_bounds_validation():
    dataset = B.Set[Movie](capacity=20)
    # Exceed inner Set capacity (10) via Set indexing
    with pytest.raises(IndexError, match="Index 10 out of range for Set with capacity 10"):
        _ = dataset[0].frames[10]

    # Exceed inner Set capacity (10) via direct storage access
    entry = dataset.schema().fields["frames"].children.fields["pixels"]
    with pytest.raises(ValueError, match="exceeds capacity 10 for dimension 1"):
        dataset._storage._engine.write(("frames", "pixels"), entry, np.ones((64, 64)), offset=Offset((0, 10)))

    # Exceed outer Set capacity (20) via direct storage access
    with pytest.raises(ValueError, match="exceeds capacity 20 for dimension 0"):
        dataset._storage._engine.write(("frames", "pixels"), entry, np.ones((64, 64)), offset=Offset((20, 0)))


def test_nested_set_slice_write_on_uninitialized_buffer():
    dataset = B.Set[Movie](capacity=20)
    slice_data = np.random.randn(3, 4, 64, 64)
    sub_frames = dataset[2:5]["frames"][1:5]
    sub_frames["pixels"] = slice_data
    assert dataset._storage._engine._static_staging[("frames", "pixels")].shape == (20, 10, 64, 64)
    assert np.allclose(sub_frames["pixels"], slice_data)


def test_nested_set_unbounded_write_dimension_overflow():
    dataset = B.Set[Movie](capacity=20)
    with pytest.raises(ValueError, match="dimension 1 length 15 exceeds capacity 10"):
        dataset["frames"]["pixels"] = np.ones((20, 15, 64, 64))


def test_multidimensional_2d_slice_both_axes(as_engine):
    # 20 movies, each with 10 frames -> total tensor (20, 10, 64, 64)
    dataset = B.Set[Movie](capacity=20)
    dataset["movie_id"] = np.arange(20)[..., None]

    # Populate 3D grid of frames (20, 10, 64, 64)
    frames_buffer = dataset["frames"]
    noise_frames = np.random.randn(20, 10, 64, 64).astype(np.float32)
    frames_buffer["pixels"] = noise_frames

    dataset = as_engine(dataset)

    # 1. SLICE DIMENSION 0: Take 5 movies (indices 2 to 7)
    five_movies = dataset[2:7]
    assert len(five_movies) == 5

    # 2. SLICE DIMENSION 1: Take 3 frames (indices 1 to 4) from those 5 movies
    sub_frames = five_movies["frames"][1:4]

    # 3. VERIFY SHAPE: 5 movies x 3 frames x (64, 64) pixels
    extracted_pixels = sub_frames["pixels"]
    assert extracted_pixels.shape == (5, 3, 64, 64)

    # 4. VERIFY DATA: Matches exact 2D tensor slice noise_frames[2:7, 1:4]
    assert np.allclose(extracted_pixels, noise_frames[2:7, 1:4])


def test_multidimensional_2d_slice_both_axes_ragged(as_engine):
    class Foo(B.Struct):
        bar: int

    class Frame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None,))
        frame_id: int

        metadata: B.Set[Foo] = B.Set[Foo](capacity=10)

    class Movie(B.Struct):
        frames: B.Set[Frame] = B.Set[Frame](capacity=None)
        movie_id: int

    # 20 movies, each with 10 frames -> 2D grid of ragged arrays
    dataset = B.Set[Movie](capacity=20)
    dataset["movie_id"] = np.arange(20)[..., None]

    # Populate 2D grid of frames (20 movies x 10 frames with variable lengths)
    frames_buffer = dataset[:20]["frames"]
    noise_frames = [
        [np.random.randn(10 + m * 5 + f).astype(np.float32) for f in range(10)]
        for m in range(20)
    ]
    frames_buffer["pixels"] = noise_frames

    dataset = as_engine(dataset)

    # 1. SLICE DIMENSION 0: Take 5 movies (indices 2 to 7)
    five_movies = dataset[2:7]
    assert len(five_movies) == 5

    # 2. SLICE DIMENSION 1: Take 3 frames (indices 1 to 4) from those 5 movies
    sub_frames = five_movies["frames"][1:4]

    # 3. VERIFY SHAPE / STRUCTURE: 5 movies x 3 frames
    extracted_pixels = sub_frames["pixels"]
    assert len(extracted_pixels) == 5

    # 4. VERIFY DATA: Matches exact 2D slice noise_frames[2:7][1:4]
    for i in range(5):
        assert len(extracted_pixels[i]) == 3
        for j in range(3):
            expected = noise_frames[2 + i][1 + j]
            assert np.allclose(extracted_pixels[i][j], expected)


def test_set_storage_shape_mismatch_raises():
    data = B.Set[Data](capacity=64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        data["pixels"] = np.random.uniform(size=[5, 64, 64])


def test_set_storage_capacity_exceeded_raises():
    data = B.Set[Data](capacity=64)
    with pytest.raises(ValueError, match="exceeds capacity 64"):
        data["pixels"] = np.random.uniform(size=[100, 128, 128])


def test_set_storage_incompatible_dtype_raises():
    data = B.Set[Data](capacity=64)
    with pytest.raises(TypeError, match="Cannot cast data of dtype"):
        data["foo"] = np.array([1.0 + 2.0j], dtype=np.complex128)


def test_ragged_set_operations_supported(as_engine):
    class DynamicItem(B.Struct):
        dim = B.Dim()
        pixels = B.Array[float](shape=(dim, dim))

    # Set of dynamic items has is_static == False
    ragged_set = B.Set[DynamicItem](capacity=5)
    assert not ragged_set.schema().is_static

    # 2. Writing ragged column succeeds
    ragged_set["pixels"] = [np.ones((10, 10)) for _ in range(5)]

    ragged_set = as_engine(ragged_set)

    # 3. Slicing ragged set succeeds
    sliced = ragged_set[0:5]
    assert len(sliced) == 5

    # 4. Indexing element from ragged set succeeds
    elem = ragged_set[0]
    assert isinstance(elem, DynamicItem)


def test_dynamic_set_len_and_indexing():
    class Item(B.Struct):
        val: float

    dyn_set = B.Set[Item]()
    assert dyn_set.capacity is None
    assert bool(dyn_set) is True

    # Calling len() on dynamic set raises TypeError
    with pytest.raises(TypeError, match="dynamic capacity has no defined length"):
        len(dyn_set)

    # Indexing into dynamic set without a prior bound/slice raises IndexError
    with pytest.raises(IndexError, match="Cannot index into a Set with dynamic capacity"):
        _ = dyn_set[0]


def test_basic_ragged_set_assign(as_engine):

    class Sample(B.Struct):
        latent = B.Array[np.float32](shape=(None,))

    sample_1 = Sample(latent=np.random.uniform(size=(128)))
    sample_2 = Sample(latent=np.random.uniform(size=(256)))

    samples = B.Set[Sample](capacity=10)

    samples[0] = sample_1
    samples[1] = sample_2

    samples = as_engine(samples)

    # Verify per-element reading
    assert np.allclose(samples[0].latent, sample_1.latent)
    assert np.allclose(samples[1].latent, sample_2.latent)

    # Verify column-wise reading
    latent_view = samples["latent"]
    assert np.allclose(latent_view[0], sample_1.latent)
    assert np.allclose(latent_view[1], sample_2.latent)
    assert len(latent_view[0]) == 128
    assert len(latent_view[1]) == 256


def test_nested_ragged_set_indexed_write():
    class RaggedFrame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None,))
        frame_id: int

    class RaggedMovie(B.Struct):
        frames: B.Set[RaggedFrame] = B.Set[RaggedFrame](capacity=10)
        movie_id: int

    dataset = B.Set[RaggedMovie](capacity=20)

    # Indexed write into nested ragged array leaf
    arr_0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    arr_3 = np.array([4.0, 5.0], dtype=np.float64)
    dataset[0].frames[0].pixels = arr_0
    dataset[0].frames[3].pixels = arr_3

    # Verify populated slots read correctly
    assert np.allclose(dataset[0].frames[0].pixels, arr_0)
    assert np.allclose(dataset[0].frames[3].pixels, arr_3)

    # Verify unassigned slot raises AttributeError
    with pytest.raises(AttributeError, match="has not been initialized"):
        _ = dataset[0].frames[1].pixels

    # Verify unassigned parent movie raises AttributeError
    with pytest.raises(AttributeError, match="has not been initialized"):
        _ = dataset[1].frames[0].pixels


def test_nested_ragged_set_sparse_indexing_and_arrow(as_engine):
    class RaggedFrame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None,))

    class RaggedMovie(B.Struct):
        frames: B.Set[RaggedFrame] = B.Set[RaggedFrame](capacity=10)

    dataset = B.Set[RaggedMovie](capacity=5)

    # Write movie 0 frame 0 and movie 2 frame 3, leaving movie 1 unwritten
    arr_0_0 = np.array([10.0, 20.0], dtype=np.float64)
    arr_2_3 = np.array([30.0, 40.0, 50.0], dtype=np.float64)
    dataset[0].frames[0].pixels = arr_0_0
    dataset[2].frames[3].pixels = arr_2_3

    dataset = as_engine(dataset)

    # Read back populated values through engine
    assert np.allclose(dataset[0].frames[0].pixels, arr_0_0)
    assert np.allclose(dataset[2].frames[3].pixels, arr_2_3)

    # Unwritten movie 1 access
    with pytest.raises((AttributeError, ValueError)):
        _ = dataset[1].frames[0].pixels


def test_nested_ragged_set_3d_indexed_write():
    class RaggedFrame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None,))

    class RaggedMovie(B.Struct):
        frames: B.Set[RaggedFrame] = B.Set[RaggedFrame](capacity=10)

    class Project(B.Struct):
        movies: B.Set[RaggedMovie] = B.Set[RaggedMovie](capacity=5)

    projects = B.Set[Project](capacity=3)

    arr = np.arange(12, dtype=np.float64)
    projects[1].movies[2].frames[0].pixels = arr

    assert np.allclose(projects[1].movies[2].frames[0].pixels, arr)

    with pytest.raises(AttributeError, match="has not been initialized"):
        _ = projects[1].movies[2].frames[1].pixels

    with pytest.raises(AttributeError, match="has not been initialized"):
        _ = projects[0].movies[0].frames[0].pixels


def test_nested_ragged_set_slice_write_length_mismatch_raises():
    class RaggedFrame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None,))

    class RaggedMovie(B.Struct):
        frames: B.Set[RaggedFrame] = B.Set[RaggedFrame](capacity=10)

    dataset = B.Set[RaggedMovie](capacity=5)

    # Slice write with mismatched length raises ValueError
    with pytest.raises(ValueError, match="does not match slice size"):
        dataset[:3]["frames"]["pixels"] = [
            [np.array([1.0])] * 10,
            [np.array([2.0])] * 10,
        ]  # Only 2 elements provided for slice of size 3


def test_nested_dynamic_ragged_set_cross_field_validation():
    class Frame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(None,))
        labels: B.Array[int] = B.Array(shape=(None,))

    class Movie(B.Struct):
        frames: B.Set[Frame] = B.Set[Frame](capacity=None)

    dataset = B.Set[Movie](capacity=2)

    # Stage pixels with 3 frames for movie 0, 2 frames for movie 1
    dataset[:2]["frames"]["pixels"] = [
        [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0])],
        [np.array([5.0]), np.array([6.0])],
    ]

    # Stage conflicting labels counts (2 frames for movie 0 instead of 3)
    dataset[:2]["frames"]["labels"] = [
        [np.array([1]), np.array([2])],  # Conflicting length: 2 != 3
        [np.array([3]), np.array([4])],
    ]

    # Compiling to Arrow should detect cross-field inconsistency and raise ValueError
    with pytest.raises(ValueError, match="Inconsistent child counts under dynamic Set"):
        dataset.to_arrow()


def test_ragged_set_ndarray_column_assignment(as_engine):
    class Particle(B.Struct):
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set = B.Set[Particle](capacity=5)
    latents = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    p_set["embeddings"] = latents

    p_set = as_engine(p_set)
    for i in range(5):
        assert np.allclose(p_set[i].embeddings, latents[i])
    assert np.allclose(p_set["embeddings"][0], latents[0])
    assert len(p_set["embeddings"]) == 5


def test_ragged_set_ndarray_slice_assignment(as_engine):
    class Particle(B.Struct):
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set = B.Set[Particle](capacity=10)
    latents = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    p_set[2:7]["embeddings"] = latents

    p_set = as_engine(p_set)
    for i in range(5):
        assert np.allclose(p_set[2 + i].embeddings, latents[i])


def test_ragged_set_view_assignment():
    class Particle(B.Struct):
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set1 = B.Set[Particle](capacity=5)
    latents = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    p_set1["embeddings"] = latents

    p_set2 = B.Set[Particle](capacity=5)
    p_set2["embeddings"] = p_set1["embeddings"]

    for i in range(5):
        assert np.allclose(p_set2[i].embeddings, latents[i])

    batch = p_set2.to_arrow()
    assert batch.num_rows == 5


def test_sliced_set_to_arrow(as_engine):
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set = B.Set[Particle](capacity=10)
    pixels = np.arange(10 * 16, dtype=np.float32).reshape(10, 4, 4)
    embeddings = np.arange(10 * 8, dtype=np.float64).reshape(10, 8)
    p_set["pixels"] = pixels
    p_set["embeddings"] = embeddings

    p_set = as_engine(p_set)
    sliced = p_set[2:7]
    assert len(sliced) == 5

    batch = sliced.to_arrow()
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 5
    assert batch.schema.names == ["pixels", "embeddings"]

    restored = B.Set[Particle].from_arrow(batch)
    assert len(restored) == 5
    for i in range(5):
        assert np.allclose(restored[i].pixels, pixels[2 + i])
        assert np.allclose(restored[i].embeddings, embeddings[2 + i])


def test_sliced_set_concat(as_engine):
    class Particle(B.Struct):
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set = B.Set[Particle](capacity=10)
    embeddings = np.arange(10 * 8, dtype=np.float64).reshape(10, 8)
    p_set["embeddings"] = embeddings

    p_set = as_engine(p_set)
    chunk1 = p_set[0:4]
    chunk2 = p_set[4:7]
    chunk3 = p_set[7:10]

    combined = B.Set.concat(chunk1, chunk2, chunk3)
    assert len(combined) == 10
    for i in range(10):
        assert np.allclose(combined[i].embeddings, embeddings[i])


def test_empty_set_concat():
    class Particle(B.Struct):
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set1 = B.Set[Particle](capacity=0)
    p_set1["embeddings"] = np.empty((0, 8), dtype=np.float64)

    p_set2 = B.Set[Particle](capacity=0)
    p_set2["embeddings"] = np.empty((0, 8), dtype=np.float64)

    combined = B.Set.concat(p_set1, p_set2)
    assert len(combined) == 0


def test_set_boolean_mask_filtering(as_engine):
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set = B.Set[Particle](capacity=6)
    pixels = np.arange(6 * 16, dtype=np.float32).reshape(6, 4, 4)
    embeddings = np.arange(6 * 8, dtype=np.float64).reshape(6, 8)
    p_set["pixels"] = pixels
    p_set["embeddings"] = embeddings

    p_set = as_engine(p_set)

    # 1. NumPy boolean array
    mask_np = np.array([True, False, True, False, False, True])
    sub_np = p_set[mask_np]
    assert len(sub_np) == 3
    assert np.allclose(sub_np[0].pixels, pixels[0])
    assert np.allclose(sub_np[1].pixels, pixels[2])
    assert np.allclose(sub_np[2].pixels, pixels[5])

    # 2. Python list of bools
    mask_list = [False, True, False, True, False, False]
    sub_list = p_set[mask_list]
    assert len(sub_list) == 2
    assert np.allclose(sub_list[0].embeddings, embeddings[1])
    assert np.allclose(sub_list[1].embeddings, embeddings[3])

    # 3. All-False mask -> empty set
    empty_sub = p_set[np.array([False] * 6)]
    assert len(empty_sub) == 0

    # 4. All-True mask -> full set
    full_sub = p_set[np.array([True] * 6)]
    assert len(full_sub) == 6


def test_set_boolean_mask_errors():
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(2, 2))

    p_set = B.Set[Particle](capacity=3)
    p_set["pixels"] = np.zeros((3, 2, 2), dtype=np.float32)

    # Length mismatch
    with pytest.raises(IndexError, match="Boolean mask length 2 does not match"):
        _ = p_set[np.array([True, False])]

    # Single boolean indexing
    with pytest.raises(TypeError, match="Cannot index Set with a single boolean"):
        _ = p_set[True]


def test_set_integer_indices_take(as_engine):
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        embeddings: B.Array[float] = B.Array(shape=(None,))

    p_set = B.Set[Particle](capacity=5)
    pixels = np.arange(5 * 16, dtype=np.float32).reshape(5, 4, 4)
    embeddings = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    p_set["pixels"] = pixels
    p_set["embeddings"] = embeddings

    p_set = as_engine(p_set)

    # 1. Python list of ints with negative index
    taken_list = p_set[[0, -1, 2]]
    assert len(taken_list) == 3
    assert np.allclose(taken_list[0].pixels, pixels[0])
    assert np.allclose(taken_list[1].pixels, pixels[4])
    assert np.allclose(taken_list[2].pixels, pixels[2])

    # 2. NumPy array of ints
    taken_np = p_set[np.array([1, 3])]
    assert len(taken_np) == 2
    assert np.allclose(taken_np[0].embeddings, embeddings[1])
    assert np.allclose(taken_np[1].embeddings, embeddings[3])

    # 3. Duplicate indices
    taken_dup = p_set[[0, 0, 1]]
    assert len(taken_dup) == 3
    assert np.allclose(taken_dup[0].pixels, pixels[0])
    assert np.allclose(taken_dup[1].pixels, pixels[0])
    assert np.allclose(taken_dup[2].pixels, pixels[1])

    # 4. Empty indices
    assert len(p_set[[]]) == 0
    assert len(p_set[np.array([], dtype=int)]) == 0

    # 5. Out of bounds errors
    with pytest.raises(IndexError, match="out of bounds"):
        _ = p_set[[0, 5]]

    with pytest.raises(IndexError, match="out of bounds"):
        _ = p_set[[-6]]


def test_set_clustering_filtering_workflow(as_engine):
    class Particle(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        embeddings: B.Array[float] = B.Array(shape=(None,))

    num_particles = 12
    num_classes = 3
    p_set = B.Set[Particle](capacity=num_particles)
    pixels = np.arange(num_particles * 16, dtype=np.float32).reshape(num_particles, 4, 4)
    embeddings = np.arange(num_particles * 8, dtype=np.float64).reshape(num_particles, 8)
    p_set["pixels"] = pixels
    p_set["embeddings"] = embeddings

    p_set = as_engine(p_set)

    # Predicted labels for each particle
    predicted_labels = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1, 2]
    labels = np.asarray(predicted_labels)

    # User's target workflow:
    classes = [p_set[labels == k] for k in range(num_classes)]

    assert len(classes) == 3
    assert len(classes[0]) == 4
    assert len(classes[1]) == 4
    assert len(classes[2]) == 4

    # Verify elements in class 0
    cls0_indices = [0, 3, 6, 7]
    for i, orig_idx in enumerate(cls0_indices):
        assert np.allclose(classes[0][i].pixels, pixels[orig_idx])
        assert np.allclose(classes[0][i].embeddings, embeddings[orig_idx])


class MockEngine(_BaseStorage):

    def read(self, key: KeyPath, entry: Entry) -> Any:
        print(f"Read key {key} for entry {entry}")

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        print(f"Write key {key} for entry {entry}")


def test_basic_set_assign():

    class Foo(B.Struct):

        pixels = B.Array[np.float32](shape=(128, 128))


    ds = B.Set[Foo](capacity=10, storage=MockEngine())

    ds[0] = Foo(
        storage=MockEngine(),
        pixels=np.random.uniform(size=[128, 128]),
    )


if __name__ == "__main__":
    test_basic_set_assign()