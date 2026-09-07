import numpy as np
import pytest
import pyarrow as pa

import scipion_bridge as B
from scipion_bridge.core.struct import (
    RaggedArrayView,
    ArrayEntry,
    ArraySetEntry,
    RaggedArraySetEntry,
    SchemaEntry,
    SchemaSetEntry,
    ArrayStorage,
    Schema,
)
from scipion_bridge.core.struct.utils.dask_serialization import serialize_set, deserialize_set


class Metadata(B.Struct):
    tag: int
    latent = B.Array[np.float32](shape=(None,))


class Particle(B.Struct):
    pixels = B.Array[np.float32](shape=(32, 32))
    metadata: Metadata


def test_arrow_sequence_constructor():
    p1 = Particle(
        pixels=np.ones((32, 32), dtype=np.float32),
        metadata=Metadata(tag=1, latent=np.array([1.0, 2.0], dtype=np.float32)),
    )
    p2 = Particle(
        pixels=np.zeros((32, 32), dtype=np.float32),
        metadata=Metadata(tag=2, latent=np.array([3.0, 4.0, 5.0], dtype=np.float32)),
    )

    particles = B.Set[Particle]([p1, p2])
    assert len(particles) == 2
    assert particles.capacity == 2

    # Verify elements
    assert np.array_equal(particles[0].pixels, p1.pixels)
    assert particles[0].metadata.tag == 1
    assert np.array_equal(particles[0].metadata.latent, p1.metadata.latent)

    assert np.array_equal(particles[1].pixels, p2.pixels)
    assert particles[1].metadata.tag == 2
    assert np.array_equal(particles[1].metadata.latent, p2.metadata.latent)


def test_arrow_record_batch_conversion_and_freeze():
    p1 = Particle(
        pixels=np.ones((32, 32), dtype=np.float32),
        metadata=Metadata(tag=10, latent=np.ones(64, dtype=np.float32)),
    )
    p2 = Particle(
        pixels=np.zeros((32, 32), dtype=np.float32),
        metadata=Metadata(tag=20, latent=np.ones(128, dtype=np.float32)),
    )

    particles = B.Set[Particle]([p1, p2])
    batch = particles.to_arrow()

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 2
    assert "pixels" in batch.schema.names
    assert "metadata" in batch.schema.names

    # Schema inspection
    pixels_type = batch.schema.field("pixels").type
    assert isinstance(pixels_type, pa.FixedShapeTensorType)
    assert pixels_type.shape == [32, 32]

    meta_type = batch.schema.field("metadata").type
    assert isinstance(meta_type, pa.StructType)

    # Immutability verification
    with pytest.raises(RuntimeError, match="Cannot mutate a frozen Set"):
        particles[0] = p2


def test_arrow_ragged_array_view():
    class Sample(B.Struct):
        latent = B.Array[np.float32](shape=(None,))

    samples = B.Set[Sample](capacity=3)
    samples[0] = Sample(latent=np.array([1.0, 2.0], dtype=np.float32))
    samples[1] = Sample(latent=np.array([3.0, 4.0, 5.0], dtype=np.float32))
    samples[2] = Sample(latent=np.array([6.0], dtype=np.float32))

    latent_view = samples["latent"]
    assert isinstance(latent_view, RaggedArrayView)
    assert len(latent_view) == 3

    assert np.array_equal(latent_view[0], np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(latent_view[1], np.array([3.0, 4.0, 5.0], dtype=np.float32))
    assert np.array_equal(latent_view[2], np.array([6.0], dtype=np.float32))

    # Test list conversion and repr
    as_list = latent_view.to_list()
    assert len(as_list) == 3
    assert all(isinstance(arr, np.ndarray) for arr in as_list)
    assert "RaggedArrayView" in repr(latent_view)
    assert "shapes=" in repr(latent_view)


def test_arrow_2d_slicing():
    class SimpleData(B.Struct):
        pixels = B.Array[np.float32](shape=(16, 16))
        value: int

    data_set = B.Set[SimpleData](capacity=10)
    for i in range(10):
        data_set[i] = SimpleData(
            pixels=np.full((16, 16), i, dtype=np.float32),
            value=i * 10,
        )

    # 2D indexing data_set[2:5, "pixels"] is rejected
    with pytest.raises(TypeError, match="Invalid Set index type"):
        _ = data_set[2:5, "pixels"]

    # Chained slicing data_set[2:5]["pixels"] is supported
    sliced_pixels = data_set[2:5]["pixels"]
    assert sliced_pixels.shape == (3, 16, 16)
    assert np.all(sliced_pixels[0] == 2)
    assert np.all(sliced_pixels[1] == 3)
    assert np.all(sliced_pixels[2] == 4)


def test_arrow_set_concatenation():
    p1 = Particle(
        pixels=np.ones((32, 32), dtype=np.float32),
        metadata=Metadata(tag=1, latent=np.ones(10, dtype=np.float32)),
    )
    p2 = Particle(
        pixels=np.full((32, 32), 2.0, dtype=np.float32),
        metadata=Metadata(tag=2, latent=np.ones(20, dtype=np.float32)),
    )
    p3 = Particle(
        pixels=np.full((32, 32), 3.0, dtype=np.float32),
        metadata=Metadata(tag=3, latent=np.ones(30, dtype=np.float32)),
    )

    set1 = B.Set[Particle]([p1])
    set2 = B.Set[Particle]([p2, p3])

    combined = B.Set.concat(set1, set2)
    assert len(combined) == 3
    assert combined.capacity == 3

    assert np.all(combined[0].pixels == 1.0)
    assert np.all(combined[1].pixels == 2.0)
    assert np.all(combined[2].pixels == 3.0)

    assert combined[0].metadata.tag == 1
    assert combined[1].metadata.tag == 2
    assert combined[2].metadata.tag == 3

    assert len(combined[0].metadata.latent) == 10
    assert len(combined[1].metadata.latent) == 20
    assert len(combined[2].metadata.latent) == 30


def test_dask_ipc_roundtrip():
    p1 = Particle(
        pixels=np.full((32, 32), 7.0, dtype=np.float32),
        metadata=Metadata(tag=77, latent=np.full(15, 7.7, dtype=np.float32)),
    )
    particles = B.Set[Particle]([p1])

    header, frames = serialize_set(particles)
    recovered = deserialize_set(header, frames)

    assert len(recovered) == 1
    assert np.all(recovered[0].pixels == 7.0)
    assert recovered[0].metadata.tag == 77
    assert len(recovered[0].metadata.latent) == 15
    assert np.allclose(recovered[0].metadata.latent, 7.7)


def test_unified_storage_read_write_and_schema_dataclasses():
    # Verify dataclass behaviour
    entry = ArrayEntry(dtype=np.dtype("float32"), shape=(10, 10))
    assert entry.is_static
    assert repr(entry).startswith("ArrayEntry(")

    set_entry = entry.to_set_entry(capacity=5)
    assert isinstance(set_entry, ArraySetEntry)
    assert set_entry.capacity == 5

    ragged = RaggedArraySetEntry(dtype=np.dtype("int32"), shape=(None,), capacity=4)
    assert not ragged.is_static

    # Verify unified read/write on ArrayStorage
    schema = Schema(dtype=None, fields={"data": set_entry, "ragged": ragged})
    storage = ArrayStorage(schema=schema, capacity=5)

    # Test static write and read
    arr_data = np.ones((5, 10, 10), dtype=np.float32)
    storage.write("data", set_entry, arr_data)
    read_data = storage.read("data", set_entry)
    assert np.array_equal(read_data, arr_data)

    # Test ragged write and read
    ragged_items = [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    storage.write("ragged", ragged, ragged_items)
    ragged_view = storage.read("ragged", ragged)
    assert isinstance(ragged_view, RaggedArrayView)
    assert np.array_equal(ragged_view[0], np.array([1, 2], dtype=np.int32))
    assert np.array_equal(ragged_view[1], np.array([3], dtype=np.int32))


def test_polymorphic_storage_engine_lifecycle_and_views():
    from scipion_bridge.core.struct.storage import (
        _StagingEngine,
        _ArrowEngine,
        ArrayStorageView,
    )

    class Item(B.Struct):
        pixels = B.Array[float](shape=(16, 16))
        tag: int

    set_schema = B.Set[Item].schema()

    # 1. Staging mode initially
    storage = ArrayStorage(schema=set_schema, capacity=4)
    assert not storage.is_frozen
    assert isinstance(storage._engine, _StagingEngine)
    assert storage.capacity == 4

    # Write initial data
    noise = np.random.randn(4, 16, 16).astype(np.float32)
    storage.write("pixels", set_schema.fields["pixels"], noise)

    # 2. Create an ArrayStorageView BEFORE freezing
    view = ArrayStorageView(schema=set_schema, parent=storage, path=(), offset=(1,))
    view_pixels_before = view.read("pixels", set_schema.fields["pixels"])
    assert np.allclose(view_pixels_before, noise[1])

    # 3. Freeze storage into an Arrow RecordBatch
    batch = storage.to_record_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 4
    assert storage.is_frozen
    assert isinstance(storage._engine, _ArrowEngine)

    # 4. Verify existing ArrayStorageView remains fully valid and readable after freeze
    view_pixels_after = view.read("pixels", set_schema.fields["pixels"])
    assert np.allclose(view_pixels_after, noise[1])

    # 5. Verify mutation is strictly rejected on frozen engine
    with pytest.raises(RuntimeError, match="Cannot mutate a frozen Set"):
        storage.write("pixels", set_schema.fields["pixels"], noise)

    with pytest.raises(RuntimeError, match="Cannot mutate a frozen Set"):
        view.write("pixels", set_schema.fields["pixels"], noise[0])

    # 6. Direct ingestion via from_record_batch starts immediately as _ArrowEngine
    direct_storage = ArrayStorage.from_record_batch(batch, schema=set_schema)
    assert direct_storage.is_frozen
    assert isinstance(direct_storage._engine, _ArrowEngine)
    assert direct_storage.capacity == 4


def test_keypath_schema_tree_iter_and_view_qualification():
    """Verify KeyPath tuples across schema tree iteration and storage view qualification."""
    from scipion_bridge.core.struct.schema import KeyPath
    from scipion_bridge.core.struct.storage import ArrayStorageView

    class Metadata(B.Struct):
        tag: int
        rate: float

    class Frame(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(32, 32))
        metadata: Metadata

    frame_schema = Frame.schema()

    # 1. Verify Schema.tree_iter yields (KeyPath, ArrayEntryBase)
    leaves = dict(frame_schema.tree_iter())
    assert ("pixels",) in leaves
    assert ("metadata", "tag") in leaves
    assert ("metadata", "rate") in leaves
    for path, entry in leaves.items():
        assert isinstance(path, tuple)
        assert all(isinstance(seg, str) for seg in path)

    # 2. Verify ArrayStorageView qualification with KeyPath tuples
    root_storage = ArrayStorage(schema=frame_schema)
    meta_view = ArrayStorageView(schema=Metadata.schema(), parent=root_storage, path=("metadata",))

    assert meta_view.qualify_path(("tag",)) == ("metadata", "tag")
    assert meta_view.qualify_path("tag") == ("metadata", "tag")

    # 3. Verify literal dots are preserved without string splitting
    assert meta_view.qualify_path(("channel.1",)) == ("metadata", "channel.1")
    assert meta_view.qualify_path("channel.1") == ("metadata", "channel.1")


def test_struct_storage_keypath_direct_read_write():
    """Verify Struct reading and writing using atomic KeyPath tuples and single strings."""
    class Sample(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(4, 4))
        label: int

    schema = Sample.schema()
    storage = ArrayStorage(schema=schema)

    # Write using KeyPath tuple
    data = np.ones((4, 4), dtype=np.float32)
    storage.write(("pixels",), schema.fields["pixels"], data)

    # Read using KeyPath tuple and string
    assert np.allclose(storage.read(("pixels",), schema.fields["pixels"]), data)
    assert np.allclose(storage.read("pixels", schema.fields["pixels"]), data)

    # Membership check
    assert ("pixels",) in storage
    assert "pixels" in storage
    assert ("nonexistent",) not in storage


def test_set_indexing_three_primitives():
    """Verify strictly the 3 primitives: set[int], set[slice], set[str] and rejection of tuples."""
    class Sub(B.Struct):
        val: int

    class Container(B.Struct):
        pixels: B.Array[float] = B.Array(shape=(8, 8))
        sub: Sub

    s = B.Set[Container](capacity=5)

    # 1. Single string field access (column write)
    noise = np.random.randn(5, 8, 8).astype(np.float32)
    s["pixels"] = noise
    assert ("pixels",) in s._storage

    # 2. Row index read (int)
    item = s[0]
    assert isinstance(item, Container)

    # 3. Row slice read (slice)
    sub_set = s[1:3]
    assert isinstance(sub_set, B.Set)
    assert len(sub_set) == 2

    # 4. Multi-component tuple indexing raises TypeError
    with pytest.raises(TypeError, match="Invalid Set index type"):
        _ = s["sub", "val"]

    with pytest.raises(TypeError, match="Invalid Set key type"):
        s["sub", "val"] = np.ones((5, 1))


def test_arrow_missing_values_bitmask():
    """Verify that uninitialized and partially written fields generate Arrow validity bitmasks."""
    class Sample(B.Struct):
        pixels = B.Array[float](shape=(4, 4))
        tag: int
        ragged = B.Array[float](shape=(None,))

    s = B.Set[Sample](capacity=4)
    # Partially write pixels: only 2 rows populated
    s["pixels"] = np.ones((2, 4, 4), dtype=np.float64)
    # Do not write tag at all (completely uninitialized)
    # Ragged with explicit None in middle
    s["ragged"] = [np.array([1.0, 2.0]), None, np.array([3.0])]

    batch = s.to_arrow()
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 4

    # 1. Completely uninitialized field (tag): all 4 rows are null
    tag_col = batch.column("tag")
    assert tag_col.null_count == 4
    for i in range(4):
        assert not tag_col[i].is_valid

    # 2. Partially populated field (pixels): first 2 valid, last 2 null
    pixels_col = batch.column("pixels")
    assert pixels_col.null_count == 2
    assert pixels_col[0].is_valid
    assert pixels_col[1].is_valid
    assert not pixels_col[2].is_valid
    assert not pixels_col[3].is_valid

    # 3. Ragged field: index 1 is None, index 3 is padded None -> 2 nulls
    ragged_col = batch.column("ragged")
    assert ragged_col.null_count == 2
    assert ragged_col[0].is_valid
    assert not ragged_col[1].is_valid
    assert ragged_col[2].is_valid
    assert not ragged_col[3].is_valid

