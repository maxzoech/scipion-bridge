from typing import Any, Optional, Tuple
import numpy as np
import pytest
import scipion_bridge as B
from scipion_bridge.core.struct.key_path import KeyPath
from scipion_bridge.core.struct.schema import Entry, Schema, ArrayEntryBase, ArrayEntry
from scipion_bridge.core.struct.storage import _BaseStorage, StorageView


def assert_array_entry(
    entry: Entry,
    expected_shape: Tuple[Optional[int], ...],
    is_static: Optional[bool] = None,
) -> None:
    """Narrows Entry to ArrayEntryBase and validates shape and static status."""
    assert isinstance(entry, ArrayEntryBase)
    assert entry.shape == expected_shape
    if is_static is not None:
        assert entry.is_static == is_static


def get_child_struct(entry: Entry) -> Schema:
    """Narrows Entry and returns its children Schema."""
    assert entry.children is not None
    return entry.children


class MockStorage(_BaseStorage):
    """Stateful mock storage recording read/write operations and holding data in memory."""

    def __init__(
        self,
        root: KeyPath = KeyPath(),
        parent: Optional[_BaseStorage] = None,
    ) -> None:
        super().__init__(root=root, parent=parent)
        self.reads: list[Tuple[KeyPath, Entry]] = []
        self.writes: list[Tuple[KeyPath, Entry, Any]] = []
        self.data: dict[str, Any] = {}

    def read(self, key: KeyPath, entry: Entry) -> Any:
        self.reads.append((key, entry))
        return self.data.get(str(key), None)

    def write(self, key: KeyPath, entry: Entry, data: Any) -> None:
        self.writes.append((key, entry, data))
        self.data[str(key)] = data


# ==============================================================================
# Suite 1: Struct Schema & Metaprogramming Tests
# ==============================================================================


class TestStructSchema:
    def test_simple_struct(self):
        class SimpleStruct(B.Struct):
            val_int: int
            val_float: float
            val_bool: bool

        struct = SimpleStruct()
        struct.print_schema()

        assert struct.schema().is_static
        assert set(struct.schema().fields.keys()) == {"val_int", "val_float", "val_bool"}
        assert_array_entry(struct.schema().fields["val_int"], (1,))

    def test_nested_struct(self):
        class Foo(B.Struct):
            a: np.complex64
            b: float

        class CTF(B.Struct):
            voltage_kv: float
            amplitude_contrast: float
            foo: Foo

        struct = CTF()
        assert struct.schema().is_static
        assert set(struct.schema().fields.keys()) == {"voltage_kv", "amplitude_contrast", "foo"}
        foo_schema = get_child_struct(struct.schema().fields["foo"])
        assert set(foo_schema.fields.keys()) == {"a", "b"}
        struct.schema().print_tree()

    def test_basic_inheritance(self):
        class BaseRecord(B.Struct):
            id: int
            weight: float

        class ExtendedRecord(BaseRecord):
            is_active: bool
            pixels: B.Array[float] = B.Array(shape=(64, 64))

        record = ExtendedRecord()
        assert record.schema().is_static
        assert set(record.schema().fields.keys()) == {"id", "weight", "is_active", "pixels"}
        assert_array_entry(record.schema().fields["id"], (1,))
        assert_array_entry(record.schema().fields["weight"], (1,))
        assert_array_entry(record.schema().fields["is_active"], (1,))
        assert_array_entry(record.schema().fields["pixels"], (64, 64), is_static=True)

    def test_inheritance_dimension_validation_errors(self):
        class FixedBase(B.Struct):
            dim = B.Dim(64)

        # 1. Overriding fixed dimension with None raises ValueError
        with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'dim' \(value=64\) with None"):
            class InvalidChild(FixedBase):
                dim = B.Dim(None)

        # 2. Overriding dimension with invalid default value type raises TypeError
        with pytest.raises(TypeError, match="was assigned an invalid default value of type"):
            class InvalidAnnotatedChild(FixedBase):
                dim: B.Dim = 12.34  # type: ignore

        # 3. Unannotated invalid attribute in subclass raises TypeError
        with pytest.raises(TypeError, match="contains class-level attributes missing type annotations"):
            class UnannotatedChild(FixedBase):
                dim = "invalid_dimension"

    def test_inheritance_with_nested_structs_and_sets(self):
        class Inner(B.Struct):
            feature: float
            vector = B.Array[float](shape=(10,))

        class ContainerBase(B.Struct):
            inner: Inner
            items: B.Set[Inner]

        class ContainerChild(ContainerBase):
            name_id: int
            extra_items: B.Set[Inner]

        child = ContainerChild()
        assert set(child.schema().fields.keys()) == {"inner", "items", "name_id", "extra_items"}

        inner_schema = get_child_struct(child.schema().fields["inner"])
        assert set(inner_schema.fields.keys()) == {"feature", "vector"}
        assert_array_entry(inner_schema.fields["vector"], (10,), is_static=True)

        items_entry = child.schema().fields["items"]
        assert items_entry.children is not None
        assert set(items_entry.children.fields.keys()) == {"feature", "vector"}

    def test_class_vs_instance_get_semantics(self):
        class Record(B.Struct):
            dim = B.Dim(64)

        # Access on class returns the Arg/Dim descriptor instance
        assert isinstance(Record.dim, B.Arg)
        assert Record.dim.value == 64
        assert Record.dim.name == "dim"

        # Access on instance returns the resolved integer value
        inst = Record()
        assert inst.dim == 64

    def test_unassigned_class_vs_instance_get(self):
        class DynamicRecord(B.Struct):
            dim = B.Dim()

        assert isinstance(DynamicRecord.dim, B.Arg)
        assert DynamicRecord.dim.value is None

        inst = DynamicRecord()
        assert inst.dim is None

    def test_instance_set_raises_attribute_error(self):
        class Record(B.Struct):
            dim = B.Dim(64)

        inst = Record()
        with pytest.raises(AttributeError, match="class-level schema parameter"):
            inst.dim = 128

    def test_dim_equality_with_ints_and_dims(self):
        d1 = B.Dim(64)
        d2 = B.Dim(64)
        d3 = B.Dim(128)
        d_none1 = B.Dim()
        d_none2 = B.Dim()

        assert d1 == d2
        assert d1 == 64
        assert 64 == d1
        assert d1 != d3
        assert d1 != 128
        assert d_none1 == d_none2
        assert d_none1 == None
        assert d1 != d_none1

    def test_int_conversion(self):
        d = B.Dim(42)
        assert int(d) == 42

        d_unassigned = B.Dim()
        with pytest.raises(TypeError, match="Cannot convert unassigned Dim to int"):
            int(d_unassigned)

    def test_repr_and_str(self):
        d = B.Dim(64, name="H")
        assert str(d) == "H:64"
        assert "Arg(64, name='H')" in repr(d)

        d_unbound = B.Dim(name="W")
        assert str(d_unbound) == "W"

    def test_sibling_subclasses_do_not_mutate_base_dimension(self):
        """Edge case: subclassing must not mutate the parent class descriptor."""
        class Base(B.Struct):
            H = B.Dim()
            pixels = B.Array[float](shape=(H, H))

        assert Base.H.value is None

        # Create Subclass A with H=64
        class SubA(Base):
            H = B.Dim(64)
            latents = B.Array[float](shape=(H,))

        assert SubA.H.value == 64
        # Base.H must remain unassigned (None), NOT mutated to 64
        assert Base.H.value is None, "BUG: Defining SubA mutated Base.H!"

        SubA.print_schema()

    def test_sibling_subclasses_do_not_poison_each_other(self):
        """Edge case: defining multiple siblings with different values."""
        class Base(B.Struct):
            H = B.Dim()
            pixels = B.Array[float](shape=(H, H))

        class Sub64(Base):
            H = B.Dim(64)

        class Sub128(Base):
            H = B.Dim(128)

        class SubDynamic(Base):
            H = B.Dim()

        assert Sub64.H.value == 64
        assert Sub128.H.value == 128
        assert SubDynamic.H.value is None
        assert Base.H.value is None

    def test_multi_tier_inheritance_isolation(self):
        class Root(B.Struct):
            dim = B.Dim()

        class MidA(Root):
            dim = B.Dim(32)

        class MidB(Root):
            dim = B.Dim(64)

        class LeafA(MidA):
            pass

        class LeafB(MidB):
            pass

        assert Root.dim.value is None
        assert MidA.dim.value == 32
        assert MidB.dim.value == 64
        assert LeafA.dim.value == 32
        assert LeafB.dim.value == 64

    def test_diamond_inheritance_dimension_resolution(self):
        class Root(B.Struct):
            dim = B.Dim()

        class Left(Root):
            dim = B.Dim(16)

        class Right(Root):
            dim = B.Dim(16)

        class Diamond(Left, Right):
            pass

        assert Diamond.dim.value == 16
        assert Root.dim.value is None

    def test_override_fixed_dimension_with_none_rejected(self):
        class FixedBase(B.Struct):
            dim = B.Dim(64)

        with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'dim' \(value=64\) with None"):
            class InvalidSub(FixedBase):
                dim = B.Dim(None)

    def test_override_dimension_with_invalid_type_rejected(self):
        with pytest.raises(TypeError, match="Expected Dim, int, or None"):
            B.Dim("invalid_string")  # type: ignore

        with pytest.raises(TypeError, match="Expected Dim, int, or None"):
            B.Dim([64])  # type: ignore

    def test_validate_method_type_checks(self):
        d = B.Dim(64, name="H")
        with pytest.raises(TypeError, match="Expected Dim, int, or None"):
            d.validate("string_value")

        with pytest.raises(ValueError, match="Cannot override fixed dimension"):
            d.validate(None)

    def test_dimension_name_scoping_no_crosstalk(self):
        class Inner(B.Struct):
            H = B.Dim(64)
            pixels = B.Array[float](shape=(H, H))

        class Outer(B.Struct):
            H = B.Dim()
            inner: Inner
            image = B.Array[float](shape=(H, H))

        assert Inner.H.value == 64
        assert Outer.H.value is None
        assert_array_entry(Inner.schema().fields["pixels"], (64, 64))
        assert_array_entry(Outer.schema().fields["image"], (None, None))


# ==============================================================================
# Suite 2: Array Descriptor Validation & Error Handling Tests
# ==============================================================================


class TestArrayDescriptor:
    def test_array_instantiation_validation(self):
        # 1. Missing shape raises ValueError
        with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
            B.Array()

        with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
            B.Array[float]()

        # 2. Non-sequence shape raises TypeError
        with pytest.raises(TypeError, match="Expected shape to be a tuple or list of dimensions"):
            B.Array[float](shape=128)  # type: ignore

        # 3. Negative dimension raises ValueError
        with pytest.raises(ValueError, match="Array dimension cannot be negative"):
            B.Array[float](shape=(-1, 64))

        # 4. Shape with list input works
        arr = B.Array[float](shape=[64, 64])
        assert len(arr.shape_spec) == 2
        assert arr.shape_spec[0].value == 64
        assert arr.shape_spec[1].value == 64

    def test_array_unassigned_annotation_raises_helpful_error(self):
        with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
            class BadStruct(B.Struct):
                pixels: B.Array[float]

    def test_array_class_access_semantics(self):
        class Particle(B.Struct):
            H = B.Dim(64)
            pixels = B.Array[np.float32](shape=(H, H))

        # Class-level access yields an Array bound to the owner class
        arr = Particle.pixels
        assert isinstance(arr, B.Array)
        assert arr.dtype == np.dtype(np.float32)
        assert arr.shape == (64, 64)
        assert repr(arr) == "Array[float32](shape=(64, 64), owner=Particle)"
        entry = arr.convert_to_entry()
        assert isinstance(entry, ArrayEntryBase)
        assert entry.shape == (64, 64)

        # Calling schema() on uninstantiated Array class raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Cannot get schema directly from an uninstantiated Array class"):
            B.Array.schema()

        # Calling default() raises ValueError
        with pytest.raises(ValueError, match="Missing required argument 'shape' for Array"):
            B.Array.default()

    def test_array_missing_dtype_error(self):
        # Defining a Struct with an Array lacking a dtype specification raises TypeError during class creation
        with pytest.raises(TypeError, match="missing a dtype specification"):
            class UnspecifiedArrayStruct(B.Struct):
                pixels = B.Array(shape=(64, 64))  # type: ignore[var-annotated]

    def test_array_schema_convertible_methods(self):
        arr = B.Array[float](shape=(10, 10))

        with pytest.raises(NotImplementedError, match="Cannot get schema directly from an uninstantiated Array class"):
            B.Array.schema()

        entry = arr.convert_to_entry()
        assert isinstance(entry, ArrayEntryBase)
        assert entry.shape == (10, 10)
        assert repr(arr) == "Array[float64](shape=(10, 10))"

    def test_array_descriptor_non_struct_type_error(self):
        class NonStruct:
            pixels = B.Array[float](shape=(10,))

        non_struct_inst = NonStruct()
        with pytest.raises(TypeError, match="on non-Struct instance"):
            _ = non_struct_inst.pixels  # type: ignore

        with pytest.raises(TypeError, match="on non-Struct instance"):
            non_struct_inst.pixels = np.ones(10)  # type: ignore


# ==============================================================================
# Suite 3: Struct Storage Inference & KeyPath Navigation Tests (via MockStorage)
# ==============================================================================


class _StorageBar(B.Struct):
    data_1: int
    data_2: int


class _StorageFoo(B.Struct):
    pixels = B.Array[np.float32](shape=(128, 128))
    bar: _StorageBar


class TestStructStorageInference:
    """Decomposed storage inference tests and expanded interaction scenarios."""

    Bar = _StorageBar
    Foo = _StorageFoo

    def test_storage_inference_array_write(self):
        """Verify that setting an Array attribute routes to storage.write with correct KeyPath."""
        foo = self.Foo(storage=MockStorage())
        sample_pixels = np.random.uniform(size=(128, 128)).astype(np.float32)

        foo.pixels = sample_pixels

        assert len(foo.storage.writes) == 1 # type: ignore
        written_key, written_entry, written_data = foo.storage.writes[0] # type: ignore

        expected_key = KeyPath().append("pixels")
        assert written_key == expected_key
        assert isinstance(written_entry, ArrayEntry)
        assert written_entry.dtype == np.dtype(np.float32)
        assert written_entry.shape == (128, 128)
        assert np.array_equal(written_data, sample_pixels)
        assert np.array_equal(foo.storage.data[str(expected_key)], sample_pixels) # type: ignore

    def test_storage_inference_array_read(self):
        """Verify that reading an Array attribute calls storage.read with correct KeyPath."""
        foo = self.Foo(storage=MockStorage())
        expected_key = KeyPath().append("pixels")
        seeded_data = np.ones((128, 128), dtype=np.float32)
        foo.storage.data[str(expected_key)] = seeded_data  # type: ignore

        val = foo.pixels

        assert len(foo.storage.reads) == 1 # type: ignore
        read_key, read_entry = foo.storage.reads[0] # type: ignore
        assert read_key == expected_key
        assert isinstance(read_entry, ArrayEntry)
        assert read_entry.dtype == np.dtype(np.float32)
        assert np.array_equal(val, seeded_data)

    def test_storage_inference_substruct_view_scoping(self):
        """Verify that accessing a nested Struct field creates a StorageView with scoped root."""
        foo = self.Foo(storage=MockStorage())

        bar = foo.bar

        assert isinstance(bar, self.Bar)
        assert isinstance(bar.storage, StorageView)
        expected_substruct_path = KeyPath().append("bar")
        assert bar.storage.root == expected_substruct_path
        assert bar.storage.parent is foo.storage
        assert bar.storage.root_storage is foo.storage

    def test_storage_inference_substruct_field_read(self):
        """Verify that reading a field from a substruct view resolves the nested path."""
        foo = self.Foo(storage=MockStorage())
        target_path = KeyPath().append("bar").append("data_1")
        foo.storage.data[str(target_path)] = 42 # type: ignore

        val = foo.bar.data_1

        assert len(foo.storage.reads) == 1 # type: ignore
        read_key, read_entry = foo.storage.reads[0] # type: ignore
        assert read_key == target_path
        assert isinstance(read_entry, ArrayEntry)
        assert read_entry.dtype == np.dtype(np.int64)
        assert read_entry.shape == (1,)
        assert val == 42

    def test_storage_inference_substruct_assignment_copy(self):
        """Verify that assigning a struct into a substruct copies all leaves across storages."""
        target_foo = self.Foo(storage=MockStorage())
        source_bar = self.Bar(storage=MockStorage())

        # Write to source_bar
        source_bar.data_1 = 10
        source_bar.data_2 = 20

        # Perform substruct assignment
        target_foo.bar = source_bar

        # Source storage should have been read for all leaf fields
        source_read_paths = [key for key, _ in source_bar.storage.reads] # type: ignore
        assert KeyPath().append("data_1") in source_read_paths
        assert KeyPath().append("data_2") in source_read_paths

        # Target storage should have written all leaf fields under the 'bar' prefix
        target_write_paths = [key for key, _, _ in target_foo.storage.writes] # type: ignore
        expected_target_d1 = KeyPath().append("bar").append("data_1")
        expected_target_d2 = KeyPath().append("bar").append("data_2")
        assert expected_target_d1 in target_write_paths
        assert expected_target_d2 in target_write_paths

        # Assert data transferred correctly
        assert target_foo.storage.data[str(expected_target_d1)] == 10 # type: ignore
        assert target_foo.storage.data[str(expected_target_d2)] == 20 # type: ignore

    def test_deep_nested_struct_storage_paths(self):
        """Verify 3+ tier nesting path resolution and recursive copying."""
        class Level3(B.Struct):
            val: int

        class Level2(B.Struct):
            child: Level3

        class Level1(B.Struct):
            mid: Level2

        root_struct = Level1(storage=MockStorage())
        deep_path = KeyPath().append("mid").append("child").append("val")
        root_struct.storage.data[str(deep_path)] = 777 # type: ignore

        # Reading through 3 tiers of Structs
        assert root_struct.mid.child.val == 777 # type: ignore
        assert root_struct.storage.reads[-1][0] == deep_path # type: ignore

        # Assigning at intermediate tier Level2
        source_l2 = Level2(storage=MockStorage())
        source_l2.child.val = 888

        root_struct.mid = source_l2
        assert root_struct.storage.data[str(deep_path)] == 888  # type: ignore
        assert root_struct.mid.child.val == 888

    def test_substruct_assignment_from_storage_view(self):
        """Verify assigning foo1.bar = foo2.bar (view-to-view substruct copy)."""
        foo1 = self.Foo(storage=MockStorage())
        foo2 = self.Foo(storage=MockStorage())

        # Write to foo2's substruct
        foo2.bar.data_1 = 123
        foo2.bar.data_2 = 456

        # Assign view to view
        foo1.bar = foo2.bar

        # Validate that foo1 received the data under 'bar' prefix
        assert foo1.bar.data_1 == 123
        assert foo1.bar.data_2 == 456
        assert foo1.storage.data[str(KeyPath().append("bar").append("data_1"))] == 123  # type: ignore
        assert foo1.storage.data[str(KeyPath().append("bar").append("data_2"))] == 456  # type: ignore

    def test_struct_init_kwargs_write_to_storage(self):
        """Verify that passing kwargs during Struct initialization routes to storage.write."""
        sample_pixels = np.ones((128, 128), dtype=np.float32)
        mock_storage = MockStorage()

        foo = self.Foo(storage=mock_storage, pixels=sample_pixels)

        expected_key = KeyPath().append("pixels")
        assert len(mock_storage.writes) == 1
        assert mock_storage.writes[0][0] == expected_key
        assert np.array_equal(mock_storage.writes[0][2], sample_pixels)
        assert np.array_equal(foo.pixels, sample_pixels)

    def test_substruct_assignment_validation_errors(self):
        """Verify that assigning invalid types or mismatched Struct classes raises errors."""
        foo = self.Foo(storage=MockStorage())

        # 1. Assigning non-struct raises TypeError
        with pytest.raises(TypeError, match="Expected Struct value"):
            foo.bar = "invalid_string"  # type: ignore

        with pytest.raises(TypeError, match="Expected Struct value"):
            foo.bar = 12345  # type: ignore

        # 2. Assigning an unrelated Struct raises ValueError (schema mismatch)
        class Unrelated(B.Struct):
            other: float

        with pytest.raises(ValueError):
            foo.bar = Unrelated(storage=MockStorage())  # type: ignore

    def test_struct_instances_storage_isolation(self):
        """Verify that distinct struct instances with separate storages remain isolated."""
        foo1 = self.Foo(storage=MockStorage())
        foo2 = self.Foo(storage=MockStorage())

        foo1.pixels = np.ones((128, 128), dtype=np.float32)
        foo1.bar.data_1 = 1

        assert len(foo2.storage.writes) == 0  # type: ignore
        assert len(foo2.storage.reads) == 0  # type: ignore
        assert len(foo2.storage.data) == 0  # type: ignore


# ==============================================================================
# Suite 4: Concrete Storage Dependent Tests (Skipped during storage engine rewrite)
# ==============================================================================


@pytest.mark.skip(reason="Storage engine rewrite in progress")
class TestStructConcreteStorage:
    def test_basic_struct_storage(self):
        class Data(B.Struct):
            pixels: B.Array[float] = B.Array[float](shape=(128, 128,))
            bar: float

        data = Data()
        noise = np.random.uniform(size=[128, 128])

        data.bar = 42.0
        data.pixels = noise

        assert data.bar == 42.0
        assert np.allclose(data.pixels, noise)

    def test_nested_struct_storage(self):
        class Metadata(B.Struct):
            latent = B.Array[np.float32](shape=(128,))
            bar: float

        class Data(B.Struct):
            pixels = B.Array[float](shape=(128, 128))
            metadata = Metadata()

        data = Data()
        latent_data = np.random.uniform(size=[128,]).astype(np.float32)
        pixel_data = np.random.uniform(size=[128, 128]).astype(np.float64)

        # Assign values across nested and primitive fields
        data.metadata.latent = latent_data
        data.metadata.bar = 3.14
        data.pixels = pixel_data

        # Assertions
        assert np.allclose(data.metadata.latent, latent_data)
        assert np.isclose(data.metadata.bar, 3.14)
        assert np.allclose(data.pixels, pixel_data)

    def test_assign_struct(self):
        class Metadata(B.Struct):
            latent = B.Array[np.float32](shape=(128,))
            bar: float

        class Data(B.Struct):
            pixels = B.Array[float](shape=(128, 128))
            metadata = Metadata()

        latent_data = np.random.uniform(size=[128])

        metadata = Metadata(
            latent=latent_data,
            bar=42.0,
        )

        data = Data(pixels=np.random.uniform(size=(128, 128)))
        data.metadata = metadata

        assert data.metadata.bar == 42.0
        assert np.allclose(data.metadata.latent, metadata.latent)

    def test_multilevel_assign_struct(self):
        class Bar(B.Struct):
            a: int
            b: int

        class Metadata(B.Struct):
            latent = B.Array[np.float32](shape=(128,))
            bar: Bar

        class Data(B.Struct):
            pixels = B.Array[float](shape=(128, 128))
            metadata = Metadata()

        latent_data = np.random.uniform(size=[128])

        metadata = Metadata(
            latent=latent_data,
            bar=Bar(
                a=42,
                b=24,
            ),
        )

        data = Data(pixels=np.random.uniform(size=(128, 128)))
        data.metadata = metadata

        assert data.metadata.bar.a == 42.0
        assert data.metadata.bar.b == 24.0
        assert np.allclose(data.metadata.latent, metadata.latent)

    def test_basic_dynamic_struct(self):
        class Sample(B.Struct):
            latent = B.Array[np.float32](shape=(None,))

        noise_small = np.random.uniform(size=[128,])
        noise_large = np.random.uniform(size=[256,])

        sample_small = Sample(latent=noise_small)
        sample_large = Sample(latent=noise_large)

        assert np.allclose(sample_small.latent, noise_small)
        assert np.allclose(sample_large.latent, noise_large)

    def test_array_descriptor_error_handling(self):
        class Record(B.Struct):
            dim = B.Dim()
            pixels = B.Array[float](shape=(dim, dim))

        inst = Record()

        # 1. Reading uninitialized dynamic array raises AttributeError
        with pytest.raises(AttributeError, match="has not been initialized"):
            _ = inst.pixels

        # 2. Writing valid dynamic array succeeds and is readable
        inst.pixels = np.ones((10, 10))
        assert np.array_equal(inst.pixels, np.ones((10, 10)))

        # 3. Writing array with invalid rank raises ValueError
        with pytest.raises(ValueError, match="Dimension count mismatch|Shape mismatch"):
            inst.pixels = np.ones(10)


# ==============================================================================
# Suite 5: Struct Specialization Tests (Skipped while feature is WIP)
# ==============================================================================


@pytest.mark.skip(reason="Specialization feature WIP")
class TestStructSpecialization:
    def test_multilevel_inheritance(self):
        class Level1(B.Struct):
            val_a: int
            dim1 = B.Dim()

        class Level2(Level1):
            val_b: float
            arr1 = B.Array[float](shape=(Level1.dim1, Level1.dim1))

        class Level3(Level2):
            val_c: bool
            dim1 = B.Dim(32)
            arr2 = B.Array[int](shape=(Level1.dim1, 10))

        level3 = Level3()
        assert level3.schema().is_static
        assert set(level3.schema().fields.keys()) == {"val_a", "val_b", "val_c", "arr1", "arr2"}
        assert_array_entry(level3.schema().fields["arr1"], (32, 32), is_static=True)
        assert_array_entry(level3.schema().fields["arr2"], (32, 10), is_static=True)
        assert_array_entry(level3.schema().fields["val_a"], (1,), is_static=True)

    def test_inheritance_dimension_chaining_and_specialization(self):
        class BaseFoo(B.Struct):
            batch_size = B.Arg()

        class Foo(BaseFoo):
            batch_size = B.Arg(128)
            box_size = B.Arg(None)
            pixels = B.Array[float](shape=(BaseFoo.batch_size, box_size, box_size))

        class FooFlex(Foo):
            embeddings = B.Array[float](shape=(BaseFoo.batch_size, None))

        assert not Foo.schema().is_static
        assert_array_entry(Foo.schema().fields["pixels"], (128, None, None))

        assert not FooFlex.schema().is_static
        assert set(FooFlex.schema().fields.keys()) == {"pixels", "embeddings"}
        assert_array_entry(FooFlex.schema().fields["pixels"], (128, None, None))
        assert_array_entry(FooFlex.schema().fields["embeddings"], (128, None))

        class FooStatic(Foo):
            box_size = B.Arg(64)

        assert FooStatic.schema().is_static
        assert_array_entry(FooStatic.schema().fields["pixels"], (128, 64, 64))

    def test_schema_specialization(self):
        class Foo(B.Struct):
            H = B.Dim(None)
            pixels = B.Array[float](shape=(H, H))

        class Foo128(Foo, specializations={"H": 128}):
            pass

        FooStatic128 = Foo.static(H=128)

        assert Foo.H.value is None
        assert not Foo.schema().is_static
        assert_array_entry(Foo.schema().fields["pixels"], (None, None), is_static=False)

        assert Foo128.H.value == 128
        assert Foo128.schema().is_static
        assert_array_entry(Foo128.schema().fields["pixels"], (128, 128), is_static=True)

        assert FooStatic128.H.value == 128
        assert FooStatic128.schema().is_static
        assert_array_entry(FooStatic128.schema().fields["pixels"], (128, 128), is_static=True)

        inst_128 = Foo128()
        inst_static = FooStatic128()
        assert_array_entry(inst_128.schema().fields["pixels"], (128, 128), is_static=True)
        assert_array_entry(inst_static.schema().fields["pixels"], (128, 128), is_static=True)

    def test_multidim_partial_and_full_specialization(self):
        class TensorData(B.Struct):
            H = B.Dim(None)
            W = B.Dim(None)
            C = B.Dim(None)
            image = B.Array[float](shape=(H, W, C))
            mask = B.Array[bool](shape=(H, W))

        TensorRGB = TensorData.static(C=3)
        assert TensorRGB.C.value == 3
        assert TensorRGB.H.value is None
        assert TensorRGB.W.value is None
        assert not TensorRGB.schema().is_static
        assert_array_entry(TensorRGB.schema().fields["image"], (None, None, 3), is_static=False)
        assert_array_entry(TensorRGB.schema().fields["mask"], (None, None), is_static=False)

        TensorFixed = TensorRGB.static(H=64, W=128)
        assert TensorFixed.H.value == 64
        assert TensorFixed.W.value == 128
        assert TensorFixed.C.value == 3
        assert TensorFixed.schema().is_static
        assert_array_entry(TensorFixed.schema().fields["image"], (64, 128, 3), is_static=True)
        assert_array_entry(TensorFixed.schema().fields["mask"], (64, 128), is_static=True)

    def test_multi_level_inheritance_specialization(self):
        class BaseVolume(B.Struct):
            Z = B.Dim(None)
            Y = B.Dim(None)
            X = B.Dim(None)
            data = B.Array[float](shape=(Z, Y, X))

        class SlabVolume(BaseVolume, specializations={"Z": 1}):
            pass

        assert SlabVolume.Z.value == 1
        assert SlabVolume.Y.value is None
        assert SlabVolume.X.value is None
        assert not SlabVolume.schema().is_static
        assert_array_entry(SlabVolume.schema().fields["data"], (1, None, None), is_static=False)

        class SquareSlab(SlabVolume, specializations={"Y": 256}):
            pass

        assert SquareSlab.Z.value == 1
        assert SquareSlab.Y.value == 256
        assert SquareSlab.X.value is None
        assert not SquareSlab.schema().is_static
        assert_array_entry(SquareSlab.schema().fields["data"], (1, 256, None), is_static=False)

        Cube256 = SquareSlab.static(X=256)
        assert Cube256.Z.value == 1
        assert Cube256.Y.value == 256
        assert Cube256.X.value == 256
        assert Cube256.schema().is_static
        assert_array_entry(Cube256.schema().fields["data"], (1, 256, 256), is_static=True)

    def test_nested_struct_specialization(self):
        class Patch(B.Struct):
            size = B.Dim(None)
            pixels = B.Array[float](shape=(size, size))

        Patch64 = Patch.static(size=64)

        class Container(B.Struct):
            N = B.Dim(None)
            patch = Patch64()
            weights = B.Array[float](shape=(N,))

        assert not Container.schema().is_static
        patch_schema = get_child_struct(Container.schema().fields["patch"])
        assert patch_schema.is_static is True
        assert_array_entry(patch_schema.fields["pixels"], (64, 64), is_static=True)
        assert_array_entry(Container.schema().fields["weights"], (None,), is_static=False)

        ContainerFixed = Container.static(N=10)
        assert ContainerFixed.schema().is_static is True
        fixed_patch_schema = get_child_struct(ContainerFixed.schema().fields["patch"])
        assert fixed_patch_schema.is_static is True
        assert_array_entry(fixed_patch_schema.fields["pixels"], (64, 64), is_static=True)
        assert_array_entry(ContainerFixed.schema().fields["weights"], (10,), is_static=True)

    def test_specialization_error_handling(self):
        class Foo(B.Struct):
            H = B.Dim(None)
            pixels = B.Array[float](shape=(H, H))

        with pytest.raises(TypeError, match="unexpected dimension argument: 'UNKNOWN'"):
            Foo.static(UNKNOWN=64)

        with pytest.raises(TypeError, match="Unknown schema overwrite argument 'UNKNOWN'"):
            class InvalidSub(Foo, specializations={"UNKNOWN": 64}):
                pass

        Foo64 = Foo.static(H=64)
        with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'H' \(value=64\) with None"):
            Foo64.static(H=None)  # type: ignore

        with pytest.raises(ValueError, match=r"Cannot override fixed dimension 'H' \(value=64\) with None"):
            class InvalidSubFixed(Foo64, specializations={"H": None}):  # type: ignore
                pass

        with pytest.raises(TypeError, match="Expected Dim, int, or None"):
            Foo.static(H="invalid")  # type: ignore

    def test_set_of_specialized_struct_schema(self):
        class Particle(B.Struct):
            H = B.Dim(None)
            pixels = B.Array[float](shape=(H, H))

        Particle128 = Particle.static(H=128)
        set_schema = B.Set[Particle128].schema()  # type: ignore[valid-type]
        assert set_schema.is_static is True
        assert "pixels" in set_schema.fields
        assert_array_entry(set_schema.fields["pixels"], (128, 128), is_static=True)

    def test_struct_containing_specialized_set(self):
        class Particle(B.Struct):
            H = B.Dim(None)
            pixels = B.Array[float](shape=(H, H))

        Particle64 = Particle.static(H=64)

        class Micrograph(B.Struct):
            W = B.Dim(None)
            raw = B.Array[float](shape=(W, W))
            particles = B.Set[Particle64](capacity=10)  # type: ignore[valid-type]

        assert not Micrograph.schema().is_static
        MicrographFixed = Micrograph.static(W=1024)
        assert MicrographFixed.schema().is_static is True
        assert_array_entry(MicrographFixed.schema().fields["raw"], (1024, 1024), is_static=True)
        particles_schema = get_child_struct(MicrographFixed.schema().fields["particles"])
        assert_array_entry(particles_schema.fields["pixels"], (64, 64), is_static=True)