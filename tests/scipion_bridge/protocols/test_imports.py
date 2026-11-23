"""Tests for @canonical registration and dynamic import protocol synthesis."""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest
import starfile

import scipion_bridge as B
from scipion_bridge import (
    Collection,
    Field,
    Input,
    Protocol,
    Proxy,
    Set,
    Struct,
    ValidationError,
    canonical,
    get_canonical_types,
    resolver,
)
from scipion_bridge.core.typed.canonical_types import unregister_canonical
from scipion_bridge.core.typed.resolve import current_registry
from scipion_bridge.single_particle import Class2D, Particle
from scipion_bridge.single_particle.mrc import write_stack
from scipion_bridge.utils.protocols import ImportMode, synthesize_import_protocols
from scipion_bridge.utils.protocols import imports as imports_mod

# -- Sample canonical types and proxies for unit tests -------------------------


@canonical
class _SampleScalar(Struct):
    value: float


class _SampleTxtProxy(Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".txt"


class _SampleJsonProxy(Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".json"


@resolver
def _resolve_txt_to_sample_scalar(value: _SampleTxtProxy) -> _SampleScalar:
    num = float(value.path.read_text().strip())
    return _SampleScalar(value=num)


@resolver
def _resolve_json_to_sample_scalar(value: _SampleJsonProxy) -> _SampleScalar:
    num = float(value.path.read_text().strip())
    return _SampleScalar(value=num * 2.0)


@canonical
class _SampleUnreachable(Struct):
    x: int


@canonical
class _SampleCollected(Struct):
    x: float


class _SampleColProxy(Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".col"


@resolver
def _resolve_col_to_collection(
    value: _SampleColProxy,
) -> Collection[_SampleCollected]:
    lines = [
        float(line) for line in value.path.read_text().splitlines() if line.strip()
    ]
    col = Collection[_SampleCollected](capacity=len(lines))
    for i, val in enumerate(lines):
        col[i] = _SampleCollected(x=val)
    return col


# -- @canonical registry tests -------------------------------------------------


def test_canonical_decorator_registers_and_is_idempotent():
    assert Particle in get_canonical_types()
    assert Class2D in get_canonical_types()
    assert _SampleScalar in get_canonical_types()

    before = get_canonical_types()
    returned = canonical(_SampleScalar)
    assert returned is _SampleScalar
    assert get_canonical_types() == before


def test_canonical_rejects_non_class():
    with pytest.raises(TypeError, match="only decorate classes"):
        canonical("not_a_class")  # type: ignore[arg-type]


def test_canonical_reload_updates_in_place():
    cls_v1 = type("_ReloadedType", (Struct,), {"__module__": "fake_reload_mod"})
    cls_v2 = type("_ReloadedType", (Struct,), {"__module__": "fake_reload_mod"})
    try:
        canonical(cls_v1)
        assert cls_v1 in get_canonical_types()
        count_before = len(get_canonical_types())

        canonical(cls_v2)
        registered = get_canonical_types()
        assert len(registered) == count_before
        assert cls_v2 in registered
        assert cls_v1 not in registered
    finally:
        unregister_canonical(cls_v2)


# -- Synthesis, naming, outputs, caching, and pickling -------------------------


def test_synthesize_only_reachable_targets():
    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}

    # Particle has Path -> Set[Particle] only
    assert "ImportSetOfParticle" in synthesized
    assert "ImportParticle" not in synthesized
    assert "ImportCollectionOfParticle" not in synthesized

    # Class2D has no Path resolver yet
    assert "ImportClass2D" not in synthesized
    assert "ImportSetOfClass2D" not in synthesized
    assert "ImportCollectionOfClass2D" not in synthesized

    # _SampleScalar has Path -> _SampleScalar only
    assert "Import_SampleScalar" in synthesized
    assert "ImportSetOf_SampleScalar" not in synthesized
    assert "ImportCollectionOf_SampleScalar" not in synthesized

    # _SampleCollected has Path -> Collection[_SampleCollected] only
    assert "ImportCollectionOf_SampleCollected" in synthesized
    assert "Import_SampleCollected" not in synthesized
    assert "ImportSetOf_SampleCollected" not in synthesized

    # _SampleUnreachable has no Path resolver
    assert "Import_SampleUnreachable" not in synthesized


def test_synthesized_protocol_metadata_outputs_and_caching():
    first_run = {cls.__name__: cls for cls in synthesize_import_protocols()}
    second_run = {cls.__name__: cls for cls in synthesize_import_protocols()}

    proto_cls = first_run["ImportSetOfParticle"]
    assert proto_cls is second_run["ImportSetOfParticle"]
    assert proto_cls.__module__ == "scipion_bridge.utils.protocols.imports"
    assert proto_cls.__qualname__ == "ImportSetOfParticle"
    assert proto_cls.outputs() == {"output": Set[Particle]}
    assert proto_cls.__doc__ is not None
    assert "FLUSH" in proto_cls.__doc__

    # Zero-arg instantiation works
    inst = proto_cls()
    assert isinstance(inst, Protocol)
    assert inst.outputs() == {"output": Set[Particle]}
    assert "path" in inst.configuration.inputs
    assert "mode" in inst.configuration.parameters


def test_collision_disambiguation_is_order_independent():
    cls_a = type("SharedThing", (Struct,), {"__module__": "pkg_a.models"})
    cls_b = type("SharedThing", (Struct,), {"__module__": "pkg_b.models"})

    current_registry().add_resolver(
        Path,
        cls_a,
        lambda value: cls_a(),
        namespace="pkg_a.models",
    )
    current_registry().add_resolver(
        Path,
        cls_b,
        lambda value: cls_b(),
        namespace="pkg_b.models",
    )

    try:
        canonical(cls_a)
        canonical(cls_b)
        names_ab = {
            cls.__name__
            for cls in synthesize_import_protocols()
            if "SharedThing" in cls.__name__
        }
        assert names_ab == {
            "ImportPkgAModels_SharedThing",
            "ImportPkgBModels_SharedThing",
        }

        # Re-register in reverse order -> identical names
        unregister_canonical(cls_a)
        unregister_canonical(cls_b)
        canonical(cls_b)
        canonical(cls_a)
        names_ba = {
            cls.__name__
            for cls in synthesize_import_protocols()
            if "SharedThing" in cls.__name__
        }
        assert names_ba == names_ab
    finally:
        unregister_canonical(cls_a)
        unregister_canonical(cls_b)


def test_synthesized_protocol_pickle_and_pep562_getattr():
    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    proto_cls = synthesized["ImportSetOfParticle"]

    # Remove module attribute to exercise PEP 562 __getattr__ during unpickling
    delattr(imports_mod, "ImportSetOfParticle")
    data = pickle.dumps(proto_cls)
    restored = pickle.loads(data)
    assert restored is proto_cls

    with pytest.raises(AttributeError):
        getattr(imports_mod, "ImportDoesNotExist12345")


# -- Extensions & validation ---------------------------------------------------


def test_field_extensions_public_attribute():
    f_default = Field(default="a")
    assert f_default.extensions is None

    f_ext = Field(optional=False, extensions=(".star", ".xmd"))
    assert f_ext.extensions == (".star", ".xmd")
    assert f_ext != Field(optional=False, extensions=(".star",))

    inp = Input(optional=False, extensions=(".mrc",))
    assert inp.extensions == (".mrc",)


def test_allowed_extensions_derived_and_exposed_on_input():
    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}

    particle_cls = synthesized["ImportSetOfParticle"]
    assert particle_cls.allowed_extensions == (".star", ".xmd")
    path_field = particle_cls().configuration.inputs["path"]
    assert path_field.extensions == (".star", ".xmd")
    assert path_field.help is not None
    assert ".star" in path_field.help
    assert ".xmd" in path_field.help

    scalar_cls = synthesized["Import_SampleScalar"]
    assert scalar_cls.allowed_extensions == (".json", ".txt")
    assert scalar_cls().configuration.inputs["path"].extensions == (".json", ".txt")


def test_validation_error_on_mismatched_extension():
    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    scalar_proto = synthesized["Import_SampleScalar"]()

    scalar_proto.validate_path(Path("/tmp/ok.txt"))
    scalar_proto.validate_path(Path("/tmp/ok.JSON"))
    # Glob patterns are filtered rather than rejected up front
    scalar_proto.validate_path(Path("/tmp/*"))

    with pytest.raises(ValidationError, match=r"\.json.*\.txt"):
        scalar_proto.validate_path(Path("/tmp/bad.csv"))


# -- Deterministic fingerprinting & precomputed resolvers ----------------------


def test_fingerprint_is_deterministic_and_invalidates_on_parameter_change():
    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    proto_cls = synthesized["Import_SampleScalar"]

    p1 = proto_cls()
    with pytest.raises(RuntimeError, match="only defined once its pipeline is built"):
        p1.fingerprint()

    p1.get_pipeline()
    fp1 = p1.fingerprint()

    # Re-synthesize and instantiate again -> identical fingerprint
    synthesized_again = {cls.__name__: cls for cls in synthesize_import_protocols()}
    p2 = synthesized_again["Import_SampleScalar"]()
    p2.get_pipeline()
    fp2 = p2.fingerprint()
    assert fp1 == fp2

    # Different parameter value -> different fingerprint
    p3 = proto_cls(mode=ImportMode.WATCH)
    p3.get_pipeline()
    assert p3.fingerprint() != fp1


def test_setup_precomputes_resolvers_before_launch():
    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    proto = synthesized["Import_SampleScalar"]()
    assert proto._default_resolver is None
    assert proto._extension_resolvers == ()

    proto.get_pipeline()
    assert proto._default_resolver is not None
    ext_map = dict(proto._extension_resolvers)
    assert set(ext_map) == {".json", ".txt"}


# -- In-process ONCE & WATCH execution ----------------------------------------


def test_in_process_once_single_file_and_glob_with_multi_proxy(tmp_path):
    txt_file = tmp_path / "a.txt"
    txt_file.write_text("3.5")
    json_file = tmp_path / "b.json"
    json_file.write_text("4.0")
    ignored_file = tmp_path / "c.csv"
    ignored_file.write_text("999")

    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    proto = synthesized["Import_SampleScalar"]()
    proto.get_pipeline()

    # Single .txt file -> 3.5
    single_out = list(proto._discover_and_load(txt_file))
    assert len(single_out) == 1
    assert float(single_out[0]["output"].value) == pytest.approx(3.5)

    # Single .json file -> routes through _SampleJsonProxy (4.0 * 2.0 = 8.0)
    json_out = list(proto._discover_and_load(json_file))
    assert len(json_out) == 1
    assert float(json_out[0]["output"].value) == pytest.approx(8.0)

    # Disallowed single file raises ValidationError
    with pytest.raises(ValidationError):
        list(proto._discover_and_load(ignored_file))

    # Glob pattern filters out .csv and loads both .txt and .json with their proxies
    glob_out = list(proto._discover_and_load(tmp_path / "*"))
    values = sorted(float(item["output"].value) for item in glob_out)
    assert values == pytest.approx([3.5, 8.0])


def test_in_process_watch_detects_new_file_via_os_watcher(tmp_path):
    initial = tmp_path / "first.txt"
    initial.write_text("1.0")

    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    proto = synthesized["Import_SampleScalar"](mode=ImportMode.WATCH)
    proto.get_pipeline()

    gen = proto._discover_and_load(tmp_path / "*.txt")
    # First item is the existing file
    first_item = next(gen)
    assert float(first_item["output"].value) == pytest.approx(1.0)

    # Drop a new file from a background thread while the OS watcher is listening
    def _create_later() -> None:
        time.sleep(0.2)
        (tmp_path / "ignored.csv").write_text("999")
        (tmp_path / "second.txt").write_text("2.5")

    writer = threading.Thread(target=_create_later, daemon=True)
    writer.start()
    try:
        second_item = next(gen)
        assert float(second_item["output"].value) == pytest.approx(2.5)
    finally:
        gen.close()
        writer.join(timeout=2.0)


# -- End-to-end Ray integration test ------------------------------------------


def _write_sample_particle_stack(directory: Path, stem: str, count: int) -> Path:
    mrcs_path = directory / f"{stem}.mrcs"
    star_path = directory / f"{stem}.star"

    images = np.arange(count * 4 * 4, dtype=np.float32).reshape(count, 4, 4)
    write_stack(mrcs_path, images, overwrite=True)

    df = pd.DataFrame(
        {
            "rlnImageName": [f"{i + 1:06d}@{mrcs_path.name}" for i in range(count)],
            "rlnCoordinateX": np.arange(count, dtype=np.float64),
            "rlnCoordinateY": np.arange(count, dtype=np.float64) * 2.0,
        },
    )
    starfile.write(df, star_path, overwrite=True)
    return star_path


class _CountParticles(Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()

    @classmethod
    def outputs(cls):
        return {"count": int}

    def steps(self):
        return self.particles.map(lambda batch: {"count": len(batch)})


@pytest.mark.usefixtures("ray_cluster")
def test_ray_import_set_of_particle_standalone_and_chained(tmp_path):
    import ray
    from scipion_bridge.backend.ray import RayPipelineRunner

    @ray.remote
    class _Collector:
        def __init__(self) -> None:
            self.items: List[Any] = []

        def add(self, value: Any) -> None:
            self.items.append(value)

        def get(self) -> List[Any]:
            return list(self.items)

    star_a = _write_sample_particle_stack(tmp_path, "stack_a", 3)
    _write_sample_particle_stack(tmp_path, "stack_b", 2)

    synthesized = {cls.__name__: cls for cls in synthesize_import_protocols()}
    ImportSetOfParticle = synthesized["ImportSetOfParticle"]

    # Standalone: glob matching both .star files (ignoring .mrcs files)
    collector = _Collector.remote()
    with RayPipelineRunner(
        ImportSetOfParticle(),
        sink=lambda item: ray.get(collector.add.remote(len(item["output"]))),
    ) as runner:
        runner.run(inputs={"path": tmp_path / "*"})

    assert sum(ray.get(collector.get.remote())) == 5

    # Chained: ImportSetOfParticle() | _CountParticles()
    chain_collector = _Collector.remote()
    chain = ImportSetOfParticle() | _CountParticles()
    with RayPipelineRunner(
        chain,
        sink=lambda out: ray.get(chain_collector.add.remote(out["count"])),
    ) as runner:
        runner.run(inputs={"path": star_a})

    assert sum(ray.get(chain_collector.get.remote())) == 3
