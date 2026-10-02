import logging
import os
import warnings
from pathlib import Path
import tempfile

from scipion_bridge.core.typed.resolve import (
    current_registry,
    Registry,
    resolver,
)

import scipion_bridge as sb
from scipion_bridge.backend.standalone.container import Container
from scipion_bridge.core.utils.arc import manager as arc_manager

import pytest
from typing import Optional, Tuple

temp_base_dir = tempfile.gettempdir()


class TempFileMock:

    def __init__(self):
        self.count = 0

    def new_temporary_file(
        self, suffix: Optional[str] = None, prefix: Optional[str] = None
    ) -> os.PathLike:
        file = f"{temp_base_dir}/{prefix or ''}temp_file_{self.count}{suffix or ''}"
        self.count += 1

        return Path(file)

    def delete(self, path: os.PathLike):
        pass


class Volume(sb.Proxy):

    @classmethod
    def file_ext(cls):
        return ".vol"


class TextFile(sb.Proxy):

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".txt"


@pytest.mark.filterwarnings(
    "ignore:Counting references for non-temporary file.*is deprecated"
)
def test_conversion_to_typed_proxy():

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):

        untyped = sb.Proxy(Path("/path/to/proxy"), managed=True)
        assert arc_manager.get_count(Path("/path/to/proxy")) == 1

        typed = untyped.typed(astype=TextFile, copy_data=False)
        assert str(typed.path) == "/path/to/proxy.txt"

        assert arc_manager.get_count(Path("/path/to/proxy")) == 1
        assert arc_manager.get_count(Path("/path/to/proxy.txt")) == 1

        del typed, untyped

    proxy_obj = sb.Proxy(Path(f"{temp_base_dir}/test_file"), managed=True)
    with open(proxy_obj.path, mode="w") as f:
        f.write("Hello World")

    proxy_obj = proxy_obj.typed(astype=TextFile)
    assert arc_manager.is_tracked(Path(f"{temp_base_dir}/test_file")) == False
    assert arc_manager.is_tracked(Path(f"{temp_base_dir}/test_file.txt")) == True

    with open(proxy_obj.path, mode="r") as f:
        assert f.read() == "Hello World"


def test_resolve_proxy_output():

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):
        p = current_registry().resolve(sb.Output(Volume), astype=sb.Proxy)
        assert str(p.path) == f"{temp_base_dir}/temp_file_0.vol"

        del p


def test_resolve_proxy():
    import os
    from pathlib import Path

    def _resolve_output_to_proxy(output: sb.Output):
        ext = str(output.dtype.file_ext())
        return output.dtype(Path("/path/to/output" + ext), managed=False)

    registry = Registry()
    registry.add_resolver(Path, str, lambda x: str(x))
    registry.add_resolver(sb.Proxy, Path, lambda x: x.path)
    registry.add_resolver(sb.Output, sb.Proxy, _resolve_output_to_proxy)

    resolved_path = registry.resolve(sb.Proxy(Path("/path/to/file.txt")), str)
    assert resolved_path == "/path/to/file.txt"

    resolved_path = registry.resolve(sb.Proxy("/path/to/file.txt"), str)  # type: ignore
    assert resolved_path == "/path/to/file.txt"

    resolved_proxy = registry.resolve(sb.Output(TextFile), sb.Proxy)
    assert str(resolved_proxy.path) == "/path/to/output.txt"

    resolved_path = registry.resolve(sb.Output(TextFile), str)
    assert resolved_path == "/path/to/output.txt"


def test_resolve_proxified():

    @sb.proxify
    def foo(
        inputs: sb.ResolveProxy[TextFile],
        outputs: sb.ResolveProxy = sb.Output(TextFile),
    ) -> Optional[sb.Proxy]:
        assert inputs == "/path/to/input.txt"
        assert outputs == "/path/to/output.txt"

        return None

    input_proxy = TextFile(Path("/path/to/input.txt"))
    output_proxy = TextFile(Path("/path/to/output.txt"))

    out = foo(input_proxy, output_proxy)
    assert out is not None
    assert str(out.path) == "/path/to/output.txt"

    out = foo(Path("/path/to/input.txt"), Path("/path/to/output.txt"))
    assert isinstance(out, sb.Proxy)
    assert out.path == Path("/path/to/output.txt")
    assert out.managed == False


def test_resolve_proxy_multi_output():

    @sb.proxify
    def foo(
        output_1=sb.Output(Volume),
        output_2=sb.Output(Volume),
    ):
        pass

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        output: Tuple[sb.Proxy, sb.Proxy] = foo()  # type: ignore

        assert str(output[0].path) == f"{temp_base_dir}/temp_file_0.vol"
        assert str(output[1].path) == f"{temp_base_dir}/temp_file_1.vol"


def test_nested_proxies():

    @sb.proxify
    def func_1(output_path=sb.Output(TextFile)):
        assert isinstance(output_path, str)

        with open(output_path, "w+") as f:

            f.write("Write from func 1")

    @sb.proxify
    def func_2(output_path=sb.Output(TextFile)):
        return func_1(output_path)

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        output = func_2(sb.Output(TextFile))
        assert isinstance(output, sb.Proxy)
        assert str(output.path) == f"{temp_base_dir}/temp_file_0.txt"
        assert output.managed == True

        with open(output.path) as f:
            assert f.read() == "Write from func 1"


def test_nested_proxy_groups():

    @sb.proxify
    def func_1(output_path=sb.Output(sb.ParticleStackProxy)):
        assert isinstance(output_path, str)
        meta_path = Path(output_path)
        stack_path = Path(output_path).with_suffix(".mrcs")
        with open(meta_path, "w+") as f:
            f.write("Star metadata from func 1")
        with open(stack_path, "w+") as f:
            f.write("MRC stack data from func 1")

    @sb.proxify
    def func_2(output_path=sb.Output(sb.ParticleStackProxy)):
        return func_1(output_path)

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        output = func_2()

        assert str(output.metadata.path) == f"{temp_base_dir}/temp_file_0.star"
        assert str(output.particle_stack.path) == f"{temp_base_dir}/temp_file_0.mrcs"

        assert output.managed == True
        assert output.metadata.managed == True
        assert output.particle_stack.managed == True

        with open(output.metadata.path) as f:
            assert f.read() == "Star metadata from func 1"
        with open(output.particle_stack.path) as f:
            assert f.read() == "MRC stack data from func 1"


def test_return_value_warning():

    @sb.proxify
    def foo(output: sb.Resolve[sb.Proxy, sb.Output] = sb.Output(TextFile)):
        return 42

    @sb.proxify
    def func_1(output_path: sb.Resolve[sb.Proxy, sb.Output]):
        pass

    @sb.proxify
    def func_2():
        return func_1(sb.Output(TextFile))

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        with pytest.warns(UserWarning):
            foo(sb.Output(TextFile))

        with warnings.catch_warnings(record=True) as w:
            func_2()

            assert len(w) == 0


def test_proxify_with_params():

    # logging.basicConfig(level=logging.DEBUG)

    @sb.proxify
    def foo(
        inputs: sb.ResolveProxy[TextFile],
        outputs: sb.ResolveProxy[sb.Output] = sb.Output(Volume),
        bar: Optional[Tuple] = None,
        *,
        value=None,
    ):

        assert inputs == "/path/to/inputs.txt"
        assert outputs == f"{temp_base_dir}/temp_file_0.vol"
        assert bar == "1 2 3"
        assert value == "42"

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.typed.core_resolvers",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):
        out = foo(Path("/path/to/inputs.txt"), bar=(1, 2, 3), value=42)

        assert out is not None
        assert str(out.path) == f"{temp_base_dir}/temp_file_0.vol"

        del out


def test_resolve_proxify_with_type_error():

    @sb.proxify
    def foo(inputs: sb.ResolveProxy[TextFile]):
        assert inputs == "/path/to/text_file.txt"

    with pytest.raises(TypeError):
        foo(Volume(Path("/path/to/volume.vol")))  # Fails because wrong type
        foo(Path("/path/to/volume.vol"))  # Fails because of wrong extension

    foo(Path("/path/to/text_file.txt"))  # Correctly resolves


def test_combine_proxify_and_resolve():
    import numpy as np

    class MyVolume(sb.Proxy):

        @classmethod
        def file_ext(cls):
            return ".custom"

    class OtherVolume(sb.Proxy):

        @classmethod
        def file_ext(cls):
            return ".something"

    @resolver
    def resolve_numpy_to_my_volume2(value: np.ndarray) -> OtherVolume:
        return OtherVolume(Path("/path/to/volume.something"), managed=True)

    @resolver
    def resolve_numpy_to_my_volume(value: np.ndarray) -> MyVolume:
        return MyVolume(Path(f"{temp_base_dir}/temp_file_0.custom"), managed=True)

    data = np.random.uniform(1.0, 1.0, size=[16, 16, 16])

    @sb.proxify
    def foo(
        bar: sb.Resolve[str], outputs: sb.ResolveProxy[MyVolume] = sb.Output(MyVolume)
    ):
        assert bar == "42.0"
        assert outputs == f"{temp_base_dir}/temp_file_0.custom"

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):
        output_new = foo(42.0)
        assert str(output_new.path) == f"{temp_base_dir}/temp_file_0.custom"  # type: ignore

        output_numpy = foo(bar=42.0, outputs=data)
        assert str(output_numpy.path) == f"{temp_base_dir}/temp_file_0.custom"  # type: ignore

        del output_new, output_numpy


def test_named_proxy():

    PosFile = sb.namedproxy("PosFile", file_ext=".pos")

    @sb.proxify
    def foo(
        position: sb.ResolveProxy[PosFile], result: sb.ResolveProxy = sb.Output(PosFile)
    ):
        assert position == "/path/to/position.pos"

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):
        result = foo(PosFile(path=Path("/path/to/position.pos")))
        assert result.managed == True  # type: ignore
        assert result.managed == True  # type: ignore

        foo(Path("/path/to/position.pos"))


def test_proxy_group_basic():
    group = sb.ParticleStackProxy(Path("/data/my_particles"), managed=False)
    assert group.base_path == Path("/data/my_particles")
    assert group.metadata.path == Path("/data/my_particles.star")
    assert group.particle_stack.path == Path("/data/my_particles.mrcs")
    assert group.primary_proxy == group.metadata
    assert group.path == Path("/data/my_particles.star")


def test_proxy_group_new_temporary_proxy():
    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        group = sb.ParticleStackProxy.new_temporary_proxy()
        assert group.managed == True
        assert group.metadata.managed == True
        assert group.particle_stack.managed == True

        assert arc_manager.get_count(group.metadata.path) == 1
        assert arc_manager.get_count(group.particle_stack.path) == 1

        del group


@pytest.mark.filterwarnings(
    "ignore:Counting references for non-temporary file.*is deprecated"
)
def test_untyped_proxy_to_proxy_group_conversion():
    untyped = sb.Proxy(Path("/data/particles_raw"), managed=True)
    group = untyped.typed(astype=sb.ParticleStackProxy)

    assert isinstance(group, sb.ParticleStackProxy)
    assert group.base_path == Path("/data/particles_raw")
    assert group.metadata.path == Path("/data/particles_raw.star")
    assert group.particle_stack.path == Path("/data/particles_raw.mrcs")


def test_proxify_with_proxy_group():
    @sb.proxify
    def process_particles(
        stack_input: sb.ResolveProxy[sb.ParticleStackProxy],
        stack_output: sb.ResolveProxy = sb.Output(sb.ParticleStackProxy),
    ):
        assert stack_input == "/path/to/input_particles.star"
        # The output path will be a temp file base path without extension
        assert "temp_file" in str(stack_output)

        print(f"Input: {stack_input}, Output: {stack_output}")

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        input_group = sb.ParticleStackProxy(Path("/path/to/input_particles"))
        out_group = process_particles(input_group)

        assert isinstance(out_group, sb.ParticleStackProxy)
        assert out_group.managed == True
        assert str(out_group.metadata.path).endswith(".star")


def test_proxy_group_abstract_instantiation():
    class IncompleteGroup(sb.ProxyGroup):
        meta: sb.Proxy

    with pytest.raises(TypeError):
        IncompleteGroup(Path("/data/test"))


def test_proxy_group_validation_and_mapping():
    group = sb.ParticleStackProxy(Path("/data/particles"))

    # Test path and primary_proxy
    assert group.primary_proxy == group.metadata
    assert group.path == group.metadata.path

    # Test Mapping interface
    assert len(group) == 2
    assert set(group.keys()) == {"metadata", "particle_stack"}
    assert group["metadata"] == group.metadata

    assert set(iter(group)) == {"metadata", "particle_stack"}

    # Test base_path with extension error
    with pytest.raises(
        ValueError, match="ProxyGroup base_path must not have an extension"
    ):
        sb.ParticleStackProxy(Path("/data/particles.star"))

    # Test invalid kwarg name
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        sb.ParticleStackProxy(Path("/data/particles"), invalid_arg=123)

    # Test wrong proxy class kwarg
    wrong_proxy = sb.Proxy(Path("/data/particles.vol"))
    with pytest.raises(
        TypeError, match="Expected field 'metadata' to be an instance of"
    ):
        sb.ParticleStackProxy(Path("/data/particles"), metadata=wrong_proxy)


class Half1Volume(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".mrc"

    @classmethod
    def suffix(cls) -> Optional[str]:
        return "_half1"


class Half2Volume(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".mrc"

    @classmethod
    def suffix(cls) -> Optional[str]:
        return "_half2"


class CTFMicrographProxy(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".star"

    @classmethod
    def prefix(cls) -> Optional[str]:
        return "ctf_"


class PrefixedSuffixedProxy(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".vol"

    @classmethod
    def prefix(cls) -> Optional[str]:
        return "job_"

    @classmethod
    def suffix(cls) -> Optional[str]:
        return "_final"


def test_proxy_prefix_and_suffix():
    p1 = Half1Volume(Path("/data/exp_half1.mrc"))
    assert p1.path == Path("/data/exp_half1.mrc")
    assert Half1Volume.suffix() == "_half1"
    assert Half1Volume.prefix() is None
    assert Half1Volume.extension() == ".mrc"

    p2 = CTFMicrographProxy(Path("/data/ctf_mic001.star"))
    assert p2.path == Path("/data/ctf_mic001.star")
    assert CTFMicrographProxy.prefix() == "ctf_"
    assert CTFMicrographProxy.suffix() is None

    p3 = PrefixedSuffixedProxy(Path("/data/job_run01_final.vol"))
    assert p3.path == Path("/data/job_run01_final.vol")


def test_proxy_metaclass_prefix_and_suffix_validation():
    # Valid resolutions
    p1 = current_registry().resolve(Path("/data/exp_half1.mrc"), Half1Volume)
    assert isinstance(p1, Half1Volume)
    assert p1.path == Path("/data/exp_half1.mrc")

    p2 = current_registry().resolve(Path("/data/ctf_mic001.star"), CTFMicrographProxy)
    assert isinstance(p2, CTFMicrographProxy)
    assert p2.path == Path("/data/ctf_mic001.star")

    # Mismatch suffix
    with pytest.raises(TypeError, match="The file suffix did not match the proxy"):
        current_registry().resolve(Path("/data/exp_half2.mrc"), Half1Volume)

    # Mismatch prefix
    with pytest.raises(TypeError, match="The file prefix did not match the proxy"):
        current_registry().resolve(Path("/data/raw_mic001.star"), CTFMicrographProxy)

    # Mismatch extension
    with pytest.raises(TypeError, match="The file extension did not match the proxy"):
        current_registry().resolve(Path("/data/exp_half1.vol"), Half1Volume)


def test_untyped_to_prefixed_suffixed_proxy():
    untyped = sb.Proxy(Path("/data/volume"), managed=False)

    typed_half1 = untyped.typed(astype=Half1Volume, copy_data=False)
    assert typed_half1.path == Path("/data/volume_half1.mrc")

    typed_ctf = untyped.typed(astype=CTFMicrographProxy, copy_data=False)
    assert typed_ctf.path == Path("/data/ctf_volume.star")

    typed_both = untyped.typed(astype=PrefixedSuffixedProxy, copy_data=False)
    assert typed_both.path == Path("/data/job_volume_final.vol")


def test_new_temporary_proxy_with_prefix_and_suffix():
    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        # Without base_path
        p1 = Half1Volume.new_temporary_proxy()
        assert str(p1.path) == f"{temp_base_dir}/temp_file_0_half1.mrc"

        p2 = CTFMicrographProxy.new_temporary_proxy()
        assert str(p2.path) == f"{temp_base_dir}/ctf_temp_file_1.star"

        p3 = PrefixedSuffixedProxy.new_temporary_proxy()
        assert str(p3.path) == f"{temp_base_dir}/job_temp_file_2_final.vol"

        # With base_path
        p4 = PrefixedSuffixedProxy.new_temporary_proxy(
            base_path=Path("/custom/dir/experiment")
        )
        assert p4.path == Path("/custom/dir/job_experiment_final.vol")

        del p1, p2, p3, p4


class HalfMapsGroup(sb.ProxyGroup):
    half1: Half1Volume
    half2: Half2Volume

    @property
    def primary_proxy(self) -> sb.Proxy:
        return self.half1


class RefinementGroup(sb.ProxyGroup):
    half1: Half1Volume
    half2: Half2Volume

    @classmethod
    def prefix(cls) -> Optional[str]:
        return "run_"

    @classmethod
    def suffix(cls) -> Optional[str]:
        return "_it025"

    @property
    def primary_proxy(self) -> sb.Proxy:
        return self.half1


def test_proxy_group_with_child_suffixes():
    group = HalfMapsGroup(Path("/data/job12/map"))
    assert group.base_path == Path("/data/job12/map")
    assert group.half1.path == Path("/data/job12/map_half1.mrc")
    assert group.half2.path == Path("/data/job12/map_half2.mrc")
    assert group.primary_proxy == group.half1
    assert group.path == Path("/data/job12/map_half1.mrc")


def test_proxy_group_with_group_prefix_and_suffix():
    group = RefinementGroup(Path("/data/job12/map"))
    assert group.base_path == Path("/data/job12/map")
    # Formula: {child_prefix}{group_prefix}{base_stem}{group_suffix}{child_suffix}{ext}
    assert group.half1.path == Path("/data/job12/run_map_it025_half1.mrc")
    assert group.half2.path == Path("/data/job12/run_map_it025_half2.mrc")
    assert group.path == Path("/data/job12/run_map_it025_half1.mrc")


def test_proxy_group_new_temporary_proxy_with_suffixes():
    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        group = RefinementGroup.new_temporary_proxy()
        assert group.managed == True
        assert (
            str(group.half1.path) == f"{temp_base_dir}/run_temp_file_0_it025_half1.mrc"
        )
        assert (
            str(group.half2.path) == f"{temp_base_dir}/run_temp_file_0_it025_half2.mrc"
        )

        assert arc_manager.get_count(group.half1.path) == 1
        assert arc_manager.get_count(group.half2.path) == 1

        del group


def test_named_proxy_with_prefix_and_suffix():
    CustomProxy = sb.namedproxy(
        "CustomProxy",
        file_ext=".dat",
        prefix="raw_",
        suffix="_filtered",
    )

    p = CustomProxy(Path("/data/raw_experiment_filtered.dat"))
    assert p.path == Path("/data/raw_experiment_filtered.dat")
    assert CustomProxy.file_ext() == ".dat"
    assert CustomProxy.prefix() == "raw_"
    assert CustomProxy.suffix() == "_filtered"


def test_proxify_with_suffixed_proxy_group():
    @sb.proxify
    def refine_volumes(
        half_in: sb.ResolveProxy[RefinementGroup],
        half_out: sb.ResolveProxy = sb.Output(RefinementGroup),
    ):
        assert half_in == "/data/run_map_it025_half1.mrc"
        assert "temp_file" in str(half_out)

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        in_group = RefinementGroup(Path("/data/map"))
        out_group = refine_volumes(in_group)

        assert isinstance(out_group, RefinementGroup)
        assert out_group.managed == True
        assert str(out_group.half1.path).endswith("_it025_half1.mrc")
        assert str(out_group.half2.path).endswith("_it025_half2.mrc")


def test_proxify_with_prefixed_suffixed_proxy_output():
    @sb.proxify
    def run_job(
        inp: sb.ResolveProxy[PrefixedSuffixedProxy],
        out: sb.ResolveProxy[PrefixedSuffixedProxy] = sb.Output(PrefixedSuffixedProxy),
    ):
        assert inp.startswith("/data/job_")
        assert inp.endswith("_final.vol")
        assert "job_" in str(out)
        assert str(out).endswith("_final.vol")

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()
    with container.temp_file_provider.override(temp_file_mock):
        # 1. Test default Output(PrefixedSuffixedProxy) resolution
        input_proxy = PrefixedSuffixedProxy(Path("/data/job_sample_final.vol"))
        output_result = run_job(input_proxy)

        assert isinstance(output_result, PrefixedSuffixedProxy)
        assert output_result.managed == True
        assert output_result.path.name.startswith("job_")
        assert output_result.path.name.endswith("_final.vol")

        # 2. Test direct Output resolution from registry
        resolved_output = current_registry().resolve(
            sb.Output(PrefixedSuffixedProxy), astype=sb.Proxy
        )
        assert isinstance(resolved_output, PrefixedSuffixedProxy)
        assert resolved_output.path.name.startswith("job_")
        assert resolved_output.path.name.endswith("_final.vol")

        # 3. Test overriding output with a valid explicit Path
        custom_out_path = Path("/custom/job_run02_final.vol")
        explicit_out_result = run_job(
            Path("/data/job_sample_final.vol"), out=custom_out_path
        )
        assert isinstance(explicit_out_result, PrefixedSuffixedProxy)
        assert explicit_out_result.path == custom_out_path
        assert explicit_out_result.managed == False

        # 4. Test passing an invalid Path that violates suffix/prefix raises TypeError
        with pytest.raises(TypeError, match="The file suffix did not match"):
            run_job(Path("/data/job_sample_wrong.vol"))

        with pytest.raises(TypeError, match="The file prefix did not match"):
            run_job(Path("/data/wrong_sample_final.vol"))

        del output_result, resolved_output, explicit_out_result


class SizedChildProxyA(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".a"

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        return 1024


class SizedChildProxyB(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".b"

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        return 2048


class UnsitedChildProxy(sb.Proxy):
    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".u"


class SizedGroup(sb.ProxyGroup):
    part_a: SizedChildProxyA
    part_b: SizedChildProxyB

    @property
    def primary_proxy(self) -> sb.Proxy:
        return self.part_a


class MixedGroup(sb.ProxyGroup):
    part_a: SizedChildProxyA
    part_u: UnsitedChildProxy

    @property
    def primary_proxy(self) -> sb.Proxy:
        return self.part_a


def test_proxy_estimated_item_nbytes_default():
    """Verify that base Proxy.estimated_item_nbytes defaults to None."""
    p = sb.Proxy(Path("/data/test_file"))
    assert p.estimated_item_nbytes is None


def test_proxy_subclass_estimated_item_nbytes():
    """Verify that Proxy subclass can override estimated_item_nbytes."""
    p = SizedChildProxyA(Path("/data/test_file.a"))
    assert p.estimated_item_nbytes == 1024


def test_proxy_group_estimated_item_nbytes_aggregation():
    """Verify ProxyGroup.estimated_item_nbytes sums child estimates, and returns None if any is None."""
    # SizedGroup: 1024 + 2048 = 3072
    group = SizedGroup(Path("/data/item"))
    assert group.estimated_item_nbytes == 3072

    # MixedGroup: 1024 + None = None
    mixed = MixedGroup(Path("/data/item"))
    assert mixed.estimated_item_nbytes is None


def test_estimate_optimal_chunk_size():
    """Verify chunk size calculation across edge cases."""
    # Default fallback when None or <= 0
    assert sb.estimate_optimal_chunk_size(None) == 100
    assert sb.estimate_optimal_chunk_size(0) == 100
    assert sb.estimate_optimal_chunk_size(-50) == 100
    assert sb.estimate_optimal_chunk_size(None, default_chunk_size=50) == 50

    # 32MB target = 33,554,432 bytes
    # Item size 1MB (1,048,576 bytes) -> chunk_size = 32
    assert sb.estimate_optimal_chunk_size(1024 * 1024) == 32

    # Item size 256KB (262,144 bytes) -> chunk_size = 128
    assert sb.estimate_optimal_chunk_size(256 * 1024) == 128

    # Target bytes custom: 10MB (10,485,760 bytes), item size 1MB -> chunk_size = 10
    assert (
        sb.estimate_optimal_chunk_size(1024 * 1024, target_bytes=10 * 1024 * 1024) == 10
    )

    # Item larger than target: 64MB item with 32MB target -> chunk_size = 1 (not 0)
    assert sb.estimate_optimal_chunk_size(64 * 1024 * 1024) == 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_proxify_with_proxy_group()
