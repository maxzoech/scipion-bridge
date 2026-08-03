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

    def new_temporary_file(self, suffix: str) -> os.PathLike:
        
        file = f"{temp_base_dir}/temp_file_{self.count}{suffix}"
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
    def foo(bar: sb.Resolve[str], outputs: sb.ResolveProxy[MyVolume] = sb.Output(MyVolume)):
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
    def foo(position: sb.ResolveProxy[PosFile], result: sb.ResolveProxy = sb.Output(PosFile)):
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
        assert "temp_file" in stack_output

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
    assert group.get("particle_stack") == group.particle_stack
    assert "metadata" in group
    assert "nonexistent" not in group
    assert set(iter(group)) == {"metadata", "particle_stack"}
    assert list(group.values()) == [group.metadata, group.particle_stack]

    # Test base_path with extension error
    with pytest.raises(ValueError, match="ProxyGroup base_path must not have an extension"):
        sb.ParticleStackProxy(Path("/data/particles.star"))

    # Test invalid kwarg name
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        sb.ParticleStackProxy(Path("/data/particles"), invalid_arg=123)

    # Test wrong proxy class kwarg
    wrong_proxy = sb.Proxy(Path("/data/particles.vol"))
    with pytest.raises(TypeError, match="Expected field 'metadata' to be an instance of"):
        sb.ParticleStackProxy(Path("/data/particles"), metadata=wrong_proxy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_proxify_with_proxy_group()



