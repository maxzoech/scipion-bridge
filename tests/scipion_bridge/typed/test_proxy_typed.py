import os
import logging
import warnings
from pathlib import Path


from scipion_bridge.core.typed.resolve import (
    resolution_context,
    Resolve,
    current_registry,
    Registry,
    resolver,
)

from scipion_bridge.core.typed import proxy, common
from scipion_bridge.core.typed.proxy import proxify
from scipion_bridge.core.typed.proxy import Proxy, Output, ProxyParam, namedproxy
from scipion_bridge.core.environment.container import Container
from scipion_bridge.core.utils.arc import manager as arc_manager

import pytest
from typing import Optional, Tuple, Any

# logging.basicConfig(level=logging.DEBUG)


class TempFileMock:

    def __init__(self):
        self.count = 0

    def new_temporary_file(self, suffix: str) -> os.PathLike:
        file = f"/tmp/temp_file_{self.count}{suffix}"
        self.count += 1

        return Path(file)

    def delete(self, path: os.PathLike):
        pass


class Volume(Proxy):

    @classmethod
    def file_ext(cls):
        return ".vol"


class TextFile(Proxy):

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".txt"


@pytest.mark.filterwarnings(
    "ignore:Counting references for non-temporary files is deprecated"
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

        untyped = Proxy(Path("/path/to/proxy"), managed=True)
        assert arc_manager.get_count(Path("/path/to/proxy")) == 1

        typed = untyped.typed(astype=TextFile, copy_data=False)
        assert str(typed.path) == "/path/to/proxy.txt"

        assert arc_manager.get_count(Path("/path/to/proxy")) == 1
        assert arc_manager.get_count(Path("/path/to/proxy.txt")) == 1

        del typed, untyped

    proxy_obj = Proxy(Path("/tmp/test_file"), managed=True)
    with open(proxy_obj.path, mode="w") as f:
        f.write("Hello World")

    proxy_obj = proxy_obj.typed(astype=TextFile)
    assert arc_manager.is_tracked(Path("/tmp/test_file")) == False
    assert arc_manager.is_tracked(Path("/tmp/test_file.txt")) == True

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
        p = current_registry().resolve(Output(Volume), astype=Proxy)
        assert str(p.path) == "/tmp/temp_file_0.vol"

        del p


def test_resolve_proxy():
    import os
    from pathlib import Path

    def _resolve_output_to_proxy(output: Output):
        ext = str(output.dtype.file_ext())
        return output.dtype(Path("/path/to/output" + ext), managed=False)

    registry = Registry()
    registry.add_resolver(Path, str, lambda x: str(x))
    registry.add_resolver(Proxy, Path, lambda x: x.path)
    registry.add_resolver(Output, Proxy, _resolve_output_to_proxy)

    resolved_path = registry.resolve(Proxy(Path("/path/to/file.txt")), str)
    assert resolved_path == "/path/to/file.txt"

    resolved_path = registry.resolve(Proxy("/path/to/file.txt"), str)  # type: ignore
    assert resolved_path == "/path/to/file.txt"

    resolved_proxy = registry.resolve(Output(TextFile), Proxy)
    assert str(resolved_proxy.path) == "/path/to/output.txt"

    resolved_path = registry.resolve(Output(TextFile), str)
    assert resolved_path == "/path/to/output.txt"


def test_resolve_proxified():

    @proxify
    def foo(
        inputs: ProxyParam[TextFile],
        outputs: ProxyParam = Output(TextFile),
    ) -> Optional[proxy.Proxy]:
        assert inputs == "/path/to/input.txt"
        assert outputs == "/path/to/output.txt"

        return None

    input_proxy = TextFile(Path("/path/to/input.txt"))
    output_proxy = TextFile(Path("/path/to/output.txt"))

    out = foo(input_proxy, output_proxy)
    assert out is not None
    assert str(out.path) == "/path/to/output.txt"

    out = foo(Path("/path/to/input.txt"), Path("/path/to/output.txt"))
    assert isinstance(out, Proxy)
    assert out.path == Path("/path/to/output.txt")
    assert out.managed == False


def test_resolve_proxy_multi_output():

    @proxify
    def foo(
        output_1=Output(Volume),
        output_2=Output(Volume),
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
        output: Tuple[Proxy, Proxy] = foo()  # type: ignore

        assert str(output[0].path) == "/tmp/temp_file_0.vol"
        assert str(output[1].path) == "/tmp/temp_file_1.vol"


def test_nested_proxies():

    @proxify
    def func_1(output_path=Output(TextFile)):
        assert isinstance(output_path, str)

        with open(output_path, "w+") as f:

            f.write("Write from func 1")

    @proxify
    def func_2(output_path=Output(TextFile)):
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
        output = func_2(Output(TextFile))
        assert isinstance(output, Proxy)
        assert str(output.path) == "/tmp/temp_file_0.txt"
        assert output.managed == True

        with open(output.path) as f:
            assert f.read() == "Write from func 1"


def test_return_value_warning():

    @proxify
    def foo(output: Resolve[Proxy, Output] = Output(TextFile)):
        return 42

    @proxify
    def func_1(output_path: Resolve[Proxy, Output]):
        pass

    @proxify
    def func_2():
        return func_1(Output(TextFile))

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
            foo(Output(TextFile))

        with warnings.catch_warnings(record=True) as w:
            func_2()

            assert len(w) == 0


def test_proxify_with_params():

    # logging.basicConfig(level=logging.DEBUG)

    @proxify
    def foo(
        inputs: ProxyParam[TextFile],
        outputs: Resolve[Proxy, Output] = Output(Volume),
        bar: Optional[Tuple] = None,
        *,
        value=None,
    ):

        assert inputs == "/path/to/inputs.txt"
        assert outputs == "/tmp/temp_file_0.vol"
        assert bar == "1 2 3"
        assert value == "42"

    container = Container()
    container.wire(
        modules=[
            __name__,
            "scipion_bridge.core.typed.proxy",
            "scipion_bridge.core.typed.common",
            "scipion_bridge.core.utils.arc",
        ]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):
        out = foo(Path("/path/to/inputs.txt"), bar=(1, 2, 3), value=42)

        assert out is not None
        assert str(out.path) == "/tmp/temp_file_0.vol"

        del out


def test_resolve_proxify_with_type_error():

    @proxify
    def foo(inputs: ProxyParam[TextFile]):
        assert inputs == "/path/to/text_file.txt"

    with pytest.raises(TypeError):
        foo(Volume(Path("/path/to/volume.vol")))  # Fails because wrong type
        foo(Path("/path/to/volume.vol"))  # Fails because of wrong extension

    foo(Path("/path/to/text_file.txt"))  # Correctly resolves


def test_combine_proxify_and_resolve():
    import numpy as np

    class MyVolume(Proxy):

        @classmethod
        def file_ext(cls):
            return ".custom"

    class OtherVolume(Proxy):

        @classmethod
        def file_ext(cls):
            return ".something"

    @resolver
    def resolve_numpy_to_my_volume2(value: np.ndarray) -> OtherVolume:
        return OtherVolume(Path("/path/to/volume.something"), managed=True)

    @resolver
    def resolve_numpy_to_my_volume(value: np.ndarray) -> MyVolume:
        return MyVolume(Path("/tmp/temp_file_0.custom"), managed=True)

    data = np.random.uniform(1.0, 1.0, size=[16, 16, 16])

    @proxify
    def foo(bar: Resolve[str], outputs: proxy.ProxyParam[MyVolume] = Output(MyVolume)):
        assert bar == "42.0"
        assert outputs == "/tmp/temp_file_0.custom"

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
        assert str(output_new.path) == "/tmp/temp_file_0.custom"  # type: ignore

        output_numpy = foo(bar=42.0, outputs=data)
        assert str(output_numpy.path) == "/tmp/temp_file_0.custom"  # type: ignore

        del output_new, output_numpy


def test_named_proxy():

    PosFile = namedproxy("PosFile", file_ext=".pos")

    @proxify
    def foo(position: ProxyParam[PosFile], result: ProxyParam = Output(PosFile)):
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


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    test_combine_proxify_and_resolve()
