import os

from pathlib import Path
from scipion_bridge.core.utils.arc import FileReferenceCounter
from scipion_bridge.backend.standalone.container import Container

import pytest


class TempFileMock:

    def __init__(self):
        self.count = 0

    def new_temporary_file(self, suffix: str) -> os.PathLike:
        file = f"/tmp/temp_file_{self.count}{suffix}"
        self.count += 1

        return Path(file)

    def delete(self, path: os.PathLike):
        pass


def test_reference_counting():
    manager = FileReferenceCounter()

    container = Container()
    container.wire(
        modules=[__name__, "scipion_bridge.core.typed.proxy", "scipion_bridge.core.utils.arc"]
    )

    temp_file_mock = TempFileMock()

    with container.temp_file_provider.override(temp_file_mock):

        path_1 = manager.new_managed_file("txt")
        path_2 = manager.new_managed_file("vol")  # Path("/path/to/file_2.txt")

        manager.add_reference(path_1)

        assert manager.get_count(path_1) == 2

        manager.remove_reference(path_1)
        manager.add_reference(path_2)

        assert manager.get_count(path_1) == 1
        assert manager.get_count(path_2) == 2

        manager.remove_reference(path_1)

        with pytest.raises(AssertionError):
            manager.remove_reference(path_1)


if __name__ == "__main__":
    test_reference_counting()
