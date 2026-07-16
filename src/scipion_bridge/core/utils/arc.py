import os
from typing import Dict
import warnings
from pathlib import Path

from dependency_injector.wiring import Provide, inject
from ...backend.standalone.container import Container
from ..environment.temp_files import TemporaryFilesProvider

from typing import Optional


class FileReferenceCounter:

    def __init__(self) -> None:
        self.references: Dict[os.PathLike, int] = {}

    def new_managed_file(
        self,
        file_ext: Optional[str],
        temp_file_provider: TemporaryFilesProvider = Provide[
            Container.temp_file_provider
        ],
    ) -> Path:

        new_path = temp_file_provider.new_temporary_file(file_ext)
        assert isinstance(new_path, Path)

        self.references[new_path] = 1

        return new_path

    def add_reference(self, path: os.PathLike):
        if path not in self.references:
            self.references[path] = 1
            warnings.warn(
                "Counting references for non-temporary files is deprecated (these files are most likely created by the user at a persistent location, so reference counting would delete them)",
                DeprecationWarning,
            )
        else:
            count = self.references[path]
            self.references[path] = count + 1

    @inject
    def remove_reference(
        self,
        path: os.PathLike,
        temp_file_provider: TemporaryFilesProvider = Provide[
            Container.temp_file_provider
        ],
    ):
        assert (
            path in self.references
        ), f"Path {path} not managed by automatic reference counting"
        count = self.references[path]

        if count > 1:
            self.references[path] = count - 1
        else:
            temp_file_provider.delete(path)
            del self.references[path]

    def is_tracked(self, path: os.PathLike):
        return path in self.references

    def get_count(self, path: os.PathLike):
        if not self.is_tracked(path):
            warnings.warn(
                "Reference count requested for untracked path {path}; returned 0",
                UserWarning,
            )
            return 0
        else:
            return self.references[path]

    def _print_reference_count(self):  # pragma: no cover
        from tabulate import tabulate

        print(tabulate(self.references.items(), headers=["Path", "Ref. Count"]))


manager = FileReferenceCounter()
