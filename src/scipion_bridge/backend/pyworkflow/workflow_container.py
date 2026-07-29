import os
import logging
from typing import List, Sequence
from pathlib import Path
import string
import random

from ...core.environment.cmd_exec import ShellExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import ArrayStorageProvider
from ...core.environment.domain import Domain

from ..standalone.container import Container
from typing import Optional, Any

class _PyWorkflowExecProvider(ShellExecProvider):

    def __init__(self, backend, conda_env: str):
        try:
            import pwem # type: ignore
            from pyworkflow import Config # type: ignore
        except ImportError:
            raise ImportError("Using scipion bridge with scipion requires pyworkflow option. Install it using pip install \"scipion-bridge[pyworkflow]\"")

        self.backend = backend
        self.conda_env = conda_env

        env_act_cmd = f"conda activate {conda_env}"
        scipion_home = Config.SCIPION_HOME + os.path.sep

        act_cmd = env_act_cmd.replace(scipion_home, "", 1)
        base_env_act_cmd = pwem.Plugin.getCondaActivationCmd()

        self._conda_epilogue = f"{base_env_act_cmd} {act_cmd}"

    def run(self, func_name: str, domain: "Domain", args: List[str], run_args):
        del run_args

        cmd = " ".join(args)
        if domain.isolated:
            self.backend.runJob(self._conda_epilogue, cmd, numberOfMpi=1)
        else:
            self.backend.runJob(cmd, "", numberOfMpi=1)

class _PyWorkflowTempFileProvider(TemporaryFilesProvider):

    def __init__(self, backend):
        self.backend = backend

        self.temp_path = Path(os.path.abspath(backend._getTmpPath()))

        logging.info(f"Configure file with temporary path at: {self.temp_path}")

    def new_temporary_file(self, suffix: Optional[str]) -> os.PathLike:
        N = 15
        filename = "".join(random.choices(string.ascii_lowercase + string.digits, k=N))
        path = (self.temp_path / filename).with_suffix(suffix)

        logging.debug(f"Creating new temporary file at {path}")
        return path


class _PyWorkflowZarrStorageProvider(ArrayStorageProvider):

    def __init__(self, backend=None):
        self.backend = backend

    def create_group(self, shape_prefix: tuple = ()) -> Any:
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "Using Zarr storage with pyworkflow requires zarr. "
                "Install it using pip install \"scipion-bridge[pyworkflow]\""
            )

        from zarr.storage import LocalStore
        store = LocalStore()
        return zarr.group(store=store)

    def concat(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        if not arrays:
            raise ValueError("concat requires at least one array")

        import zarr
        first = arrays[0]
        target = zarr.array(first)
        
        for a in arrays[1:]:
            target.append(a, axis=axis)

        return target


def configure_pyworkflow_env(backend, *, conda_env: str, modules=None, packages=None):
    from dependency_injector import providers

    container = Container(
        shell_exec=providers.Factory(_PyWorkflowExecProvider, backend=backend, conda_env=conda_env),
        temp_file_provider=providers.Factory(_PyWorkflowTempFileProvider, backend=backend),
        storage_provider=providers.Factory(_PyWorkflowZarrStorageProvider, backend=backend),
    )

    container.wire(modules=modules, packages=packages)
    return container