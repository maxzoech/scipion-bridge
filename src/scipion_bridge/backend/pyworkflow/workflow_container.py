import os
import logging
from typing import List
from pathlib import Path
import string
import random

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from scipion_bridge.core.environment import temp_files
from scipion_bridge.core.environment.cmd_exec import ShellExecProvider
from scipion_bridge.core.environment.temp_files import TemporaryFilesProvider
from scipion_bridge.core.utils.external_call import Domain

from scipion_bridge.core.environment.container import Container as CoreContainer
from typing import Optional

import pwem # type: ignore
from pyworkflow import Config # type: ignore

class _PyWorkflowExecProvider(ShellExecProvider):

    def __init__(self, backend, conda_env: str):
        self.backend = backend
        self.conda_env = conda_env

        env_act_cmd = f"conda activate {conda_env}"
        scipion_home = Config.SCIPION_HOME + os.path.sep

        act_cmd = env_act_cmd.replace(scipion_home, "", 1)
        base_env_act_cmd = pwem.Plugin.getCondaActivationCmd()

        self._conda_epilogue = f"{base_env_act_cmd} {act_cmd}"

    def run(self, func_name, domain: Domain, args: List[str], run_args):
        del run_args

        domain_cmd = " ".join(domain.command)
        cmd = f"{self._conda_epilogue} && {domain_cmd} {func_name}"
        
        self.backend.runJob(cmd, args, numberOfMpi=1)


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


def configure_pyworkflow_env(backend, *, conda_env: str, modules=None, packages=None):
    from dependency_injector import providers

    container = CoreContainer(
        shell_exec=providers.Factory(_PyWorkflowExecProvider, backend=backend, conda_env=conda_env),
        temp_file_provider=providers.Factory(_PyWorkflowTempFileProvider, backend=backend),
    )

    container.wire(modules=modules, packages=packages)
    return container