import os
import logging
from typing import List, Sequence
from pathlib import Path
from enum import Enum
import string
import random

from ...core.environment.cmd_exec import ShellExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import ArrayStorageProvider
from ...core.environment.protocol_config import ProtocolConfigurationProvider
from ...core.environment.domain import Domain
from ...core.protocol.protocol_base import (
    _ProtocolTypeConfiguration,
    ProtocolConfiguration,
)
from ..standalone.container import Container

from typing import Optional, Any, Dict, Type, Union, get_args

class _PyWorkflowExecProvider(ShellExecProvider):

    def __init__(self, backend, conda_env: str):
        try:
            import pwem  # type: ignore
            from pyworkflow import Config  # type: ignore
        except ImportError:
            raise ImportError(
                'Using scipion bridge with scipion requires pyworkflow option. Install it using pip install "scipion-bridge[pyworkflow]"'
            )

        self.backend = backend
        self.conda_env = conda_env

        env_act_cmd = f"conda activate {conda_env}"
        scipion_home = Config.SCIPION_HOME + os.path.sep

        act_cmd = env_act_cmd.replace(scipion_home, "", 1)
        base_env_act_cmd = pwem.Plugin.getCondaActivationCmd()

        self._conda_activation_cmd = f"{base_env_act_cmd} {act_cmd} && "

    def run(self, func_name: str, domain: "Domain", args: List[str], run_args) -> int:
        del run_args

        cmd = " ".join(args)
        if domain.isolated:
            self.backend.runJob(self._conda_activation_cmd, cmd, numberOfMpi=1)
        else:
            self.backend.runJob(cmd, "", numberOfMpi=1)

        return 0


class _PyWorkflowTempFileProvider(TemporaryFilesProvider):

    def __init__(self, backend):
        self.backend = backend

        self.temp_path = Path(os.path.abspath(backend._getTmpPath()))

        logging.info(f"Configure file with temporary path at: {self.temp_path}")

    def new_temporary_file(self, suffix: Optional[str]) -> os.PathLike:
        N = 15
        filename = "".join(random.choices(string.ascii_lowercase + string.digits, k=N))
        path = self.temp_path / filename

        if suffix is not None:
            path = path.with_suffix(suffix)

        logging.debug(f"Creating new temporary file at {path}")
        return path


class _UncompressedZarrGroupWrapper:
    """Wrapper over zarr.Group ensuring datasets created on it have compression disabled by default."""

    def __init__(self, group, compressor=None):
        self._group = group
        self._compressor = compressor

    def create_dataset(self, name: str, shape: tuple, dtype: Any, **kwargs: Any):
        kwargs.setdefault("compressor", self._compressor)
        return self._group.create_dataset(name, shape=shape, dtype=dtype, **kwargs)

    def create_array(self, name: str, shape: tuple, dtype: Any, **kwargs: Any):
        kwargs.setdefault("compressor", self._compressor)
        return self._group.create_dataset(name, shape=shape, dtype=dtype, **kwargs)

    def __getitem__(self, name: str):
        return self._group[name]

    def __setitem__(self, name: str, value: Any):
        self._group[name] = value

    def __contains__(self, name: str):
        return name in self._group


class _PyWorkflowZarrStorageProvider(ArrayStorageProvider):

    def __init__(self, backend=None, compressor=None):
        self.backend = backend
        self.compressor = compressor

    def create_group(self, shape_prefix: tuple = ()) -> Any:
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "Using Zarr storage with pyworkflow requires zarr. "
                'Install it using pip install "scipion-bridge[pyworkflow]"'
            )

        try:
            from zarr.storage import LocalStore # type: ignore

            store = LocalStore()
            group = zarr.group(store=store)
        except (ImportError, AttributeError):
            group = zarr.group()

        return _UncompressedZarrGroupWrapper(group, compressor=self.compressor)

    def concat(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        if not arrays:
            raise ValueError("concat requires at least one array")

        import zarr

        first = arrays[0]
        target = zarr.array(first)

        for a in arrays[1:]:
            target.append(a, axis=axis)

        return target



def convert_scipion_to_python(val: Any, dtype: Type) -> Any:
    """Converts a value retrieved from a PyWorkflow/Scipion Param to its declared Python type."""
    if hasattr(val, "get") and callable(getattr(val, "get", None)):
        val = val.get()

    if val is None:
        return None

    if isinstance(dtype, type) and issubclass(dtype, Enum):
        if isinstance(val, int):
            return list(dtype)[val]
        return dtype(val)

    if dtype in (int, float, bool, str):
        return dtype(val)

    return val


class _PyWorkflowProtocolConfigurationProvider(ProtocolConfigurationProvider):
    """Provider for retrieving field values from PyWorkflow/Scipion protocol instances."""

    def __init__(
        self,
        backend: Any,
        configuration: Optional[
            Union[_ProtocolTypeConfiguration, ProtocolConfiguration, Dict[str, Type]]
        ] = None,
    ):
        self.backend = backend
        self.configuration = configuration

    def _get_dtype(self, name: str) -> Optional[Type]:
        assert isinstance(self.configuration, _ProtocolTypeConfiguration)

        type_hint = self.configuration.inputs.get(
            name
        ) or self.configuration.parameters.get(name)
        args = get_args(type_hint)

        return args[0]

    def get_value(self, name: str, default: Any = None) -> Any:
        assert hasattr(
            self.backend, name
        ), f"Attribute '{name}' was not found on Scipion protocol instance '{self.backend}'."

        val = getattr(self.backend, name)
        dtype = self._get_dtype(name)

        return convert_scipion_to_python(val, dtype)  # type: ignore[arg-type]


def configure_pyworkflow_env(
    backend,
    *,
    conda_env: str,
    configuration: Optional[
        Union[_ProtocolTypeConfiguration, ProtocolConfiguration, Dict[str, Type]]
    ] = None,
    modules=None,
    packages=None,
):
    from dependency_injector import providers

    container = Container(
        shell_exec=providers.Factory(
            _PyWorkflowExecProvider, backend=backend, conda_env=conda_env
        ),
        temp_file_provider=providers.Factory(
            _PyWorkflowTempFileProvider, backend=backend
        ),
        storage_provider=providers.Factory(
            _PyWorkflowZarrStorageProvider, backend=backend
        ),
        protocol_config_provider=providers.Factory(
            _PyWorkflowProtocolConfigurationProvider,
            backend=backend,
            configuration=configuration,
        ),
    )

    container.wire(modules=modules, packages=packages)
    return container
