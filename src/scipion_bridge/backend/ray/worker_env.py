"""The services of the worker processes of the Ray backend.

Every stage actor, executor and task wires a ``RayWorkerContainer`` with the
parameters of its protocol. The driver wires ``RayContainer``
(``container.py``), which adds the streaming backend.
"""

from typing import Any, Mapping, Optional, Sequence

from dependency_injector import containers, providers

from ...core.environment.cmd_exec import StandaloneExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import ArrowStorageProvider
from ...core.environment.protocol_config import StaticProtocolConfigurationProvider
from .resource_provider import RayResourceProvider


class RayWorkerContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    shell_exec = providers.Factory(StandaloneExecProvider)
    temp_file_provider = providers.Factory(TemporaryFilesProvider)
    storage_provider = providers.Factory(ArrowStorageProvider)
    parameters = providers.Object({})
    protocol_config_provider = providers.Singleton(
        StaticProtocolConfigurationProvider,
        values=parameters,
    )
    resource_provider = providers.Singleton(RayResourceProvider)


def configure_worker_env(
    parameters: Optional[Mapping[str, Any]] = None,
    packages: Sequence[str] = ("scipion_bridge",),
) -> RayWorkerContainer:
    """Wire a ``RayWorkerContainer`` in a worker process.

    Args:
        parameters: Values of the protocol parameters by protocol name,
            ``{name: {parameter: value}}``, served to ``Field.value``.
    """
    container = RayWorkerContainer(
        parameters=providers.Object(dict(parameters or {})),
    )
    container.wire(packages=list(packages))
    return container
