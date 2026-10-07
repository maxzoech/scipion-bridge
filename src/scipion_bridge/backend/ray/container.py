from typing import Any, Mapping, Optional

from dependency_injector import containers, providers
from ...core.environment.cmd_exec import StandaloneExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import ArrowStorageProvider
from ...core.environment.protocol_config import StaticProtocolConfigurationProvider
from .backend import RayBackend
from .resource_provider import RayResourceProvider


def configure_ray_env(
    modules=None,
    packages=None,
    parameters: Optional[Mapping[str, Any]] = None,
):
    """Configure and wire RayContainer.

    Args:
        parameters: Values of the protocol parameters, served to ``Field.value``.
    """
    if packages is None:
        packages = ["scipion_bridge"]
    container = RayContainer(
        parameters=providers.Object(dict(parameters or {})),
    )
    container.wire(modules=modules, packages=packages)
    return container


class RayContainer(containers.DeclarativeContainer):
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
    streaming_backend = providers.Factory(RayBackend)


configure_ray_container = configure_ray_env
