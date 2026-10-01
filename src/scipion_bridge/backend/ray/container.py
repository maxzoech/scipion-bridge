from dependency_injector import containers, providers
from ...core.environment.cmd_exec import StandaloneExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import ArrowStorageProvider
from ...core.environment.protocol_config import ProtocolConfigurationProvider
from .backend import RayBackend


class RayContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    shell_exec = providers.Factory(StandaloneExecProvider)
    temp_file_provider = providers.Factory(TemporaryFilesProvider)
    storage_provider = providers.Factory(ArrowStorageProvider)
    protocol_config_provider = providers.AbstractFactory(ProtocolConfigurationProvider)
    streaming_backend = providers.Factory(RayBackend)


def configure_ray_env(modules=None, packages=None):
    if packages is None:
        packages = ["scipion_bridge"]
    container = RayContainer()
    container.wire(modules=modules, packages=packages)
    return container
