from dependency_injector import containers, providers

from ...core.environment.cmd_exec import StandaloneExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import NumPyStorageProvider
from ...core.environment.protocol_config import ProtocolConfigurationProvider


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    shell_exec = providers.Factory(StandaloneExecProvider)
    temp_file_provider = providers.Factory(TemporaryFilesProvider)
    storage_provider = providers.Factory(NumPyStorageProvider)
    protocol_config_provider = providers.AbstractFactory(ProtocolConfigurationProvider)


def configure_default_env(modules=None, packages=["scipion_bridge"]):
    container = Container()
    
    container.wire(modules=modules, packages=packages)
    return container
