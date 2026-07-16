from dependency_injector import containers, providers

from ...core.environment.cmd_exec import StandaloneExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    shell_exec = providers.Factory(StandaloneExecProvider)
    temp_file_provider = providers.Factory(TemporaryFilesProvider)


def configure_default_env(modules=None, packages=["scipion_bridge"]):
    container = Container()
    
    container.wire(modules=modules, packages=packages)
    return container
