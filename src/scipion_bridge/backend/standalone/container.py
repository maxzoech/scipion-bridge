from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from ...core.environment.cmd_exec import ShellExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.domain import Domain


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    shell_exec = providers.Factory(ShellExecProvider)
    temp_file_provider = providers.Factory(TemporaryFilesProvider)


def configure_default_env(modules=None, packages=None):
    container = Container()
    if modules is None:
        modules = ["scipion_bridge"]
    if packages is None:
        packages = ["scipion_bridge"]
    container.wire(modules=modules, packages=packages)
    return container
