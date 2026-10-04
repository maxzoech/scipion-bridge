from dependency_injector import containers, providers

from ...core.environment.cmd_exec import StandaloneExecProvider
from ...core.environment.temp_files import TemporaryFilesProvider
from ...core.environment.storage import ArrowStorageProvider, NumPyStorageProvider
from ...core.environment.resource_provider import DefaultResourceProvider
from ...core.environment.protocol_config import ProtocolConfigurationProvider
from ...core.streaming.backend import StreamingBackendProvider

try:
    from ..ray.backend import RayBackend

    default_streaming_backend = providers.Factory(RayBackend)
except ImportError:
    default_streaming_backend = providers.AbstractFactory(StreamingBackendProvider)


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    shell_exec = providers.Factory(StandaloneExecProvider)
    temp_file_provider = providers.Factory(TemporaryFilesProvider)
    storage_provider = providers.Factory(ArrowStorageProvider)
    protocol_config_provider = providers.AbstractFactory(ProtocolConfigurationProvider)
    resource_provider = providers.Singleton(DefaultResourceProvider)
    streaming_backend = default_streaming_backend


def configure_default_env(modules=None, packages=None):
    if packages is None:
        packages = ["scipion_bridge"]
    container = Container()
    container.wire(modules=modules, packages=packages)
    return container
