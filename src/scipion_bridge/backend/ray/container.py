from typing import Any, Mapping, Optional

from dependency_injector import providers
from .backend import RayBackend
from .worker_env import RayWorkerContainer


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


class RayContainer(RayWorkerContainer):
    """The services of the driver: those of the workers, and the backend."""

    streaming_backend = providers.Factory(RayBackend)
