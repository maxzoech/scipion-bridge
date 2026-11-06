"""Services a backend provides to protocols, and the compute resources they need."""

from .cmd_exec import ShellExecProvider, StandaloneExecProvider
from .compute import ComputeAssignment, ComputeResources, TaskType
from .domain import Domain
from .protocol_config import (
    ProtocolConfigurationProvider,
    StaticProtocolConfigurationProvider,
)
from .resource_provider import DefaultResourceProvider, ResourceProvider, ResourceScope
from .storage import ArrayStorageProvider, ArrowStorageProvider, NumPyStorageProvider
from .temp_files import TemporaryFilesProvider

__all__ = [
    # Compute resources
    "ComputeResources",
    "ComputeAssignment",
    "TaskType",
    # Services
    "Domain",
    "ShellExecProvider",
    "StandaloneExecProvider",
    "ProtocolConfigurationProvider",
    "StaticProtocolConfigurationProvider",
    "ResourceScope",
    "ResourceProvider",
    "DefaultResourceProvider",
    "ArrayStorageProvider",
    "ArrowStorageProvider",
    "NumPyStorageProvider",
    "TemporaryFilesProvider",
]
