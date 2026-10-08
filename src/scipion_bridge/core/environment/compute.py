"""Compute resources (GPUs, CPUs) that the stages of a protocol require."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Environment variable telling user code how many CPU cores the process may
# use; unset means all cores available to the process.
CPUS_ENV_VAR = "SCIPION_BRIDGE_CPUS"


class TaskType(str, Enum):
    """How the resources of a protocol are held while its stages run."""

    # Every call gets the resources for its own duration, so the backend can
    # interleave many calls on few resources.
    EPHEMERAL = "ephemeral"
    # The resources are held across calls, for expensive setup such as
    # loading a model, and released when the stream is flushed.
    LONG_RUNNING = "long_running"


@dataclass(frozen=True)
class ComputeResources:
    """Resources required by every call of the stages of a protocol.

    Only the Ray backend schedules by these resources; the standalone and
    pyworkflow backends ignore them.

    Attributes:
        gpus: Number of GPUs; fractions let calls share a GPU.
        cpus: Number of CPU cores reserved for a call. ``None`` reserves
            none: the calls may use all cores of the machine and the OS
            scheduler shares them.
        task: Whether the resources are taken per call or held across calls.
    """

    gpus: float = 0
    cpus: Optional[float] = None
    task: TaskType = TaskType.EPHEMERAL

    def __post_init__(self) -> None:
        if self.gpus < 0 or (self.cpus is not None and self.cpus < 0):
            raise ValueError(
                f"Compute resources must not be negative, got gpus={self.gpus}, "
                f"cpus={self.cpus}.",
            )


@dataclass(frozen=True)
class ComputeAssignment:
    """Compute resources of a stage, with the group of stages sharing them.

    Stages of the same protocol form a group; a long-running group holds its
    resources once for all of its stages.
    """

    resources: ComputeResources
    group: str
