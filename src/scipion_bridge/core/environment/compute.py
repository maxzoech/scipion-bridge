"""Compute resources (GPUs, CPUs) that the stages of a protocol require."""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Shares of a GPU a memory claim is rounded up to.
GPU_FRACTIONS = (1 / 8, 1 / 4, 1 / 2, 1.0)

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
        min_vram: GPU memory in GiB a call needs on each of its GPUs. The
            backend claims the share of a GPU that holds it (see
            :func:`gpu_claim`), or ``gpus`` if that is more. GPU memory is
            not enforced: calls sharing a GPU must stay within their claim.
        cpus: Number of CPU cores reserved for a call. ``None`` reserves
            none: the calls may use all cores of the machine and the OS
            scheduler shares them.
        task: Whether the resources are taken per call or held across calls.
    """

    gpus: float = 0
    min_vram: Optional[float] = None
    cpus: Optional[float] = None
    task: TaskType = TaskType.EPHEMERAL

    def __post_init__(self) -> None:
        negative = [
            name
            for name, value in [
                ("gpus", self.gpus),
                ("min_vram", self.min_vram),
                ("cpus", self.cpus),
            ]
            if value is not None and value < 0
        ]
        if negative:
            raise ValueError(
                f"Compute resources must not be negative, got {self}.",
            )


def gpu_claim(resources: ComputeResources, gpu_memory: Optional[float]) -> float:
    """Number of GPUs a call requests: the larger of ``gpus`` and its memory claim.

    The memory claim is the share of a GPU holding ``min_vram``, rounded up to
    one of :data:`GPU_FRACTIONS`. ``min_vram`` applies to every GPU, so with
    ``gpus >= 1`` the count decides.

    Args:
        resources: The declared resources.
        gpu_memory: Memory in GiB of the smallest GPU, or ``None`` if unknown.
    """
    match (resources.min_vram, gpu_memory):
        case (None, _):
            return resources.gpus
        case (_, None):
            logger.warning(
                "Cannot read the GPU memory for min_vram=%s GiB; claiming whole GPUs.",
                resources.min_vram,
            )
            return max(resources.gpus, 1.0)
        case (min_vram, memory) if min_vram > memory:
            logger.warning(
                "min_vram=%s GiB does not fit on a GPU with %s GiB; claiming whole GPUs.",
                min_vram,
                memory,
            )
            return max(resources.gpus, 1.0)
        case (min_vram, memory):
            share = min(f for f in GPU_FRACTIONS if f >= min_vram / memory)
            return max(resources.gpus, share)


@dataclass(frozen=True)
class ComputeAssignment:
    """Compute resources of a stage, with the group of stages sharing them.

    Stages of the same protocol form a group; a long-running group holds its
    resources once for all of its stages.
    """

    resources: ComputeResources
    group: str
