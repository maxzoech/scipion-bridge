"""Compute resources (GPUs, CPUs) that the stages of a protocol require."""

from dataclasses import dataclass
from enum import Enum
import logging
import os
from typing import Dict, Optional

# ``numpy._core`` is the module in NumPy 2; NumPy 1.26 re-exports
# ``numpy.core`` under it. NumPy's stubs omit the function.
from numpy._core.multiarray import (
    _set_madvise_hugepage,  # pyright: ignore[reportAttributeAccessIssue]
)

logger = logging.getLogger(__name__)

# Shares of a GPU a memory claim is rounded up to.
GPU_FRACTIONS = (1 / 8, 1 / 4, 1 / 2, 1.0)

# Environment variable telling user code how many CPU cores the process may
# use; unset means all cores available to the process.
CPUS_ENV_VAR = "SCIPION_BRIDGE_CPUS"

# Environment variable telling user code the share of a GPU the process may
# use (e.g. for ``torch.cuda.set_per_process_memory_fraction``); unset means
# whole GPUs.
GPU_FRACTION_ENV_VAR = "SCIPION_BRIDGE_GPU_FRACTION"

# Environment variable NumPy reads on import: whether to advise the kernel to
# back arrays of 4 MiB or more with transparent hugepages.
NUMPY_HUGEPAGE_ENV_VAR = "NUMPY_MADVISE_HUGEPAGE"

# Part of a claimed share of a GPU that frameworks may allocate; the rest
# holds the CUDA context of the process.
_USABLE_GPU_SHARE = 0.9


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


def gpu_memory_env(num_gpus: float) -> Dict[str, str]:
    """Environment variables keeping a process within its share of a GPU.

    JAX (XLA) preallocates 75% of the memory of a GPU and TensorFlow all of
    it, so processes sharing a GPU would run out of memory. With a fraction
    of a GPU, XLA preallocates only (most of) that fraction, and TensorFlow
    allocates on demand. PyTorch allocates on demand already; user code can
    cap it with the fraction in :data:`GPU_FRACTION_ENV_VAR`.

    Args:
        num_gpus: The GPU claim of the process.
    """
    match num_gpus:
        case share if 0 < share < 1:
            return {
                "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{share * _USABLE_GPU_SHARE:.3f}",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                GPU_FRACTION_ENV_VAR: f"{share:g}",
            }
        case _:
            return {}


def numpy_hugepage_env() -> Dict[str, str]:
    """Environment variables turning off NumPy's transparent hugepage hint.

    NumPy advises the kernel to back every array of 4 MiB or more with
    transparent hugepages. With the kernel setting ``defrag=madvise``, a page
    fault in such an array compacts physical memory synchronously to build a
    2 MiB page. On a long-running host whose memory is fragmented (by the Ray
    object store in ``/dev/shm``, the page cache, pinned CUDA memory), the
    compaction mostly fails and stalls the process each time: allocating a
    stack of particles goes from a fraction of a second to a minute.

    The hint is therefore off in every process of the Ray backend, unless the
    user set ``NUMPY_MADVISE_HUGEPAGE`` in the environment of the driver, whose
    value is passed through.
    """
    match os.environ.get(NUMPY_HUGEPAGE_ENV_VAR):
        case None:
            return {NUMPY_HUGEPAGE_ENV_VAR: "0"}
        case value:
            return {NUMPY_HUGEPAGE_ENV_VAR: value}


def configure_numpy_hugepages() -> None:
    """Apply :func:`numpy_hugepage_env` to NumPy in this process.

    NumPy reads ``NUMPY_MADVISE_HUGEPAGE`` only on import, so a process that
    has imported it already (the driver) switches the hint at runtime. The
    value is parsed as NumPy parses it.
    """
    _set_madvise_hugepage(bool(int(numpy_hugepage_env()[NUMPY_HUGEPAGE_ENV_VAR])))


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
