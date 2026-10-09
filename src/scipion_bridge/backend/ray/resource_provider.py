import math
from typing import Any, Callable, Dict, Optional
import ray

from ...core.environment.compute import numpy_hugepage_env
from ...core.environment.resource_provider import ResourceProvider, ResourceScope

# Share of the cluster's CPU cores a SHARED resource is built with: enough to
# precompute constants quickly without stalling the rest of the pipeline.
SHARED_BUILD_CPU_FRACTION = 0.35


@ray.remote
def _build_shared(builder: Callable[[Any], Any], instance: Any) -> Any:
    """Build a SHARED resource, holding the CPU cores of the task meanwhile."""
    return builder(instance)


def _shared_build_cpus() -> int:
    """CPU cores of a SHARED build; Ray limits its NumPy/BLAS threads to them."""
    cluster_cpus = ray.cluster_resources().get("CPU", 1)
    return max(1, math.floor(SHARED_BUILD_CPU_FRACTION * cluster_cpus))


@ray.remote(num_cpus=0)
class RayResourceCoordinator:
    """Cluster-wide coordinator managing shared ObjectRefs in Ray Plasma store.

    Every resource is built once, by a task holding a share of the cluster's
    CPU cores only while it builds; the coordinator itself holds none.
    """

    def __init__(self) -> None:
        self._refs: Dict[str, ray.ObjectRef] = {}

    def get_or_build(
        self,
        key: str,
        builder: Callable[[Any], Any],
        instance: Any,
    ) -> ray.ObjectRef:
        if key not in self._refs:
            self._refs[key] = _build_shared.options(
                num_cpus=_shared_build_cpus(),
                runtime_env={"env_vars": numpy_hugepage_env()},
            ).remote(builder, instance)
        return self._refs[key]

    def clear(self) -> None:
        self._refs.clear()


class RayResourceProvider(ResourceProvider):
    """Ray-aware resource provider supporting worker-local and cluster-shared resources."""

    def __init__(self) -> None:
        self._local_cache: Dict[str, Any] = {}

    def clear(self) -> None:
        """Clear the worker-local resource cache."""
        self._local_cache.clear()

    @classmethod
    def reset_coordinator(cls) -> None:
        """Reset the cluster coordinator's cached object references if active."""
        if ray.is_initialized():
            try:
                coord = ray.get_actor(
                    "RayResourceCoordinator", namespace="scipion_bridge"
                )
                ray.get(coord.clear.remote())
            except ValueError:
                pass

    def get_resource(
        self,
        name: str,
        builder: Callable[[Any], Any],
        instance: Any,
        scope: ResourceScope = ResourceScope.PROCESS,
        dtype: Optional[Any] = None,
    ) -> Any:
        key = f"{instance.protocol_id}:{name}"

        if key in self._local_cache:
            return self._local_cache[key]

        match scope:
            case ResourceScope.PROCESS:
                res = builder(instance)

            case ResourceScope.SHARED:
                if not ray.is_initialized():
                    raise RuntimeError(
                        "Ray must be initialized to access SHARED resources.",
                    )
                coordinator = RayResourceCoordinator.options(
                    name="RayResourceCoordinator",
                    namespace="scipion_bridge",
                    lifetime="detached",
                    get_if_exists=True,
                    runtime_env={"env_vars": numpy_hugepage_env()},
                ).remote()

                ref = ray.get(
                    coordinator.get_or_build.remote(key, builder, instance)  # type: ignore
                )
                res = ray.get(ref)

            case _:
                raise ValueError(f"Unknown ResourceScope: {scope}")

        if dtype is not None and isinstance(dtype, type) and not isinstance(res, dtype):
            raise TypeError(
                f"Resource '{name}' expected type {dtype}, got {type(res)}.",
            )

        self._local_cache[key] = res
        return res
