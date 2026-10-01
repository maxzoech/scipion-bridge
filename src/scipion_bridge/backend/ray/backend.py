from __future__ import annotations
from typing import Dict, List, Set, Any
import ray

from scipion_bridge.core.streaming.backend import (
    CompiledPipeline,
    StreamingBackendProvider,
)
from scipion_bridge.core.streaming.ir import IROp, IRSource, IRMap, IRSink
from .actors import RaySourceActor, RayWorkerActor, RaySinkActor


class RayCompiledPipeline(CompiledPipeline):
    """
    Executable compiled streaming pipeline running on Ray.
    """

    def __init__(
        self,
        sources: Dict[str, Any],
        all_actors: List[Any],
    ):
        self._sources = sources
        self._all_actors = all_actors

    def send(self, source_name: str, value: Any) -> None:
        """Push an item into a named input source."""
        if source_name not in self._sources:
            raise KeyError(
                f"Input source '{source_name}' is not registered in this pipeline. Available: {list(self._sources.keys())}",
            )
        ray.get(self._sources[source_name].push.remote(value))

    def flush(self) -> None:
        """Flush all sources in parallel and wait for pipeline completion."""
        futures = [src.flush.remote() for src in self._sources.values()]
        if futures:
            ray.get(futures)

    def close(self) -> None:
        """Terminate all actors allocated for this pipeline."""
        for actor in self._all_actors:
            try:
                ray.kill(actor)
            except Exception:
                pass


class RayBackend(StreamingBackendProvider):
    """
    Streaming backend that compiles IR DAGs into persistent Ray actors with direct P2P messaging.
    """

    def __init__(self, init_ray: bool = True):
        if init_ray and not ray.is_initialized():
            import os
            import sys

            extra_paths = [os.getcwd(), os.path.abspath("src")]
            tests_dir = os.path.abspath("tests")
            if os.path.exists(tests_dir):
                for root, _, _ in os.walk(tests_dir):
                    extra_paths.append(root)

            all_paths = list(
                dict.fromkeys(extra_paths + [os.path.abspath(p) for p in sys.path if p])
            )
            python_path = ":".join(all_paths)
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"PYTHONPATH": python_path}},
            )

    def compile(self, ir_sinks: List[IROp]) -> RayCompiledPipeline:
        """Compile a list of IR sink nodes into an executable RayCompiledPipeline."""
        if not ir_sinks:
            raise ValueError("RayBackend.compile() requires at least one IRSink.")

        # 1. Discover all reachable IR nodes in the DAG
        all_nodes: Set[IROp] = set()
        sources_map: Dict[str, IRSource] = {}

        def _traverse(node: IROp) -> None:
            if node in all_nodes:
                return
            all_nodes.add(node)
            if isinstance(node, IRSource):
                sources_map[node.name] = node
            for up in node.upstream:
                _traverse(up)

        for sink in ir_sinks:
            _traverse(sink)

        # 2. Instantiate Ray actors for every IR node
        actor_map: Dict[IROp, Any] = {}
        for node in all_nodes:
            match node:
                case IRSource(name=name):
                    actor = RaySourceActor.remote(name=name)
                case IRMap(func=func):
                    actor = RayWorkerActor.remote(func=func)
                case IRSink(writer=writer):
                    actor = RaySinkActor.remote(writer=writer)
                case _:
                    raise NotImplementedError(
                        f"Unsupported IR op type for Ray backend: {type(node).__name__}",
                    )
            actor_map[node] = actor

        # 3. Wire downstream actor handles
        wire_futures: List[Any] = []
        for node, actor in actor_map.items():
            if isinstance(node, (IRSource, IRMap)):
                downstream_handles = [actor_map[d] for d in node.downstream]
                wire_futures.append(actor.set_downstream.remote(downstream_handles))

        if wire_futures:
            ray.get(wire_futures)

        compiled_sources = {
            name: actor_map[src_node] for name, src_node in sources_map.items()
        }
        return RayCompiledPipeline(
            sources=compiled_sources,
            all_actors=list(actor_map.values()),
        )
