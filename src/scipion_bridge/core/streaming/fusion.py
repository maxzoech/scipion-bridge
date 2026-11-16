"""Operator fusion: a compile pass running cheap maps inside a neighbouring stage.

A map without resources runs as a stage of its own; on Ray, every item then
travels to a task and back. Operators of the library that are cheap and pure
mark their map ``fusable`` (e.g. ``astype``); this pass runs them inside the
stage upstream of them, or else inside the map downstream of them:

- **Into the upstream stage** (preferred): when the map is the only
  downstream of an ``IRMap`` or ``IRAccumulate``. A map composes its
  function, an accumulator maps its emissions (of items and of the flush).
  The host takes over the downstream stages of the map.
- **Into the downstream map**: when the map has a single downstream, an
  ``IRMap`` with no other input. It keeps its compute assignment, so the
  fused map runs where it runs.

Maps are never fused into sources, sinks or ``group_by`` stages, nor across
the boundary of a ``group_by`` template; inside a template, its maps fuse
among themselves. A ``FusionPolicy`` decides whether a legal move is made.

The host keeps its identity and its fingerprint (``stage_fp``); its name
lists the operators it runs in pipeline order, e.g.
``chunk(256) → astype(Set[Particle])``, and ``fused`` holds their titles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ir import IRAccumulate, IRDemux, IRMap, IROp, describe, topological_order

# A step of an accumulator: accumulate_fn(state, item) or flush_fn(state).
Step = Callable[..., Tuple[Any, List[Any]]]


class FusionPolicy:
    """Decides whether a legal fusion is made.

    The default fuses whenever the pass may; the cost model replaces it.
    """

    def fuse(self, op: IRMap, into: IROp) -> bool:
        """Whether to run the fusable map ``op`` inside the stage ``into``."""
        return True


# The policy of ``lower``.
DEFAULT_FUSION = FusionPolicy()


class NoFusion(FusionPolicy):
    """Keeps every stage, e.g. to compare the results of a fused pipeline."""

    def fuse(self, op: IRMap, into: IROp) -> bool:
        return False


@dataclass(frozen=True)
class Then:
    """The function ``second(first(item))``."""

    first: Callable[[Any], Any]
    second: Callable[[Any], Any]

    def __call__(self, item: Any) -> Any:
        return self.second(self.first(item))


@dataclass(frozen=True)
class MapEmissions:
    """A step of an accumulator whose emitted items are mapped by ``func``."""

    step: Step
    func: Callable[[Any], Any]

    def __call__(self, *args: Any) -> Tuple[Any, List[Any]]:
        state, emitted = self.step(*args)
        return state, [self.func(item) for item in emitted]


def fuse(exits: Sequence[IROp], policy: FusionPolicy) -> List[IROp]:
    """Fuse the fusable maps of the DAG ending in ``exits``, in place.

    Args:
        exits: The exit nodes of the DAG.
        policy: Decides whether a legal fusion is made.

    Returns:
        The exits; an exit that was fused away is replaced by its host.
    """
    replaced: Dict[IROp, IROp] = {}
    for node in topological_order(exits):
        match node:
            case IRDemux(template=IROp() as template):
                (node.template,) = fuse([template], policy)
            case IRMap(fusable=True):
                host = _fuse_map(node, policy)
                if host is not None:
                    replaced[node] = host
            case _:
                pass
    return [_resolve(exit_, replaced) for exit_ in exits]


def _resolve(node: IROp, replaced: Dict[IROp, IROp]) -> IROp:
    while node in replaced:
        node = replaced[node]
    return node


def _fuse_map(op: IRMap, policy: FusionPolicy) -> Optional[IROp]:
    """Fuse ``op`` into a neighbour; returns the host, or None if it stays."""
    if len(op.upstream) != 1:
        return None

    (upstream,) = op.upstream
    if _fits_upstream(op, upstream) and policy.fuse(op, upstream):
        _into_upstream(op, upstream)
        return upstream
    downstream = _downstream_map(op)
    if downstream is not None and policy.fuse(op, downstream):
        _into_downstream(op, upstream, downstream)
        return downstream
    else:
        return None


def _fits_upstream(op: IRMap, upstream: IROp) -> bool:
    """Whether every emission of ``upstream`` goes to ``op``, and ``upstream``
    can run its function."""
    return upstream.downstream == [op] and isinstance(
        upstream,
        (IRMap, IRAccumulate),
    )


def _downstream_map(op: IRMap) -> Optional[IRMap]:
    """The map downstream of ``op`` that only consumes ``op``, if any."""
    match op.downstream:
        case [IRMap(upstream=[only]) as downstream] if only is op:
            return downstream
        case _:
            return None


def _parts(node: IROp) -> Tuple[str, ...]:
    """The titles of the operators ``node`` runs, in pipeline order."""
    return node.fused or (describe(node),)


def _into_upstream(op: IRMap, host: IROp) -> None:
    match host:
        case IRMap():
            host.func = Then(host.func, op.func)
        case IRAccumulate(flush_fn=flush_fn):
            host.accumulate_fn = MapEmissions(host.accumulate_fn, op.func)
            host.flush_fn = (
                None if flush_fn is None else MapEmissions(flush_fn, op.func)
            )
        case _:
            raise TypeError(f"Cannot fuse {describe(op)} into {describe(host)}.")
    _rename(host, (*_parts(host), *_parts(op)))
    # The host emits into the stages op emitted into, on the same ports.
    host.downstream = []
    for consumer in op.downstream:
        consumer.upstream = [host if up is op else up for up in consumer.upstream]
        host.downstream.append(consumer)
    op.upstream, op.downstream = [], []


def _into_downstream(op: IRMap, upstream: IROp, host: IRMap) -> None:
    host.func = Then(op.func, host.func)
    _rename(host, (*_parts(op), *_parts(host)))
    upstream.downstream = [host if d is op else d for d in upstream.downstream]
    host.upstream = [upstream]
    op.upstream, op.downstream = [], []


def _rename(host: IROp, parts: Tuple[str, ...]) -> None:
    assert isinstance(host, (IRMap, IRAccumulate))
    host.fused = parts
    host.name = " → ".join(parts)
