"""Pipelines of a ``group_by`` shared by several keys.

A ``group_by`` runs the pipeline of its keys on a few shared copies (workers)
rather than one copy per key, so that a fan-out to many keys does not start
the stages of a pipeline for each of them. The items of a shared copy carry
their key, as ``Keyed(key, item)``, through every stage:

- Maps are stateless: they apply their function to the item and keep its key.
- Accumulators keep a state per key, created with the first item of the key.
  Their functions only ever see the state and the items of one key.
- A nested ``group_by`` routes by the key of its outer ``group_by`` and its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .ir import IRAccumulate, IRDemux, IRMap, IROp, Keyed, Tagged, clone_ir


@dataclass(frozen=True)
class KeyedMap:
    """Function of a map applied to the item of a ``Keyed`` item."""

    func: Callable[[Any], Any]

    def __call__(self, keyed: Keyed) -> Keyed:
        return Keyed(key=keyed.key, value=self.func(keyed.value))


@dataclass(frozen=True)
class KeyedAccumulator:
    """Functions of an accumulator keeping a state per key."""

    accumulate_fn: Callable[[Any, Any], Tuple[Any, List[Any]]]
    initial_state_fn: Callable[[], Any]
    flush_fn: Optional[Callable[[Any], Tuple[Any, List[Any]]]]

    def initial_state(self) -> Dict[Any, Any]:
        return {}

    def accumulate(
        self,
        states: Dict[Any, Any],
        item: Any,
    ) -> Tuple[Dict[Any, Any], List[Keyed]]:
        key, value = _split_key(item)
        state = states[key] if key in states else self.initial_state_fn()
        states[key], emissions = self.accumulate_fn(state, value)
        return states, [Keyed(key=key, value=emission) for emission in emissions]

    def flush(self, states: Dict[Any, Any]) -> Tuple[Dict[Any, Any], List[Keyed]]:
        """Flush the state of every key, in the order the keys arrived."""
        assert self.flush_fn is not None, "The accumulator has no flush_fn."
        flushed = {key: self.flush_fn(state) for key, state in states.items()}
        return (
            {key: state for key, (state, _) in flushed.items()},
            [
                Keyed(key=key, value=emission)
                for key, (_, emissions) in flushed.items()
                for emission in emissions
            ],
        )


def _split_key(item: Any) -> Tuple[Any, Any]:
    """The key of an item, and the item an accumulator of one key receives."""
    match item:
        case Tagged(port=port, item=Keyed(key=key, value=value)):
            return key, Tagged(port=port, item=value)
        case Keyed(key=key, value=value):
            return key, value
        case _:
            raise TypeError(
                f"An item of a shared group_by pipeline has no key: {item!r}.",
            )


def _keyed_fields(node: IROp) -> Mapping[str, Any]:
    match node:
        case IRMap(func=func):
            return {"func": KeyedMap(func)}
        case IRAccumulate(
            accumulate_fn=accumulate_fn,
            initial_state_fn=initial_state_fn,
            flush_fn=flush_fn,
        ):
            keyed = KeyedAccumulator(accumulate_fn, initial_state_fn, flush_fn)
            return {
                "accumulate_fn": keyed.accumulate,
                "initial_state_fn": keyed.initial_state,
                "flush_fn": None if flush_fn is None else keyed.flush,
            }
        case IRDemux():
            return {"outer_keyed": True}
        case _:
            return {}


def share_keys(exit_: IROp) -> IROp:
    """Copy of a ``group_by`` pipeline carrying ``Keyed`` items of many keys.

    Args:
        exit_: Exit node of the pipeline (the template of an ``IRDemux``).

    Returns:
        The exit node of the copy.
    """
    (shared,) = clone_ir([exit_], changes=_keyed_fields)
    return shared
