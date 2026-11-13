"""The provenance layer of a report: which columns of the items are rows of
input files, where they are read, and where a stage turns them into owned
columns (debug view).

Built from the samples of the stages (``streaming/provenance.py``). A
reference lost at a stage costs bytes, never correctness: items spilled
after it write the rows the input files hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...struct import references
from ...struct.references import ColumnState, State
from ...struct.storage import FieldKey
from ..provenance import ItemSample, StageSamples, merge_samples
from .model import Annotations, Layer, PipelineReport, TablePanel, template_label

# The cells of the panel *Key paths*.
_CELLS: Mapping[State, str] = {"reference": "ref", "cached": "ref*", "owned": "own"}


@dataclass(frozen=True)
class _Stage:
    """What the samples of a stage tell."""

    # Per item emitted (or received, for a stage emitting nothing).
    reference_bytes: float
    cached_bytes: float
    owned_bytes: float
    spill_bytes: float
    # The state of every leaf path in the items emitted, else received.
    states: Mapping[FieldKey, State]
    # References received that left owned, and unread ones that left read.
    computed_here: Tuple[FieldKey, ...]
    read_here: Tuple[FieldKey, ...]
    # Leaves of computed_here whose values are unchanged.
    unchanged: Tuple[FieldKey, ...]
    lost_bytes: float


def provenance_layer(
    report: PipelineReport,
    samples: Mapping[str, StageSamples],
) -> Layer:
    """The provenance of the columns of the items as a layer of ``report``.

    The samples of the children of a ``group_by`` are merged into the nodes
    of its template. Phase boundaries are the edges annotated ``boundary``
    by the other layers of ``report``.
    """
    by_node: Dict[str, List[StageSamples]] = {}
    for label, stage in samples.items():
        by_node.setdefault(template_label(label), []).append(stage)
    stages = {
        node.id: stage
        for node in report.nodes
        if node.id in by_node
        and (stage := _stage_of(merge_samples(by_node[node.id]))) is not None
    }
    titles = {node.id: node.title for node in report.nodes}
    return Layer(
        name="provenance",
        title="Provenance",
        nodes={node_id: _node_annotations(stage) for node_id, stage in stages.items()},
        edges={
            (edge.source, edge.target): _edge_annotations(
                stages[edge.source],
                stages.get(edge.target),
            )
            for edge in report.edges
            if edge.source in stages
        },
        metric="reference_fraction",
        panels=_panels(report, stages),
        warnings=tuple(
            _lost_warning(titles[node_id], stage)
            for node_id, stage in stages.items()
            if stage.computed_here
        ),
    )


def _stage_of(samples: StageSamples) -> Optional[_Stage]:
    items = samples.outputs or samples.inputs
    if not items:
        return None
    received = _states(samples.inputs)
    states = _states(items)
    totals = {
        state: sum(_bytes(item, state) for item in items) / len(items)
        for state in _CELLS
    }
    computed_here = tuple(
        path
        for path, state in received.items()
        if state != "owned" and states.get(path) == "owned"
    )
    unchanged = {path for item in items for s in item for path in s.unchanged}
    return _Stage(
        reference_bytes=totals["reference"],
        cached_bytes=totals["cached"],
        owned_bytes=totals["owned"],
        spill_bytes=sum(_spilled(item) for item in items) / len(items),
        states=states,
        computed_here=computed_here,
        read_here=tuple(
            path
            for path, state in received.items()
            if state == "reference" and states.get(path) == "cached"
        ),
        unchanged=tuple(path for path in computed_here if path in unchanged),
        lost_bytes=sum(
            c.nbytes
            for item in items
            for s in item
            for c in s.columns
            if c.path in computed_here
        )
        / len(items),
    )


def _states(items: Sequence[ItemSample]) -> Dict[FieldKey, State]:
    """The state of every leaf path; the first item that has it decides."""
    states: Dict[FieldKey, State] = {}
    for column in (c for item in items for s in item for c in s.columns):
        states.setdefault(column.path, column.state)
    return states


def _bytes(item: ItemSample, state: State) -> int:
    return sum(c.nbytes for s in item for c in s.columns if c.state == state)


def _spilled(item: ItemSample) -> int:
    """The bytes spilling ``item`` writes, as the reference policy decides."""
    return sum(c.nbytes for s in item for c in s.columns if _keeps_bytes(c))


def _keeps_bytes(column: ColumnState) -> bool:
    return column.state == "owned" or references.policy().keep_bytes_when_spilling(
        column,
    )


def _node_annotations(stage: _Stage) -> Annotations:
    total = stage.reference_bytes + stage.cached_bytes + stage.owned_bytes
    return {
        "reference_bytes": round(stage.reference_bytes),
        "cached_bytes": round(stage.cached_bytes),
        "owned_bytes": round(stage.owned_bytes),
        "reference_fraction": (
            (stage.reference_bytes + stage.cached_bytes) / total if total else 0.0
        ),
        "computed_here": tuple(_name(p) for p in stage.computed_here),
        "read_here": tuple(_name(p) for p in stage.read_here),
        "unchanged": tuple(_name(p) for p in stage.unchanged),
    }


def _edge_annotations(source: _Stage, target: Optional[_Stage]) -> Annotations:
    total = source.reference_bytes + source.cached_bytes + source.owned_bytes
    values: Dict[str, bool | int] = {
        "item_bytes": round(total),
        "spill_bytes": round(source.spill_bytes),
    }
    if target is not None and target.computed_here:
        values["lost"] = True
    return values


def _panels(
    report: PipelineReport,
    stages: Mapping[str, _Stage],
) -> Tuple[TablePanel, ...]:
    referenced = sorted(
        {
            path
            for stage in stages.values()
            for path, state in stage.states.items()
            if state != "owned"
        },
    )
    panels: List[TablePanel] = []
    if referenced:
        panels.append(
            TablePanel(
                "Key paths",
                ("stage", *(_name(p) for p in referenced), "other paths"),
                tuple(
                    (
                        node.title,
                        *(_cell(stages[node.id], path) for path in referenced),
                        sum(1 for p in stages[node.id].states if p not in referenced),
                    )
                    for node in report.nodes
                    if node.id in stages
                ),
            ),
        )
    boundaries = [
        edge
        for layer in report.layers
        for edge, values in layer.edges.items()
        if "boundary" in values and edge[0] in stages
    ]
    if boundaries:
        titles = {node.id: node.title for node in report.nodes}
        panels.append(
            TablePanel(
                "Spill estimate",
                ("boundary", "bytes/item", "spilled/item", "saved"),
                tuple(
                    _spill_row(f"{titles[s]} → {titles[t]}", stages[s])
                    for s, t in boundaries
                ),
            ),
        )
    return tuple(panels)


def _cell(stage: _Stage, path: FieldKey) -> str:
    state = stage.states.get(path)
    if path in stage.computed_here:
        return "C"
    else:
        return "·" if state is None else _CELLS[state]


def _spill_row(label: str, stage: _Stage) -> Tuple[str, str, str, str]:
    total = stage.reference_bytes + stage.cached_bytes + stage.owned_bytes
    saved = 1 - stage.spill_bytes / total if total else 0.0
    return (label, _size(total), _size(stage.spill_bytes), f"{saved:.0%}")


def _lost_warning(title: str, stage: _Stage) -> str:
    paths = ", ".join(_name(p) for p in stage.computed_here)
    warning = (
        f"{paths} lose their reference to the input files at {title}: items "
        f"spilled after it write {_size(stage.lost_bytes)} per item that the "
        "files hold."
    )
    if stage.unchanged:
        unchanged = ", ".join(_name(p) for p in stage.unchanged)
        warning += (
            f" The values of {unchanged} are unchanged (copied): keeping the "
            "columns of the input keeps the reference."
        )
    return warning


def _name(path: FieldKey) -> str:
    return ".".join(path)


def _size(nbytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    power = 0
    while nbytes >= 1024 and power < len(units) - 1:
        nbytes /= 1024
        power += 1
    return f"{nbytes:.0f} B" if power == 0 else f"{nbytes:.1f} {units[power]}"
