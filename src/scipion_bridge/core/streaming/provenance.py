"""Samples of the provenance of the items of a stage, for its report.

Every stage records which columns of the Sets in its first ``SAMPLES``
items in and out are references to rows of input files, read references or
owned (``references.column_states``). Sampling reads no rows and stops after
``SAMPLES`` items each way, so it costs nothing measurable.

When ``__debug__`` is set (Python runs without ``-O``), the sampled outputs
are also checked for owned columns that copy a reference of the item the
stage received (``references.unchanged_columns``). The check hashes those
columns, about 3 ms per 32 MiB, so optimized runs skip it. The provenance
layer of the report (``report/provenance.py``) draws the samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence, Tuple, cast

from ..struct import references
from ..struct.collection import Collection
from ..struct.references import ColumnState, Stored
from ..struct.schema import SchemaConvertible
from ..struct.storage import FieldKey
from .ir import Tagged

# The items sampled in and out of every stage.
SAMPLES = 3


@dataclass(frozen=True)
class SetSample:
    """The columns of a Set in an item.

    Attributes:
        where: Where the Set is in the item, e.g. ``["out"]`` or ``[0]``;
            empty for the item itself.
        columns: The state of every initialized leaf.
        unchanged: Owned leaves whose rows equal the rows of a reference in
            the item the stage received (``references.unchanged_columns``);
            empty when ``__debug__`` is not set.
    """

    where: str
    columns: Tuple[ColumnState, ...]
    unchanged: Tuple[FieldKey, ...] = ()


# The Sets of one item.
ItemSample = Tuple[SetSample, ...]


@dataclass
class StageSamples:
    """The first items a stage received and emitted, sampled."""

    inputs: List[ItemSample] = field(default_factory=list)
    outputs: List[ItemSample] = field(default_factory=list)


class ProvenanceSampler:
    """Samples the first ``SAMPLES`` items in and out of a stage."""

    def __init__(self) -> None:
        self.samples = StageSamples()

    @property
    def sampling(self) -> bool:
        """Whether items in or out are still sampled."""
        return min(len(self.samples.inputs), len(self.samples.outputs)) < SAMPLES

    def receive(self, item: Any) -> None:
        """Sample an item the stage received, before processing it."""
        if len(self.samples.inputs) < SAMPLES:
            self.samples.inputs.append(sample_of(item))

    def emit(self, received: Optional[Any], emitted: Sequence[Any]) -> None:
        """Sample the items the stage emitted for the item ``received``
        (``None`` on FLUSH)."""
        emitted = emitted[: SAMPLES - len(self.samples.outputs)]
        if emitted:
            sources = [s for _, s in sets_of(received)]
            self.samples.outputs.extend(sample_of(out, sources) for out in emitted)


def sample_of(item: Any, received: Sequence[Stored] = ()) -> ItemSample:
    """The sample of ``item``; with ``__debug__``, owned copies of the
    references of ``received`` are marked unchanged."""
    return tuple(
        SetSample(
            where,
            tuple(references.column_states(s).values()),
            references.unchanged_columns(received, s) if __debug__ else (),
        )
        for where, s in sets_of(item)
    )


def sets_of(item: Any, where: str = "") -> Iterator[Tuple[str, Stored]]:
    """The Sets and Structs in ``item`` and its dicts, lists and tuples.

    The elements of a ``Collection`` are Structs of their own; it has no
    schema itself.
    """
    match item:
        case dict():
            for key, value in item.items():
                yield from sets_of(value, f"{where}[{key!r}]")
        case list() | tuple():
            for i, value in enumerate(item):
                yield from sets_of(value, f"{where}[{i}]")
        case Tagged(item=value):
            yield from sets_of(value, where)
        case Collection(is_descriptor=False):
            for i, element in item.items():
                yield from sets_of(element, f"{where}[{i}]")
        case SchemaConvertible(is_descriptor=False):
            yield where, cast(Stored, item)
        case _:
            pass


def prefetch_all(item: Any) -> None:
    """Read the references the policy reads ahead, in the Sets of ``item``."""
    for _, s in sets_of(item):
        references.prefetch(s)


def merge_samples(samples: Sequence[StageSamples]) -> StageSamples:
    """The samples of several stages as one, e.g. the children of a ``group_by``."""
    return StageSamples(
        inputs=[item for s in samples for item in s.inputs],
        outputs=[item for s in samples for item in s.outputs],
    )
