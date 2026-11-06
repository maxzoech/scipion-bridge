"""Declarative streaming pipelines, lowered to an IR that backends execute."""

from .backend import CompiledPipeline, StageStats, StreamingBackendProvider
from .element_mapper import ElementMapConfig
from .ir import (
    DEFAULT_GROUP_BY_WORKERS,
    GroupByWorkers,
    IRAccumulate,
    IRDemux,
    IRMap,
    IROp,
    IRSink,
    IRSource,
    Keyed,
    Tagged,
    WorkersFrom,
    clone_ir,
)
from .keyed import share_keys
from .node import FLUSH, FlushSignal, LoweringContext, lower
from .ops import Op, Source
from .pipeline import Pipeline
from .sink import Sink
from .sink_writer import CallbackSinkWriter, SinkWriter
from .spill import PickleSpillStore, SpillStore, SpillStoreFactory, pickle_spill_store

__all__ = [
    # Pipelines
    "Op",
    "Source",
    "Sink",
    "Pipeline",
    "SinkWriter",
    "CallbackSinkWriter",
    "FlushSignal",
    "FLUSH",
    "Keyed",
    "ElementMapConfig",
    "StageStats",
    "SpillStore",
    "SpillStoreFactory",
    "PickleSpillStore",
    "pickle_spill_store",
    # Backends: lowering and the IR
    "StreamingBackendProvider",
    "CompiledPipeline",
    "lower",
    "LoweringContext",
    "IROp",
    "IRSource",
    "IRMap",
    "IRAccumulate",
    "IRSink",
    "IRDemux",
    "Tagged",
    "WorkersFrom",
    "GroupByWorkers",
    "DEFAULT_GROUP_BY_WORKERS",
    "clone_ir",
    "share_keys",
]
