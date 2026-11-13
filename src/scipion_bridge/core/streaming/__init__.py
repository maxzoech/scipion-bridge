"""Declarative streaming pipelines, lowered to an IR that backends execute."""

from .backend import (
    CompiledPipeline,
    StageStats,
    StreamingBackendProvider,
    merge_stats,
    throughput_of,
)
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
    stage_labels,
    topological_order,
)
from .keyed import share_keys
from .node import FLUSH, FlushSignal, LoweringContext, lower
from .ops import Op, Source
from .pipeline import Pipeline
from .report.model import (
    GpuMapPanel,
    GpuOccupant,
    GpuSlot,
    Layer,
    PipelineReport,
    ReportEdge,
    ReportNode,
    TablePanel,
    report_of,
)
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
    "merge_stats",
    "throughput_of",
    "SpillStore",
    "SpillStoreFactory",
    "PickleSpillStore",
    "pickle_spill_store",
    # Reports
    "PipelineReport",
    "ReportNode",
    "ReportEdge",
    "Layer",
    "TablePanel",
    "GpuMapPanel",
    "GpuSlot",
    "GpuOccupant",
    "report_of",
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
    "topological_order",
    "stage_labels",
    "share_keys",
]
