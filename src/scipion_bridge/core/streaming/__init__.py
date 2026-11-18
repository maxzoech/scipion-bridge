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
    IRCheckpoint,
    IRSource,
    Keyed,
    Tagged,
    WorkersFrom,
    clone_ir,
    stage_labels,
    topological_order,
)
from .fusion import FusionPolicy, NoFusion
from .keyed import share_keys
from .lineage import ElementSegment, LineageRecord
from .lineage_graph import Ancestry, ElementOrigin, LineageGraph
from .node import FLUSH, FlushSignal, LoweringContext, lower
from .ops import Op, Source
from .pipeline import Pipeline
from .run_store import (
    CheckpointInfo,
    Commit,
    FileRunStore,
    RunStore,
    RunStoreMismatchError,
    StageInfo,
)
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
from .sink_writer import (
    CallbackSinkWriter,
    CommittingSinkWriter,
    SinkWriter,
    StoreSinkWriter,
    UniqueElements,
)
from .spill import PickleSpillStore, SpillStore, SpillStoreFactory, pickle_spill_store

__all__ = [
    # Pipelines
    "Op",
    "Source",
    "Sink",
    "Pipeline",
    "SinkWriter",
    "CallbackSinkWriter",
    "CommittingSinkWriter",
    "StoreSinkWriter",
    "UniqueElements",
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
    # Runs and lineage
    "RunStore",
    "FileRunStore",
    "StageInfo",
    "Commit",
    "CheckpointInfo",
    "RunStoreMismatchError",
    "LineageGraph",
    "LineageRecord",
    "ElementSegment",
    "Ancestry",
    "ElementOrigin",
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
    "FusionPolicy",
    "NoFusion",
    "IROp",
    "IRSource",
    "IRMap",
    "IRAccumulate",
    "IRSink",
    "IRCheckpoint",
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
