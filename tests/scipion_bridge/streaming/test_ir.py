from scipion_bridge.core.streaming.element_mapper import ElementMapper
from scipion_bridge.core.streaming import Source
from scipion_bridge.core.streaming import Sink
from scipion_bridge.core.streaming import CallbackSinkWriter
from scipion_bridge.core.streaming import IRSource, IRMap, IRSink, IRAccumulate
from scipion_bridge.core.streaming import IRCheckpoint, NoFusion
from scipion_bridge.core.streaming import stage_labels, topological_order
from scipion_bridge.core.streaming.ir import bypass, describe
from scipion_bridge.core.streaming import lower


def test_linear_lowering():
    results = []
    writer = CallbackSinkWriter(lambda x: results.append(x))

    source = Source("input")
    mapped = source.map_batch(lambda x: x * 2)
    sink_node = mapped.write_to(writer)

    ir_sinks = lower([sink_node])
    assert len(ir_sinks) == 1

    sink_ir = ir_sinks[0]
    assert isinstance(sink_ir, IRSink)
    assert sink_ir.writer is writer
    assert len(sink_ir.upstream) == 1

    map_ir = sink_ir.upstream[0]
    assert isinstance(map_ir, IRMap)
    assert map_ir.downstream == [sink_ir]
    assert len(map_ir.upstream) == 1

    src_ir = map_ir.upstream[0]
    assert isinstance(src_ir, IRSource)
    assert src_ir.name == "input"
    assert src_ir.downstream == [map_ir]


def test_branching_dag_lowering():
    w1 = CallbackSinkWriter(lambda x: None)
    w2 = CallbackSinkWriter(lambda x: None)

    source = Source("branch_src")
    branch1 = source.map_batch(lambda x: x + 1).write_to(w1)
    branch2 = source.map_batch(lambda x: x + 2).write_to(w2)

    ir_sinks = lower([branch1, branch2])
    assert len(ir_sinks) == 2

    # Verify both branches share the exact same IRSource instance
    src_ir_1 = ir_sinks[0].upstream[0].upstream[0]
    src_ir_2 = ir_sinks[1].upstream[0].upstream[0]
    assert src_ir_1 is src_ir_2
    assert isinstance(src_ir_1, IRSource)
    assert len(src_ir_1.downstream) == 2


def test_checkpoints_and_taps_are_stages_in_the_stream():
    sink = (
        Source("input")
        .map_batch(_double)
        .checkpoint("doubled")
        .tap()
        .write_to(CallbackSinkWriter(_discard))
    )

    (ir_sink,) = lower([sink], fusion=NoFusion())

    tap = ir_sink.upstream[0]
    checkpoint = tap.upstream[0]
    assert isinstance(tap, IRCheckpoint) and not tap.replay
    assert isinstance(checkpoint, IRCheckpoint) and checkpoint.replay
    assert [describe(n) for n in (checkpoint, tap)] == ["checkpoint(doubled)", "tap"]


def test_bypassed_nodes_leave_their_input_to_their_downstream():
    sink = (
        Source("input")
        .map_batch(_double)
        .checkpoint()
        .map_batch(_double)
        .write_to(CallbackSinkWriter(_discard))
    )
    (ir_sink,) = lower([sink], fusion=NoFusion())
    labels = stage_labels(topological_order([ir_sink]))

    (copy,), originals = bypass([ir_sink], lambda n: isinstance(n, IRCheckpoint))

    nodes = topological_order([copy])
    assert [labels[originals[n]] for n in nodes] == [
        "0:source(input)",
        "1:map(_double)",
        "3:map(_double)",
        "4:sink(CallbackSinkWriter)",
    ]
    assert nodes[2].upstream == [nodes[1]]
    # The original is untouched.
    assert isinstance(ir_sink.upstream[0].upstream[0], IRCheckpoint)


def _double(x):
    return x * 2


def _discard(x):
    pass


def test_chunk_lowering():
    writer = CallbackSinkWriter(lambda x: None)
    source = Source("input")
    chunked = source.chunk(10, drop_last=True)
    sink_node = chunked.write_to(writer)

    ir_sinks = lower([sink_node])
    assert len(ir_sinks) == 1

    sink_ir = ir_sinks[0]
    assert isinstance(sink_ir, IRSink)
    assert len(sink_ir.upstream) == 1

    accum_ir = sink_ir.upstream[0]
    assert isinstance(accum_ir, IRAccumulate)
    assert accum_ir.downstream == [sink_ir]
    assert len(accum_ir.upstream) == 1
    assert accum_ir.flush_fn is not None

    src_ir = accum_ir.upstream[0]
    assert isinstance(src_ir, IRSource)
    assert src_ir.downstream == [accum_ir]
    assert accum_ir.name == "chunk(10)"


def test_map_element_lowering():
    def preprocess(x):
        return x

    source = Source("input")
    sink_node = source.map_element(preprocess, workers=2).write_to(
        CallbackSinkWriter(lambda x: None),
    )

    (sink_ir,) = lower([sink_node])
    element_ir = sink_ir.upstream[0]

    # A stateless map; the worker pools are cached per process.
    assert isinstance(element_ir, IRMap)
    assert isinstance(element_ir.func, ElementMapper)
    assert element_ir.func.config.workers == 2
    assert element_ir.name == f"map_element({preprocess.__qualname__})"


def test_same_named_sources_lower_to_one_ir_source():
    w1 = CallbackSinkWriter(lambda x: None)
    w2 = CallbackSinkWriter(lambda x: None)

    # Two distinct nodes with the same name, as created by repeated accesses
    # to the same protocol input.
    branch1 = Source("x").map_batch(lambda x: x + 1).write_to(w1)
    branch2 = Source("x").map_batch(lambda x: x + 2).write_to(w2)

    ir_sinks = lower([branch1, branch2])

    src_ir_1 = ir_sinks[0].upstream[0].upstream[0]
    src_ir_2 = ir_sinks[1].upstream[0].upstream[0]
    assert src_ir_1 is src_ir_2
    assert isinstance(src_ir_1, IRSource)
    assert len(src_ir_1.downstream) == 2


def test_map_lowering_is_named():
    def double(x):
        return x * 2

    sink_node = (
        Source("input")
        .map_batch(double)
        .write_to(
            CallbackSinkWriter(lambda x: None),
        )
    )

    map_ir = lower([sink_node])[0].upstream[0]
    assert isinstance(map_ir, IRMap)
    assert map_ir.name.startswith("map(") and "double" in map_ir.name


def test_collect_and_flatten_lowering():
    sink_node = (
        Source("input")
        .collect(7)
        .map_batch(lambda s: [s])
        .flatten()
        .write_to(CallbackSinkWriter(lambda x: None))
    )

    flatten_ir = lower([sink_node])[0].upstream[0]
    assert isinstance(flatten_ir, IRAccumulate)
    assert flatten_ir.name == "flatten"
    assert flatten_ir.flush_fn is None

    collect_ir = flatten_ir.upstream[0].upstream[0]
    assert isinstance(collect_ir, IRAccumulate)
    assert collect_ir.name == "collect(7)"
    assert collect_ir.flush_fn is not None
