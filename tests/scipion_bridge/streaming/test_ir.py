from scipion_bridge.core.streaming.ops import Source
from scipion_bridge.core.streaming.sink import Sink
from scipion_bridge.core.streaming.sink_writer import CallbackSinkWriter
from scipion_bridge.core.streaming.ir import IRSource, IRMap, IRSink, IRAccumulate
from scipion_bridge.core.streaming.node import lower


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


def test_checkpoint_dag_lowering():
    w_ckpt = CallbackSinkWriter(lambda x: None)
    w_terminal = CallbackSinkWriter(lambda x: None)

    source = Source("input")
    stage1 = source.map_batch(lambda x: x * 2)
    stage1.checkpoint(w_ckpt)
    stage2 = stage1.map_batch(lambda x: x + 10)
    terminal_sink = stage2.write_to(w_terminal)

    # Lowering from terminal_sink and finding all sinks
    # Note: checkpoint attaches a Sink to stage1
    checkpoint_sink = [n for n in stage1.downstream if isinstance(n, Sink)][0]

    ir_sinks = lower([terminal_sink, checkpoint_sink])
    assert len(ir_sinks) == 2

    # Verify stage1 IR node has 2 downstream nodes (IRSink for checkpoint, IRMap for stage2)
    stage2_ir_map = ir_sinks[0].upstream[0]
    stage1_ir_map = stage2_ir_map.upstream[0]
    assert len(stage1_ir_map.downstream) == 2


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

    assert isinstance(element_ir, IRAccumulate)
    assert element_ir.flush_fn is None
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
