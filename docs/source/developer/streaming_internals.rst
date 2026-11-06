Streaming Internals
===================

The streaming module has two layers: the user-facing **operations** and a
minimal **intermediate representation** (IR). Operations are lowered to IR
nodes, and backends compile and run only the IR. This article describes
lowering, the IR primitives and the contract every backend fulfills, and then
walks through the two most common extensions: adding an operation and writing
a backend.

.. contents:: On this page
   :local:
   :depth: 2

Surface operations and lowering
-------------------------------

Every operation is a ``Node`` (``core/streaming/node.py``) with ``upstream``
and ``downstream`` edges and a ``lower(ctx)`` method. Chaining methods on an
``Op`` (``core/streaming/ops.py``) creates nodes and connects them with
``Op.op(node)``.

``lower(nodes)`` walks the graph upstream from the given sinks with a
``LoweringContext``:

* the context memoizes lowered nodes by identity, so shared subgraphs are
  lowered once;
* it deduplicates sources by name, which is what makes all accesses to a
  protocol input one source;
* each node lowers itself to one IR node, and the context wires the IR edges
  in the same order as the surface edges, so the position of an upstream node
  (its *port*) is preserved;
* a node that receives the same stream on two inputs is rejected, because
  ports would be ambiguous.

Only nodes reachable upstream from the given sinks are lowered. This is why a
``checkpoint`` sink must be passed to ``Pipeline.from_sink`` explicitly, and
why dangling branches are silently absent from a compiled pipeline.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface operation
     - Lowered to
   * - ``Source(name)``
     - ``IRSource(name)``, one per name
   * - ``map``, ``map_batch``
     - ``IRMap(func, compute)``
   * - ``map_element``
     - ``IRMap(ElementMapper(func, config))``. Worker pools are cached per
       process.
   * - ``chunk``, ``collect``, ``flatten``
     - ``IRAccumulate`` with the operation's accumulate, initial-state and
       flush functions
   * - ``scan``
     - ``IRAccumulate`` whose state holds the fold; ``on_flush`` becomes the
       flush function
   * - ``combine_latest``, ``merge``
     - ``IRAccumulate(tag_inputs=True)``
   * - ``group_by``
     - ``IRDemux(key_fn, template, source_name, max_keys, workers)``
   * - ``sink``, ``write_to``, ``checkpoint``
     - ``IRSink(writer)``

``Protocol.get_pipeline()`` adds two things before lowering: it assigns the
protocol's ``ComputeAssignment`` to every ``MapOp`` and ``MapElementOp`` of
its lineage (including those inside ``group_by`` pipelines) that has none, and
it appends a map that validates the outputs.

The IR primitives
-----------------

The IR (``core/streaming/ir.py``) consists of dataclasses with edge lists.

``IRSource(name)``
   An ingestion point. Items arrive through ``CompiledPipeline.send(name, item)``.

``IRMap(func, name, compute)``
   A stateless 1:1 transformation. ``compute`` is the ``ComputeAssignment``
   (resources plus the group of the protocol) that every call requires, or
   ``None``.

``IRAccumulate(accumulate_fn, initial_state_fn, flush_fn, tag_inputs)``
   A stateful N:M transformation. The backend keeps one ``state`` per stage:

   .. code-block:: python

       state = initial_state_fn()
       state, outputs = accumulate_fn(state, item)    # for every item
       state, outputs = flush_fn(state)               # on FLUSH (optional)

   With ``tag_inputs=True``, items are wrapped as ``Tagged(port, item)``,
   where ``port`` is the position of the sending stage among the node's
   upstream nodes.

``IRSink(writer)``
   A terminal node. It awaits ``writer.write(item)`` for every item and
   ``writer.finalize()`` on FLUSH.

``IRDemux(key_fn, template, source_name, max_keys, workers, outer_keyed)``
   Per-key routing for ``group_by``. ``template`` is the exit node of a
   separately lowered child graph whose only source is ``source_name``; it is
   not wired into the enclosing graph. With ``workers=None``, the backend
   compiles a copy of the template (``clone_ir``) for every new key. With
   ``workers=n``, the keys share up to ``n`` copies made by ``share_keys``
   (``core/streaming/keyed.py``): their items carry the key as
   ``Keyed(key, item)``, maps apply their function to the item
   (``KeyedMap``), and accumulators keep a state per key, created with the
   key's first item (``KeyedAccumulator``). An accumulator's ``flush_fn`` must
   therefore emit nothing on its initial state. ``WorkersFrom.BACKEND`` (the
   default) uses the backend's setting. Either way, the results are emitted
   as ``Keyed(key, result)``. A ``group_by`` nested in a shared copy has
   ``outer_keyed`` set: it receives ``Keyed(outer, item)`` and emits
   ``Keyed(outer, Keyed(key, result))``.

``clone_ir(sinks, changes)`` copies the graph upstream of ``sinks`` with
fresh edges, sharing functions and writers, and optionally replacing fields
per node. It is how backends instantiate the same template several times.

Every IR node has a keyword-only field ``stage_fp``: the fingerprint of the
computation of the stage. A protocol sets it on the nodes of its pipeline
(``Node.stage_fp``) when ``get_pipeline()`` seals it, and
``LoweringContext.lower_node`` copies it onto the lowered node. It is the
digest of the protocol's fingerprint (code, parameter values, sealed state;
``core/protocol/fingerprint.py``), the position of the node in the
``lineage`` of the pipeline and the kind of stage. Nodes of a protocol nested
in a chain keep the fingerprint of their own protocol; pipelines built
outside of protocols have ``None``. ``clone_ir`` and ``share_keys`` keep the
field, so the copies of a template share the identity of its stages.

The execution contract
----------------------

A backend must guarantee:

* **Per-stage ordering.** A stage processes items one at a time, in arrival
  order. Accumulator state and stage logic never run concurrently.
* **FLUSH barrier.** ``flush()`` returns only after all items sent before it
  have passed through every stage, every ``flush_fn`` has run, and every sink
  has been finalized. A stage with several upstream stages forwards FLUSH once
  it has received FLUSH from all of them.
* **Error propagation.** An exception in any stage fails the pipeline, and the
  driver's next ``send`` or ``flush`` raises it.
* **Backpressure.** Buffering between stages is bounded.

Maps may run concurrently with each other and with themselves, since they are
stateless; the order of their results must still be preserved on every edge.

Adding an operation
-------------------

Most new operations are stateful and lower to an ``IRAccumulate``, so no
backend needs to change. As an example, a ``count()`` operation that emits the
total number of elements of a stream of sets when the stream ends:

.. code-block:: python

    # core/streaming/ops.py
    def _count_initial_state() -> int:
        return 0


    def _count(state: int, item: Any) -> Tuple[int, List[int]]:
        match item:
            case Set():
                return (state + len(item), [])
            case _:
                raise TypeError(f"count() expected a Set, got '{type(item).__name__}'.")


    def _count_flush(state: int) -> Tuple[int, List[int]]:
        # Emits nothing on the initial state, as shared group_by pipelines require.
        match state:
            case 0:
                return (0, [])
            case total:
                return (0, [total])


    class CountOp(Op):
        """Operation node emitting the number of elements of the stream on FLUSH."""

        def __init__(self) -> None:
            super().__init__(upstream=None)

        def lower(self, ctx: LoweringContext) -> IROp:
            return IRAccumulate(
                accumulate_fn=_count,
                initial_state_fn=_count_initial_state,
                flush_fn=_count_flush,
                name="count",
            )

and a method on ``Op``:

.. code-block:: python

    def count(self) -> CountOp:
        """Emit the number of elements of a stream of sets when it ends."""
        return self.op(CountOp())

Points to observe:

* **Functions are pure.** ``accumulate_fn`` and ``flush_fn`` return a new
  state and a list of emissions; they do not keep state elsewhere. The backend
  owns the state.
* **Functions are picklable.** They are sent to the process that runs the
  stage. Module-level functions are the safest; closures work through
  cloudpickle, as ``chunk`` and ``collect`` show.
* **Flush emits nothing on the initial state**, so that the operation works
  inside a shared ``group_by`` pipeline.
* **Input types are checked**, with an error that names the operation, as the
  built-in operations do.
* **Several inputs** use ``tag_inputs=True`` and match on ``Tagged(port, item)``
  (see ``combine_latest`` and ``merge``); the method connects the other stream with
  ``other.op(node)``.
* **Validation of arguments** happens in the constructor, so that errors
  surface when the graph is built.

Test the functions in-process by lowering the operation and calling them (see
``tests/scipion_bridge/streaming/test_chunk_op.py``), and add one Ray test
for behavior that only exists across actors, such as flush ordering. Export
the new node from ``core/streaming/__init__.py`` and document the operation in
:doc:`/user_guide/streaming/index`.

An operation that cannot be expressed with the existing primitives requires a
new IR primitive, and with it an implementation in every backend. Prefer
compositions of existing primitives where possible.

Writing a backend
-----------------

A backend implements two interfaces from ``core/streaming/backend.py``:

.. code-block:: python

    class StreamingBackendProvider(abc.ABC):
        def compile(self, ir_sinks: list[IROp]) -> CompiledPipeline: ...


    class CompiledPipeline(abc.ABC):
        def send(self, source_name: str, value) -> None: ...
        def flush(self) -> None: ...
        def stats(self) -> dict[str, StageStats]: ...
        def close(self) -> None: ...

``compile`` receives the IR sinks; everything else is reachable upstream from
them. The backend is then provided as the ``streaming_backend`` service of a
container, or passed to ``Pipeline.from_sink(backend=...)``.

A minimal, single-threaded backend shows the shape of an implementation and is
useful for reasoning about new operations. It runs every node inline and
satisfies the contract trivially, except for ``group_by``:

.. code-block:: python

    import asyncio
    from typing import Any, Dict, List

    from scipion_bridge.core.streaming import (
        CompiledPipeline, IRAccumulate, IRDemux, IRMap, IROp, IRSink, IRSource,
        StageStats, StreamingBackendProvider, Tagged,
    )


    def _topological(sinks: List[IROp]) -> List[IROp]:
        order: List[IROp] = []
        seen: set = set()

        def visit(node: IROp) -> None:
            if node in seen:
                return
            seen.add(node)
            for up in node.upstream:
                visit(up)
            order.append(node)

        for sink in sinks:
            visit(sink)
        return order


    class InlinePipeline(CompiledPipeline):
        def __init__(self, sinks: List[IROp]) -> None:
            self.nodes = _topological(sinks)
            self.sources = {n.name: n for n in self.nodes if isinstance(n, IRSource)}
            self.states: Dict[IROp, Any] = {
                n: n.initial_state_fn() for n in self.nodes if isinstance(n, IRAccumulate)
            }

        def send(self, source_name: str, value: Any) -> None:
            self._emit(self.sources[source_name], [value])

        def _emit(self, node: IROp, items: List[Any]) -> None:
            for item in items:
                for down in node.downstream:
                    self._push(down, item, down.upstream.index(node))

        def _push(self, node: IROp, item: Any, port: int) -> None:
            match node:
                case IRMap(func=func):
                    self._emit(node, [func(item)])
                case IRAccumulate(tag_inputs=tag):
                    value = Tagged(port, item) if tag else item
                    self.states[node], out = node.accumulate_fn(self.states[node], value)
                    self._emit(node, out)
                case IRSink(writer=writer):
                    asyncio.run(writer.write(item))
                case IRDemux():
                    raise NotImplementedError("group_by")

        def flush(self) -> None:
            # Upstream nodes flush first, so their emissions reach downstream
            # accumulators before those flush.
            for node in self.nodes:
                match node:
                    case IRAccumulate(flush_fn=flush_fn) if flush_fn is not None:
                        self.states[node], out = flush_fn(self.states[node])
                        self._emit(node, out)
                    case IRSink(writer=writer):
                        asyncio.run(writer.finalize())
                    case _:
                        pass

        def stats(self) -> Dict[str, StageStats]:
            return {}


    class InlineBackend(StreamingBackendProvider):
        def compile(self, ir_sinks: List[IROp]) -> CompiledPipeline:
            return InlinePipeline(ir_sinks)

A real backend adds concurrency between stages (one worker per stateful
stage, maps on a pool), bounded queues between stages, FLUSH counting for
stages with several inputs, error propagation to the driver, ``group_by``
through ``clone_ir`` or ``share_keys``, and statistics. :doc:`ray_backend`
describes how the Ray backend does each of these.
