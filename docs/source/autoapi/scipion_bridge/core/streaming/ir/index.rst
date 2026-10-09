scipion_bridge.core.streaming.ir
================================

.. py:module:: scipion_bridge.core.streaming.ir

.. autoapi-nested-parse::

   Streaming Intermediate Representation (IR) primitives.

   The IR is a backend-agnostic DAG representation of the streaming pipeline.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.ir.Tagged
   scipion_bridge.core.streaming.ir.Keyed
   scipion_bridge.core.streaming.ir.IROp
   scipion_bridge.core.streaming.ir.IRSource
   scipion_bridge.core.streaming.ir.IRMap
   scipion_bridge.core.streaming.ir.IRAccumulate
   scipion_bridge.core.streaming.ir.IRSink
   scipion_bridge.core.streaming.ir.IRDemux


Functions
---------

.. autoapisummary::

   scipion_bridge.core.streaming.ir.clone_ir


Module Contents
---------------

.. py:class:: Tagged

   Item arriving on a stage with several inputs, tagged with its input port.

   The port is the position of the sending stage among the upstream stages
   of the receiving stage.


   .. py:attribute:: port
      :type:  int


   .. py:attribute:: item
      :type:  Any


.. py:class:: Keyed

   Bases: :py:obj:`NamedTuple`


   Result of a ``group_by`` pipeline, together with the key of its group.


   .. py:attribute:: key
      :type:  Any


   .. py:attribute:: value
      :type:  Any


.. py:class:: IROp

   Base class for all IR primitives with DAG edge management.


   .. py:attribute:: downstream
      :type:  List[IROp]
      :value: []



   .. py:attribute:: upstream
      :type:  List[IROp]
      :value: []



   .. py:method:: add_downstream(child: IROp) -> None

      Wire a downstream edge and reciprocal upstream edge.



.. py:class:: IRSource

   Bases: :py:obj:`IROp`


   Ingestion entry point.


   .. py:attribute:: name
      :type:  str
      :value: ''



.. py:class:: IRMap

   Bases: :py:obj:`IROp`


   Stateless 1:1 batch/element transformation.


   .. py:attribute:: func
      :type:  Callable[[Any], Any]


   .. py:attribute:: name
      :type:  Optional[str]
      :value: None



   .. py:attribute:: compute
      :type:  Optional[scipion_bridge.core.environment.compute.ComputeAssignment]
      :value: None



.. py:class:: IRAccumulate

   Bases: :py:obj:`IROp`


   Stateful stream accumulation primitive.

   Maintains internal state across incoming items and flushes,
   emitting zero or more output items downstream.


   .. py:attribute:: accumulate_fn
      :type:  Callable[[Any, Any], Tuple[Any, List[Any]]]


   .. py:attribute:: initial_state_fn
      :type:  Callable[[], Any]


   .. py:attribute:: flush_fn
      :type:  Optional[Callable[[Any], Tuple[Any, List[Any]]]]
      :value: None



   .. py:attribute:: name
      :type:  str
      :value: 'accumulate'



   .. py:attribute:: tag_inputs
      :type:  bool
      :value: False



.. py:class:: IRSink

   Bases: :py:obj:`IROp`


   Terminal/Checkpoint node delegating to an async SinkWriter.


   .. py:attribute:: writer
      :type:  Optional[scipion_bridge.core.streaming.sink_writer.SinkWriter]
      :value: None



.. py:class:: IRDemux

   Bases: :py:obj:`IROp`


   Routing of items into a child pipeline per key (``group_by``).

   ``template`` is the exit node of the child pipeline, lowered on its own; it
   is not wired into the enclosing DAG. Its only source is ``source_name``.
   The backend compiles a clone of it for every new key and emits the
   results of each child as ``Keyed(key, result)``.


   .. py:attribute:: key_fn
      :type:  Callable[[Any], Any]


   .. py:attribute:: template
      :type:  Optional[IROp]
      :value: None



   .. py:attribute:: source_name
      :type:  str
      :value: ''



   .. py:attribute:: max_keys
      :type:  Optional[int]
      :value: None



   .. py:attribute:: name
      :type:  str
      :value: 'group_by'



.. py:function:: clone_ir(sinks: Sequence[IROp]) -> List[IROp]

   Copy the IR DAG reachable upstream from ``sinks``.

   Every node is copied with fresh edges, so the copy can be wired (e.g. to
   a new sink) and compiled without touching the original. Upstream edges
   keep their order, which preserves the input ports of multi-input stages.
   Functions, writers and other attributes are shared, not copied.

   :returns: The copies of ``sinks``, in the same order.


