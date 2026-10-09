scipion_bridge.core.streaming.ir
================================

.. py:module:: scipion_bridge.core.streaming.ir

.. autoapi-nested-parse::

   Streaming Intermediate Representation (IR) primitives.

   The IR is a backend-agnostic DAG representation of the streaming pipeline.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.streaming.ir.DEFAULT_GROUP_BY_WORKERS
   scipion_bridge.core.streaming.ir.GroupByWorkers


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.ir.Tagged
   scipion_bridge.core.streaming.ir.Keyed
   scipion_bridge.core.streaming.ir.WorkersFrom
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


.. py:class:: WorkersFrom(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Number of workers of a ``group_by`` that its backend decides.


   .. py:attribute:: BACKEND


.. py:data:: DEFAULT_GROUP_BY_WORKERS
   :value: 4


.. py:data:: GroupByWorkers

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

   A pipeline shared by the keys of a ``group_by`` creates the state of a key
   with its first item, and flushes only the keys it has seen. ``flush_fn``
   must therefore emit nothing on the initial state.


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
   The backend runs clones of it, one per key or shared by several keys
   (``workers``), and emits the results of every key as ``Keyed(key, result)``.

   In a pipeline shared by several keys (see ``share_keys``), the items arrive
   as ``Keyed(outer, item)`` (``outer_keyed``); the results are emitted as
   ``Keyed(outer, Keyed(key, result))``.


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



   .. py:attribute:: workers
      :type:  GroupByWorkers


   .. py:attribute:: outer_keyed
      :type:  bool
      :value: False



   .. py:attribute:: name
      :type:  str
      :value: 'group_by'



.. py:function:: clone_ir(sinks: Sequence[IROp], changes: Callable[[IROp], Mapping[str, Any]] = _no_changes) -> List[IROp]

   Copy the IR DAG reachable upstream from ``sinks``.

   Every node is copied with fresh edges, so the copy can be wired (e.g. to
   a new sink) and compiled without touching the original. Upstream edges
   keep their order, which preserves the input ports of multi-input stages.
   Functions, writers and other attributes are shared, not copied.

   :param sinks: Exit nodes of the DAG to copy.
   :param changes: Fields to replace in the copy of a node.

   :returns: The copies of ``sinks``, in the same order.


