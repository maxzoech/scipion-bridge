scipion_bridge.core.streaming.node
==================================

.. py:module:: scipion_bridge.core.streaming.node

.. autoapi-nested-parse::

   Base class for all nodes in the streaming computational graph.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.streaming.node.FLUSH


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.node.LoweringContext
   scipion_bridge.core.streaming.node.FlushSignal
   scipion_bridge.core.streaming.node.Node


Functions
---------

.. autoapisummary::

   scipion_bridge.core.streaming.node.replace_node
   scipion_bridge.core.streaming.node.lower


Module Contents
---------------

.. py:class:: LoweringContext

   Context coordinator that manages memoization and DAG wiring during lowering.


   .. py:attribute:: memo
      :type:  Dict[int, scipion_bridge.core.streaming.ir.IROp]


   .. py:attribute:: sources
      :type:  Dict[str, scipion_bridge.core.streaming.ir.IRSource]


   .. py:method:: lower_node(node: Node) -> scipion_bridge.core.streaming.ir.IROp

      Recursively lowers a Node using polymorphism and wires DAG dependencies.



.. py:class:: FlushSignal

   Sentinel object emitted through the stream graph to trigger state flushing.


.. py:data:: FLUSH

.. py:class:: Node(upstream: Optional[List[Node]] = None)

   Base class for all nodes in the streaming computational graph.


   .. py:attribute:: upstream
      :type:  List[Node]


   .. py:attribute:: downstream
      :type:  List[Node]
      :value: []



   .. py:method:: lower(ctx: LoweringContext) -> scipion_bridge.core.streaming.ir.IROp
      :abstractmethod:


      Polymorphically lower this surface Node to its low-level IR representation.



.. py:function:: replace_node(old: Node, new: Node) -> None

   Rewire every downstream consumer of ``old`` to consume ``new`` instead.

   The position of ``old`` in each consumer's ``upstream`` list is preserved, so
   the argument order of multi-input nodes does not change. ``old`` is left
   without downstream nodes.


.. py:function:: lower(nodes: List[Node]) -> List[scipion_bridge.core.streaming.ir.IROp]

   Lower one or more DAG root/sink nodes into lowered IR nodes.

   :returns: List of lowered IROp nodes corresponding to the input nodes.


