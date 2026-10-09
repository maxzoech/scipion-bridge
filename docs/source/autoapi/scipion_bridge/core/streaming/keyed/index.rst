scipion_bridge.core.streaming.keyed
===================================

.. py:module:: scipion_bridge.core.streaming.keyed

.. autoapi-nested-parse::

   Pipelines of a ``group_by`` shared by several keys.

   A ``group_by`` runs the pipeline of its keys on a few shared copies (workers)
   rather than one copy per key, so that a fan-out to many keys does not start
   the stages of a pipeline for each of them. The items of a shared copy carry
   their key, as ``Keyed(key, item)``, through every stage:

   - Maps are stateless: they apply their function to the item and keep its key.
   - Accumulators keep a state per key, created with the first item of the key.
     Their functions only ever see the state and the items of one key.
   - A nested ``group_by`` routes by the key of its outer ``group_by`` and its own.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.keyed.KeyedMap
   scipion_bridge.core.streaming.keyed.KeyedAccumulator


Functions
---------

.. autoapisummary::

   scipion_bridge.core.streaming.keyed.share_keys


Module Contents
---------------

.. py:class:: KeyedMap

   Function of a map applied to the item of a ``Keyed`` item.


   .. py:attribute:: func
      :type:  Callable[[Any], Any]


.. py:class:: KeyedAccumulator

   Functions of an accumulator keeping a state per key.


   .. py:attribute:: accumulate_fn
      :type:  Callable[[Any, Any], Tuple[Any, List[Any]]]


   .. py:attribute:: initial_state_fn
      :type:  Callable[[], Any]


   .. py:attribute:: flush_fn
      :type:  Optional[Callable[[Any], Tuple[Any, List[Any]]]]


   .. py:method:: initial_state() -> Dict[Any, Any]


   .. py:method:: accumulate(states: Dict[Any, Any], item: Any) -> Tuple[Dict[Any, Any], List[scipion_bridge.core.streaming.ir.Keyed]]


   .. py:method:: flush(states: Dict[Any, Any]) -> Tuple[Dict[Any, Any], List[scipion_bridge.core.streaming.ir.Keyed]]

      Flush the state of every key, in the order the keys arrived.



.. py:function:: share_keys(exit_: scipion_bridge.core.streaming.ir.IROp) -> scipion_bridge.core.streaming.ir.IROp

   Copy of a ``group_by`` pipeline carrying ``Keyed`` items of many keys.

   :param exit_: Exit node of the pipeline (the template of an ``IRDemux``).

   :returns: The exit node of the copy.


