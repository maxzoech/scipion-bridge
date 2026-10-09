scipion_bridge.backend.ray.mailbox
==================================

.. py:module:: scipion_bridge.backend.ray.mailbox

.. autoapi-nested-parse::

   Buffer of the items waiting for a stage of a Ray pipeline.

   The mailbox decouples the stages: ``put`` returns as soon as an item is
   buffered, so an upstream stage does not wait while the stage is busy. Once
   ``buffer_size`` items wait, ``put`` waits for room: this is the backpressure
   that bounds the backlog of a pipeline. Waiting items stay in memory, either
   as they arrived or as references into the object store; beyond
   ``SpillPolicy.threshold`` of them, further items are written to a
   ``SpillStore`` and no reference to them is kept, so their memory is freed.
   The mailbox makes no Ray calls itself: references are awaited, which in a Ray
   actor fetches them.



Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.ray.mailbox.MailboxEntry


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.mailbox.Ref
   scipion_bridge.backend.ray.mailbox.Spilled
   scipion_bridge.backend.ray.mailbox.SpillPolicy
   scipion_bridge.backend.ray.mailbox.Mailbox


Module Contents
---------------

.. py:data:: MailboxEntry

.. py:class:: Ref

   Reference to an item in the object store, sent between stages.

   Ray resolves only object references passed directly as arguments, so a
   wrapped reference reaches the receiving stage without its data.


   .. py:attribute:: ref
      :type:  Any


.. py:class:: Spilled

   An item written to a spill store, or being written.


   .. py:attribute:: task
      :type:  asyncio.Task[Hashable]


   .. py:attribute:: store
      :type:  scipion_bridge.core.streaming.spill.SpillStore[Any]


.. py:class:: SpillPolicy

   Spill items to ``store`` once ``threshold`` items wait in memory.


   .. py:attribute:: threshold
      :type:  int


   .. py:attribute:: store
      :type:  scipion_bridge.core.streaming.spill.SpillStore[Any]


.. py:class:: Mailbox(stats: scipion_bridge.core.streaming.backend.StageStats, buffer_size: Optional[int], spill: Optional[SpillPolicy], recorder: scipion_bridge.backend.ray.profiling.Recorder = NullRecorder())

   FIFO buffer of the items waiting for a stage.

   Every item is held either in memory (as a reference into the object
   store) or in the spill store, decided when it arrives, so the order of the
   items is kept across both. FLUSH is never spilled.

   ``buffer_size`` counts items, whatever their size: in a Ray actor, every
   item in memory pins its object in the object store until it is consumed.


   .. py:method:: put(item: Any, port: int, barrier: Optional[_Barrier]) -> None
      :async:


      Buffer an item, waiting only while the mailbox is full.



   .. py:method:: get() -> MailboxEntry
      :async:


      Take the oldest waiting item; it is resolved with ``resolve``.



   .. py:method:: resolve(item: Any) -> Any
      :async:


      The item of an entry taken by ``get``, read from where it waited.



