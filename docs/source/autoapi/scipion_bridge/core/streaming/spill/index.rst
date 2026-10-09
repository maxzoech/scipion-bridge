scipion_bridge.core.streaming.spill
===================================

.. py:module:: scipion_bridge.core.streaming.spill

.. autoapi-nested-parse::

   Storage of the items that overflow the buffers of a streaming pipeline.

   A stage keeps a bounded number of the items waiting for it in memory; further
   items are handed to a ``SpillStore`` until the stage is ready for them. The
   store decides what to persist: the naive ``PickleSpillStore`` writes every
   item, while a store aware of the provenance of the data may write only the
   columns of a ``Set`` that changed and reference the data that already exists on
   disk.

   Items enter the transport between the stages of the Ray backend in a single
   place (``_deliver``), so that such a store can later also keep unchanged data
   out of the object store altogether.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.streaming.spill.H
   scipion_bridge.core.streaming.spill.SpillStoreFactory


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.spill.SpillStore
   scipion_bridge.core.streaming.spill.PickleSpillStore


Functions
---------

.. autoapisummary::

   scipion_bridge.core.streaming.spill.pickle_spill_store


Module Contents
---------------

.. py:data:: H

.. py:class:: SpillStore

   Bases: :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`H`\ ]


   Stores the items that overflow the buffer of a pipeline stage.

   Every item is stored once, read back once and then discarded. Handles are
   small and picklable; the items themselves never pass through them.


   .. py:method:: put(item: Any) -> H
      :abstractmethod:


      Store ``item`` and return the handle to read it back.



   .. py:method:: get(handle: H) -> Any
      :abstractmethod:


      Reconstruct the item stored under ``handle``.



   .. py:method:: discard(handle: H) -> None
      :abstractmethod:


      Free the storage of a consumed item.



.. py:data:: SpillStoreFactory

.. py:class:: PickleSpillStore(directory: pathlib.Path)

   Bases: :py:obj:`SpillStore`\ [\ :py:obj:`str`\ ]


   Pickles every item into a file of its own inside ``directory``.

   The directory is created on the first item, on the node of the stage.


   .. py:attribute:: directory


   .. py:method:: put(item: Any) -> str

      Store ``item`` and return the handle to read it back.



   .. py:method:: get(handle: str) -> Any

      Reconstruct the item stored under ``handle``.



   .. py:method:: discard(handle: str) -> None

      Free the storage of a consumed item.



.. py:function:: pickle_spill_store(root: pathlib.Path) -> SpillStoreFactory

   Factory of ``PickleSpillStore``s writing into a directory per stage under ``root``.


