scipion_bridge.core.streaming.element_mapper
============================================

.. py:module:: scipion_bridge.core.streaming.element_mapper

.. autoapi-nested-parse::

   Parallel per-element processing of collections (``Op.map_element``).

   For every incoming collection ``col`` the stage runs::

       for i in range(len(col)):
           col[i] = func(col[i])

   with the loop distributed over a thread pool, and forwards the same, modified
   collection. Any collection with a fixed length and random access works
   (``list``, ``np.ndarray``, ``Set``, ...).

   The stage is a stateless map: :class:`ElementMapper` only carries the id of
   its pools, which cannot be pickled. Every process running the stage (e.g. a
   Ray task worker or a long-running executor) creates the pools on its first
   collection and keeps them in a cache of the process, reused for every later
   collection. With
   ``executor="process"`` the loop still runs in the thread pool and ``func`` is
   called in a process pool, started with ``spawn`` by default since forking a
   process that has initialized CUDA or JAX corrupts the child.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.streaming.element_mapper.Executor
   scipion_bridge.core.streaming.element_mapper.StartMethod
   scipion_bridge.core.streaming.element_mapper.Workers


Classes
-------

.. autoapisummary::

   scipion_bridge.core.streaming.element_mapper.ElementMapConfig
   scipion_bridge.core.streaming.element_mapper.ElementMapper


Module Contents
---------------

.. py:data:: Executor

.. py:data:: StartMethod

.. py:data:: Workers

.. py:class:: ElementMapConfig

   Execution options of ``Op.map_element``.

   .. attribute:: workers

      Number of parallel workers. ``"auto"`` uses one worker per
      element of the collection, capped at the CPUs available to the
      process (the reserved cores if the backend reserved some); after
      ``.chunk(n)`` this is ``min(n, CPUs)``.

   .. attribute:: executor

      ``"thread"`` runs ``func`` in the thread pool;
      ``"process"`` runs it in a process pool (for pure-Python work that
      holds the GIL).

   .. attribute:: start_method

      Start method of the process pool; only valid with
      ``executor="process"``. Defaults to ``"spawn"``.

   .. attribute:: chunksize

      Number of consecutive indices per pool task. Defaults to 1
      for ``workers="auto"``, else ``ceil(len(col) / (4 * workers))``.


   .. py:attribute:: workers
      :type:  Workers
      :value: 'auto'



   .. py:attribute:: executor
      :type:  Executor
      :value: 'thread'



   .. py:attribute:: start_method
      :type:  Optional[StartMethod]
      :value: None



   .. py:attribute:: chunksize
      :type:  Optional[int]
      :value: None



.. py:class:: ElementMapper

   The function of a map_element stage: ``col -> col``, modified in place.

   .. attribute:: func

      Function applied to each element.

   .. attribute:: config

      Execution options.

   .. attribute:: stage

      Id of the stage's pools in the cache of each process.


   .. py:attribute:: func
      :type:  Callable[[Any], Any]


   .. py:attribute:: config
      :type:  ElementMapConfig


   .. py:attribute:: stage
      :type:  str
      :value: '00000000000000000000000000000000'



