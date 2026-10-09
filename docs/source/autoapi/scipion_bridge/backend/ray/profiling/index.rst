scipion_bridge.backend.ray.profiling
====================================

.. py:module:: scipion_bridge.backend.ray.profiling

.. autoapi-nested-parse::

   Profiling of Ray pipelines: a live log and a trace of every worker.

   Every process of a profiled pipeline (stage actors, map tasks, executors of
   compute groups, the driver) records what it does as events through a
   ``Recorder``. Each event is logged right away on the logger
   ``scipion_bridge.profile``; Ray forwards the output of the workers to the
   driver. The events are also sent in batches to a ``ProfileCollector``, which
   appends them to a trace file in the Trace Event Format of Chrome, opened with
   https://ui.perfetto.dev or ``chrome://tracing``.

   The trace is written while the pipeline runs: the format allows the closing
   ``]`` to be missing, so the file can be opened at any time and reloaded to see
   newer events. It lags at most ``flush_interval_s`` behind the workers.



Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.ray.profiling.logger
   scipion_bridge.backend.ray.profiling.DEFAULT_FLUSH_INTERVAL_S


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.profiling.ProfileEvent
   scipion_bridge.backend.ray.profiling.ProcessInfo
   scipion_bridge.backend.ray.profiling.Recorder
   scipion_bridge.backend.ray.profiling.NullRecorder
   scipion_bridge.backend.ray.profiling.TraceRecorder
   scipion_bridge.backend.ray.profiling.ProfileConfig
   scipion_bridge.backend.ray.profiling.TraceWriter
   scipion_bridge.backend.ray.profiling.ProfileCollector


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.ray.profiling.make_recorder
   scipion_bridge.backend.ray.profiling.start_profile


Module Contents
---------------

.. py:data:: logger

.. py:data:: DEFAULT_FLUSH_INTERVAL_S
   :value: 1.0


.. py:class:: ProfileEvent

   Something a process of the pipeline did.

   .. attribute:: name

      What happened, e.g. ``process`` or ``executor_start``.

   .. attribute:: lane

      The part of the process it happened in, e.g. the ``compute``
      loop of a stage; a thread of the process in the trace.

   .. attribute:: ts_us

      Wall clock time of the start in microseconds, comparable
      across the processes of a node.

   .. attribute:: dur_us

      Duration in microseconds, or ``None`` for an instant event.

   .. attribute:: args

      Details, e.g. the sequence number of the item.

   .. attribute:: overlapping

      Whether spans of the lane may overlap, e.g. the calls of
      a map in flight; they are drawn on rows of their own.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: lane
      :type:  str


   .. py:attribute:: ts_us
      :type:  int


   .. py:attribute:: dur_us
      :type:  Optional[int]


   .. py:attribute:: args
      :type:  Mapping[str, Any]


   .. py:attribute:: overlapping
      :type:  bool
      :value: False



.. py:class:: ProcessInfo

   The process recording events: a stage actor, a task worker or the driver.


   .. py:attribute:: label
      :type:  str


   .. py:attribute:: pid
      :type:  int


   .. py:attribute:: node
      :type:  str


.. py:class:: Recorder

   Bases: :py:obj:`abc.ABC`


   Records the events of a process.


   .. py:method:: span(name: str, lane: str, *, level: int = logging.DEBUG, overlapping: bool = False, **args: Any) -> Iterator[Dict[str, Any]]

      Record the duration of the enclosed block, which may await.

      Yields the details of the event, to which the block may add. Spans
      that may overlap others of their lane, e.g. concurrent calls, set
      ``overlapping``.



   .. py:method:: instant(name: str, lane: str, *, level: int = logging.INFO, **args: Any) -> None

      Record an event without duration.



   .. py:method:: record(event: ProfileEvent, level: int) -> None
      :abstractmethod:


      Log an event and queue it for the collector.



   .. py:method:: flush() -> None
      :abstractmethod:

      :async:


      Send the queued events and wait until the collector wrote them.



   .. py:method:: flush_sync() -> None
      :abstractmethod:


      Like ``flush``, blocking; for processes without an event loop.



.. py:class:: NullRecorder

   Bases: :py:obj:`Recorder`


   Recorder of a pipeline that is not profiled: it records nothing.


   .. py:method:: record(event: ProfileEvent, level: int) -> None

      Log an event and queue it for the collector.



   .. py:method:: flush() -> None
      :async:


      Send the queued events and wait until the collector wrote them.



   .. py:method:: flush_sync() -> None

      Like ``flush``, blocking; for processes without an event loop.



.. py:class:: TraceRecorder(process: ProcessInfo, collector: Any, flush_interval_s: float, clock: Callable[[], float] = time.monotonic)

   Bases: :py:obj:`Recorder`


   Logs the events of a process and sends them to the collector in batches.

   A batch is sent once its oldest event waited ``flush_interval_s``, checked
   when an event is recorded. Ray delivers the batches of a process in order.
   Threads may record concurrently (e.g. in a threaded actor).


   .. py:method:: record(event: ProfileEvent, level: int) -> None

      Log an event and queue it for the collector.



   .. py:method:: flush() -> None
      :async:


      Send the queued events and wait until the collector wrote them.



   .. py:method:: flush_sync() -> None

      Like ``flush``, blocking; for processes without an event loop.



.. py:class:: ProfileConfig

   Profiling of a pipeline, shared by all of its processes.

   Nested pipelines (the children of a ``group_by``) receive the
   configuration of their parent and write into its trace.


   .. py:attribute:: path
      :type:  pathlib.Path


   .. py:attribute:: collector
      :type:  Any


   .. py:attribute:: log_level
      :type:  int
      :value: 20



   .. py:attribute:: flush_interval_s
      :type:  float
      :value: 1.0



   .. py:method:: recorder(label: str) -> TraceRecorder

      The recorder of the process ``label`` this is called in.



.. py:function:: make_recorder(profile: Optional[ProfileConfig], label: str) -> Recorder

   The recorder of the process ``label``; it records nothing without profiling.


.. py:class:: TraceWriter(path: pathlib.Path)

   Appends events to a trace file in the Trace Event Format of Chrome.

   The file is a JSON array whose closing ``]`` is written by ``close``.
   Every event is preceded by its separator, so the file followed by ``]``
   is valid JSON at any time; trace viewers accept the file without it.

   The processes are numbered by the writer, since the process ids of
   different nodes may collide; the process id is part of their name. Spans
   that may overlap are written as async events, which viewers draw on rows
   of their own.


   .. py:method:: write(process: ProcessInfo, events: List[ProfileEvent]) -> None

      Append the events of a process, and flush the file.



   .. py:method:: close() -> None

      Terminate the JSON array and close the file.



.. py:class:: ProfileCollector(path: pathlib.Path)

   Writes the events of every process of a pipeline to its trace file.


   .. py:method:: record(process: ProcessInfo, events: List[ProfileEvent]) -> None

      Append a batch of events of ``process`` to the trace.



   .. py:method:: close() -> None

      Complete the trace file.



.. py:function:: start_profile(path: pathlib.Path, log_level: int = logging.INFO, flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S) -> ProfileConfig

   Start the collector of a profiled pipeline, writing the trace to ``path``.

   The collector runs on the node of the caller, so that ``path`` is local
   to it.


