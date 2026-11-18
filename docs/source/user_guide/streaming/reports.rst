Pipeline Reports
================

.. currentmodule:: scipion_bridge.core.streaming.report

A pipeline report draws the graph of a pipeline: every stage, the protocol
it belongs to, the stages inside each ``group_by``, and what the library
knows about them, such as the stats of a run. Use it to check what a chain of
protocols builds before running it, and to find the slow or starved stages
afterwards.

.. contents:: On this page
   :local:
   :depth: 2

Drawing a protocol without running it
-------------------------------------

Every protocol launched with ``RayPipelineRunner`` accepts ``--explain``. It
builds the pipeline, prints its stages and writes the report, without starting
Ray or reading any input; inputs and parameters are optional:

.. code-block:: console

    $ python -m my_protocols --explain report.html
    Pipeline: 33 stages, 2 compute groups
      0:source(particles)
      1:chunk(256)
      2:map(RayFoundationModelInference._raw_input)  [RayFoundationModelInference]
      ...
    Pipeline report: /path/report.html (graph: /path/report.dot)

Open ``report.html`` in a browser. ``report.dot`` holds the graph in
Graphviz' DOT language.

``--report report.html`` writes the report after a run instead, with the
stats of every stage. From Python, ``runner.report()`` returns the report.

Reading the report
------------------

* Stages are coloured by their compute group, the protocol whose
  ``@B.resources`` they run with; stages without resources are grey. Stage
  ids are those of ``Pipeline.stats()``.
* A ``group_by`` is drawn as a folder, its template as a dashed frame below
  it. Dashed edges lead into the template and from it to where the
  ``group_by`` sends its results: the path the items of every key take.
* Clicking a stage or an edge lists its annotations, per layer: what a
  feature of the library knows about it. The ``stats`` layer holds the stats
  of a run (the children of a ``group_by`` merged) and the metrics derived
  from them: ``throughput`` (items/s), ``s_per_item``, ``busy``,
  ``utilization`` and ``wait_mean_s`` (see :doc:`pipelines`). Its panel
  *Throughput* lists the stages, the most utilized first, and its warnings
  name the bottleneck.
* The ``placement`` layer shows where the executors of every protocol run:
  the panel *Pools* lists them, with the GPUs each reserves and the
  protocols of earlier phases it waits for; one panel per phase draws the
  GPUs and the protocols on each, and the warnings name protocols that share
  a GPU. After a run, the GPUs are named by the devices the executors ran
  on, as ``nvidia-smi`` numbers them; before, by their index in the plan.
  ``--placement ray`` shows the plan without reservations. ``--explain``
  plans for the GPUs of this machine that ``CUDA_VISIBLE_DEVICES`` shows;
  ``--explain-gpus 4,4`` plans for another cluster, e.g. two nodes of four
  GPUs.
* The ``provenance`` layer shows which columns of the items are still rows
  of the input files (references), as sampled from the first 3 items in and
  out of every stage. Per stage, ``reference_bytes``, ``cached_bytes`` (read
  references) and ``owned_bytes`` per item, and ``reference_fraction``;
  ``read_here`` names the paths whose files the stage reads, and
  ``computed_here`` the paths that arrive as references and leave as owned
  columns. Edges into such a stage are drawn red, and a warning gives the
  bytes per item a spill after it writes, and whether the values were only
  copied (then cast the input with ``.astype(T)`` and write the new fields
  into it, which keeps the reference; see :doc:`sources_and_maps`). Telling
  copies apart hashes the copied columns of the first sampled items, so it
  only runs when Python runs without ``-O`` (``__debug__``). The panel
  *Key paths* lists the state of every referenced path at every stage
  (``ref``, ``ref*`` read, ``own``, ``C`` lost here), and *Spill estimate*
  the bytes per item spilled at every phase boundary. A lost reference
  costs only bytes when spilling, never correctness.
* *Colour by* colours the stages by any numeric annotation of a layer, e.g.
  ``busy``, which shows the bottleneck, ``wait_max_s``, which shows where
  items queue, or ``throughput``. The legend gives the range of the values.

The page has no external resources, so it opens on machines without network
access.

Layout with Graphviz
--------------------

The graph is laid out with Graphviz if its ``dot`` program is installed,
which avoids crossing edges better than the built-in layout used otherwise.
Graphviz is a program, not a Python package:

.. code-block:: console

    $ conda install graphviz      # or: apt install graphviz

The footer of the page names the layout used.

From Python
-----------

.. code-block:: python

    from pathlib import Path

    from scipion_bridge.backend.ray import RayBackend
    from scipion_bridge.core.streaming import Pipeline, Source, lower
    from scipion_bridge.core.streaming.report import report_of, render_html, render_text

    done = Source("particles").chunk(256).map(embed).sink(save)

    # The graph alone, without compiling it.
    report = report_of(lower([done]))
    print(render_text(report))

    # A compiled pipeline adds the placement of its pools; compiling starts
    # no actors, so this costs nothing on the cluster.
    print(render_text(Pipeline.from_sink(done, backend=RayBackend()).report()))

    # A pipeline that ran adds the stats of its stages.
    with Pipeline.from_sink(done, backend=RayBackend()) as pipeline:
        ...
    Path("report.html").write_text(render_html(pipeline.report()))

``render_dot`` returns the DOT text and ``render_svg`` the drawing alone
(``layout="graphviz"``, ``"builtin"`` or ``"auto"``). ``report.to_json()``
is a stable format for other tools.
