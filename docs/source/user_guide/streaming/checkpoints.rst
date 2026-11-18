Checkpoints
===========

.. currentmodule:: scipion_bridge.core.streaming

A checkpoint keeps the items passing a point of the stream in the run store
of the pipeline, by their item ids, while the stream goes on. The kept items
can be inspected while the pipeline runs, next to their lineage. They are the
points a rerun will continue from.

.. contents:: On this page
   :local:
   :depth: 2

Keeping items
-------------

.. code-block:: python

    embeddings = chunks.map(self._forward).checkpoint("embeddings")
    classes = embeddings.map(self._classify).tap("classes")

* ``checkpoint(name)`` keeps the items as a point to rerun from. Put one after
  expensive work.
* ``tap(name)`` keeps them only to inspect them.

Both pass every item on unchanged, at once. The items are written in the
background, so a checkpoint does not delay the stream. With a slow disk it
waits before taking further items (at most 8 writes at once), instead of
filling its memory. ``name`` appears in the label of the stage, e.g.
``3:checkpoint(embeddings)``.

Without a run store (``RayBackend(run_store=...)``, or ``--run-dir`` of the
runner), checkpoints and taps are left out of the pipeline and cost nothing.
The other stages keep their labels.

The outputs of protocols
^^^^^^^^^^^^^^^^^^^^^^^^

Every protocol ends with a checkpoint of its validated outputs, named after
the protocol (``checkpoint(Classify2D)``). A chain of protocols keeps the
outputs of each. With ``--run-dir`` the runner keeps them all;
``--no-checkpoint-outputs`` (``checkpoint_outputs=False``) leaves them out
and keeps only the checkpoints the protocols place themselves.

What the run store keeps
------------------------

For every checkpoint and tap the run directory holds the items, one file per
item named by its id, and a commit per item in the order the items passed.
A checkpoint is *complete* once the end of the stream (FLUSH) passed it: it
kept every item. ``FileRunStore.read_checkpoints()`` lists them,
``read_item(label, item_id)`` reads an item back.

The lineage viewer (:doc:`/user_guide/backends/ray`, "The lineage of a
run") lists the checkpoints of a run with their items. For a kept item it
shows its structure: the fields of its Structs and Sets, small values, and
thumbnails of 2D images such as the representatives of classes. Sets are not
read for it: their rows may reference input files.

Items are kept as pickles, so reading them needs the classes of the items
(e.g. the protocol's package) where they are read.

Rows read from input files (e.g. the pixels of particles read from their
stacks) are kept as references to the files, not as bytes, if the process of
the checkpoint can read the files unchanged: reading the item reads them
from the files again. Rows computed in the pipeline are kept as bytes. A
checkpoint after a map computing pixels (resampling, CTF correction) keeps
all of them: for particles of 300×300 float32 pixels, 360 kB per particle,
or 36 GB for 100,000 particles. Put checkpoints before such maps where they
are cheap to compute again. If an input file changes after a checkpoint kept
its rows, reading them fails (``StaleReferenceError``).

.. _stages-that-depend-on-all-inputs:

Stages that depend on all their inputs
--------------------------------------

.. warning::

   ``collect``, ``scan``, the latest side of ``combine_latest`` and custom
   ``accumulate`` stages compute from everything they received. Their state
   is not kept: after a crash, they need all their inputs again. Put a
   ``checkpoint()`` in front of a long ``collect`` (preferably in front of
   cheap maps computing pixels, so that it keeps references instead of
   pixels), so that its inputs need not be computed again.

.. warning::

   With several loaders, and wherever streams meet, items arrive in a
   different order in every run. Which rows share a chunk, and which rows
   ``collect(n)`` gathers (the first ``n`` rows to *arrive*), differ between
   runs. Do not use ``collect`` to select training and validation sets, or
   any split a model is fitted or evaluated on: select those rows by their
   values or element ids (e.g. a hash of the element id) in a map, so the
   split is the same in every run.

Reruns
------

Running a pipeline again in the run store of an earlier run continues it.
Today a sink skips only the items it receives again *with the same item
ids*. Item ids follow the order items arrive in, so this takes the same
order as the earlier run: one loader, and no stage that depends on arrival
order before the sink (``scan``, ``collect``, a ``group_by`` with several
workers). With several loaders, items are chunked differently in every run,
their ids differ, and the sinks skip nothing (see :doc:`sinks`, "Commits and
reruns").

Reruns that skip by *input rows* instead, replaying checkpoints so that a
rerun computes only the rows none of its checkpoints and sinks kept, are
planned next. They do not need the same order.
