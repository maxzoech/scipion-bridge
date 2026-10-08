Quickstart
==========

This page takes you from describing your data to running a streaming protocol
on Ray, in five steps. Each step introduces one part of the library and links to
the guide that covers it in depth.

.. code-block:: python

    import numpy as np
    import scipion_bridge as B

All public names live in the ``scipion_bridge`` namespace, by convention
imported as ``B``.

1. Describe one element with a ``Struct``
-----------------------------------------

A :class:`~scipion_bridge.core.struct.struct.Struct` describes the layout of a
single element, much like a ``dataclass``. Scalar annotations (``float``,
``int``, ``bool``) become one-element arrays, and nested structs become nested
groups of columns. ``B.Array`` declares an array field with a fixed rank. Use
``None`` for dimensions whose size may vary from element to element.

.. code-block:: python

    class Coordinate(B.Struct):
        x: float
        y: float


    class Particle(B.Struct):
        pixels: B.Array[np.float32] = B.Array(shape=(None, None))
        coordinate: Coordinate
        sampling_rate: float


    p = Particle(
        pixels=np.zeros((64, 64), np.float32),
        coordinate=Coordinate(x=10.0, y=20.0),
        sampling_rate=1.35,
    )
    p.coordinate.x    # 10.0
    p.pixels.shape    # (64, 64)

``print_schema()`` shows the physical layout that the struct describes:

.. code-block:: text

    >>> Particle.print_schema()
    /
    ├── pixels: Array[float32], shape: [None, None]
    ├── coordinate (struct)
    │   ├── x: Array[float64], shape: [1]
    │   └── y: Array[float64], shape: [1]
    └── sampling_rate: Array[float64], shape: [1]

The library ships ready-made structs for single particle analysis in
``scipion_bridge.single_particle`` (``Particle``, ``CTF``, ``Coordinate``,
``Acquisition``, ``Class2D``).

2. Batch elements with a ``Set``
--------------------------------

A ``Set[T]`` stores many structs **column by column**. You can still treat it
like a list of objects, but you can also read a whole column at once as an
array:

.. code-block:: python

    particles = B.Set[Particle]([
        Particle(
            pixels=np.full((64, 64), i, np.float32),
            coordinate=Coordinate(x=float(i), y=0.0),
            sampling_rate=1.35,
        )
        for i in range(1_000)
    ])

    len(particles)                       # 1000
    particles[3].coordinate.x            # 3.0    (object view of one row)
    particles["coordinate"]["x"]         # array of shape (1000, 1)
    first_hundred = particles[:100]      # a view, no copy

Because the data is columnar, slicing, concatenation, and handing columns to
NumPy, PyTorch or JAX are cheap. See :doc:`../user_guide/structs`.

3. Build a stream
-----------------

Streams are built *declaratively*. Chaining operations does not run anything;
it records a graph. ``Source`` names an input of the graph:

.. code-block:: python

    from scipion_bridge.core.streaming import Pipeline, Source


    def normalize(batch: B.Set[Particle]) -> B.Set[Particle]:
        pixels = batch["pixels"]
        ...
        return batch


    particles_in = Source("particles")
    out = (
        particles_in
        .chunk(256)            # re-batch the incoming Sets into 256 elements
        .map(normalize)        # runs as its own pipeline stage
        .sink(print)           # terminal node
    )

Compile the graph into a ``Pipeline`` and push data into it. Leaving the
``with`` block *flushes* the pipeline. A flush pushes out partial chunks, drains
in-flight work, and finalizes the sinks:

.. code-block:: python

    with Pipeline.from_sink(out) as pipeline:
        pipeline.send(particles=particles[:600])
        pipeline.send(particles=particles[600:])
    pipeline.close()

Each ``map`` is a separate stage, and consecutive stages work on different
batches at the same time. Splitting work into several maps therefore overlaps
it, for example CPU preprocessing of one batch with the GPU forward pass of the
next. See :doc:`../user_guide/streaming`.

4. Wrap it in a ``Protocol``
----------------------------

A :class:`~scipion_bridge.core.protocol.protocol_base.Protocol` packages a
stream with declared inputs, parameters, and outputs. With these declarations
the same class can run as a Ray job, a command line tool, or a Scipion protocol
with a generated GUI form.

.. code-block:: python

    from scipion_bridge import single_particle as spa


    class Normalize(B.Protocol):
        # Inputs are streams. Use them like a Source.
        particles: B.Input[B.Set[spa.Particle]]

        # Fields are parameters. Read them with ``.value``.
        chunk_size: B.Field[int] = B.Field(default=256, label="Chunk size")

        def outputs(self):
            return {"particles": B.Set[spa.Particle]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._normalize)
            )

        def _normalize(self, batch):
            ...
            # The last stage returns a dict that matches outputs().
            return {"particles": batch}

5. Run it
---------

``RayPipelineRunner`` resolves each input from a file path into the declared
type (here: a STAR file into ``Set[Particle]``). It reads the input in chunks,
streams the chunks through the compiled pipeline, and prints per-stage
statistics at the end:

.. code-block:: python

    from pathlib import Path
    from scipion_bridge.backend.ray import RayPipelineRunner

    with RayPipelineRunner(Normalize(), parameters={"chunk_size": 512}) as runner:
        runner.run(particles=Path("particles.star"))

To turn the protocol into a command line program, call
``launch_as_terminal_application()``. It builds the ``argparse`` arguments from
the declared fields:

.. code-block:: python

    if __name__ == "__main__":
        RayPipelineRunner(Normalize()).launch_as_terminal_application()

.. code-block:: bash

    python normalize.py --particles particles.star --chunk-size 512

See :doc:`../user_guide/protocols` and :doc:`../user_guide/backends`.

Bonus: call an external program
-------------------------------

Most cryo-EM tools are command line programs that read and write files.
``@shell_command`` turns an empty Python function into a call to such a
program, and ``@proxify`` creates and cleans up the temporary files:

.. code-block:: python

    Latents = B.namedproxy("Latents", file_ext=".fmlatents")
    fm = B.Domain("foundation-models", ["fm"])


    @B.proxify
    @B.shell_command(domain=fm, name="inference", particles="input")
    def compute_latents(
        *,
        particles: B.ResolveProxy[B.ParticleStackProxy],
        weights: str,
        output: B.ResolveProxy[Latents] = B.Output(Latents),
    ):
        pass


    latents = compute_latents(particles=batch, weights="cryouni-b")
    # runs: fm inference --input /tmp/....star --weights cryouni-b --output /tmp/....fmlatents
    # and returns a Latents proxy; its file is deleted once `latents` is garbage collected.

See :doc:`../user_guide/shell_commands` and :doc:`../user_guide/proxies`.

Next steps
----------

* :doc:`../user_guide/overview`: the motivation and architecture.
* :doc:`../tutorials/2d_classification`: a complete, stateful streaming
  protocol.
* :doc:`../user_guide/sharp_bits`: common pitfalls.
