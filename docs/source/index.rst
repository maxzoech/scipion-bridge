Scipion Bridge
==============

**Scipion Bridge** is a Python library for writing high-performance, streaming
cryo-EM protocols that run on a laptop, inside `Scipion
<https://scipion.i2pc.es>`_, or distributed across a `Ray <https://www.ray.io>`_
cluster, all from the same code.

You describe *what* a protocol computes as a small declarative graph. The library
decides *how* to run it: how to batch the data, how to move it between workers,
how to convert it into the files an external program expects, and when to clean
those files up again.

.. code-block:: python

    import scipion_bridge as B
    from scipion_bridge import single_particle as spa


    class FoundationModelInference(B.Protocol):
        particles: B.Input[B.Set[spa.Particle]]
        chunk_size: B.Field[int] = B.Field(default=10_000)

        def outputs(self):
            return {"latents": B.Set[Latent]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._compute_latents)
                .map(self._to_output)
            )

The library has four building blocks. Each one can be used without the others:

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`Array types <user_guide/structs>`
     - ``Struct``, ``Set`` and ``Collection``: typed, columnar containers backed
       by NumPy, Awkward Array and Apache Arrow. They are cheap to slice, batch
       and serialize.
   * - :doc:`Streaming <user_guide/streaming>`
     - A declarative streaming API (``map``, ``chunk``, ``collect``,
       ``combine_latest``, ``group_by``, ...). It is lowered to a small
       intermediate representation that a backend such as Ray executes.
   * - :doc:`Type resolution <user_guide/type_resolution>` and
       :doc:`proxies <user_guide/proxies>`
     - A graph of converter functions that turns data into the type a function
       asks for. *Proxies* wrap temporary files and delete them automatically
       through reference counting.
   * - :doc:`Shell commands <user_guide/shell_commands>`
     - ``@shell_command`` makes a command line program look like a normal Python
       function.

Where to start
--------------

* **New to the library?** Read :doc:`getting_started/installation`, then work
  through :doc:`getting_started/quickstart`.
* **Want the big picture?** :doc:`user_guide/overview` covers why the library
  exists and how its layers fit together.
* **Porting a Scipion protocol?** :doc:`tutorials/2d_classification` builds a
  streaming 2D classification pipeline from start to finish.
* **Debugging something odd?** :doc:`user_guide/sharp_bits` lists the behaviors
  that most often surprise new users.
* **Extending the library?** The :doc:`developer_guide` explains the
  intermediate representation, how to write a backend, and the internal
  services.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/overview
   user_guide/structs
   user_guide/streaming
   user_guide/protocols
   user_guide/shell_commands
   user_guide/type_resolution
   user_guide/proxies
   user_guide/backends
   user_guide/sharp_bits

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/2d_classification
   examples

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   api
   developer_guide
   autoapi/index
