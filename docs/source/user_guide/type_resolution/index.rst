Type Resolution
===============

.. currentmodule:: scipion_bridge

The same particles can exist in many forms: an in-memory ``Set[Particle]``, a
STAR file with an MRC stack, a ``SetOfParticles`` in a Scipion project, a
NumPy array. **Type resolution** converts between these forms automatically.
Code states the type it needs, and the library finds a chain of converter
functions that produces it from the type it has.

.. code-block:: python

    @B.resolve_params
    def mean_defocus(particles: B.Resolve[B.Set[spa.Particle]]) -> float:
        return float(particles["ctf"]["defocus_u"].mean())

    mean_defocus(particles)                              # a Set, used as is
    mean_defocus(spa.ParticleStackProxy("data/particles")) # read from disk first

Type resolution is what connects the layers of the library: the Ray runner
uses it to read inputs from files, ``@B.proxify`` to write data to the files
programs expect, and the Scipion backend to convert from and to Scipion
objects.

.. contents:: On this page
   :local:
   :depth: 2

The resolution graph
--------------------

A **resolver** is a function that converts a value of one type into an
equivalent value of another type. All registered resolvers form a directed
graph: **nodes are types, and edges are resolvers**. Resolving a value to type
``T`` searches the **shortest path** from the value's type to ``T`` and
applies the resolvers along it, in order.

.. code-block:: text

      Set[Particle]            Path
            │   ▲                │
            ▼   │ (slice-aware)  ▼
       ParticleStackProxy ◄── StarfileProxy
            │
            ▼
         FuncParam  (the argument of a shell command)

Subclasses are handled automatically: if no resolver is registered for a
value's type, the search starts from the nearest base class in its MRO that
has one. If no path exists, ``TypeError`` is raised, naming both types.

When several paths have the same length, ties are broken in this order:
resolvers defined in the caller's local scope win, then paths with a
slice-aware step, then resolvers from more specific modules.

Resolving values
----------------

There are three ways to resolve a value.

**Directly**, with ``B.resolve``:

.. code-block:: python

    stack = B.resolve(particles, astype=spa.ParticleStackProxy)

**Through annotated parameters**, with ``@B.resolve_params`` and
``B.Resolve[T]``. Every argument of an annotated parameter is resolved on
each call; other parameters are passed through:

.. code-block:: python

    @B.resolve_params
    def f(stack: B.Resolve[spa.ParticleStackProxy], threshold: float): ...

**Through a precomputed resolver**, with ``B.find_resolver``. It runs the
graph search once and returns a ``ComposedResolver``, which can be called
repeatedly at the cost of the conversions alone:

.. code-block:: python

    to_set = B.find_resolver(Path, B.Set[spa.Particle])
    particles = to_set(Path("particles.star"))

The Ray runner precomputes a resolver for every protocol input when it is
created, so no graph search happens while data streams.

``@B.proxify``, described in :doc:`/user_guide/proxies`, is a fourth,
specialized form: it resolves arguments to proxy types and passes their paths
to a function.

Articles in this section
------------------------

.. toctree::
   :maxdepth: 1

   writing_resolvers
   chunked_resolution
   scoping

* :doc:`writing_resolvers` covers function and class resolvers, metadata,
  and the built-in resolvers.
* :doc:`chunked_resolution` explains slice-aware resolvers and how large
  inputs are read in chunks.
* :doc:`scoping` explains which resolvers a resolution considers, how to
  override conversions locally, how to force a path, and how to debug
  resolution.
