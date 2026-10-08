Type Resolution
===============

.. currentmodule:: scipion_bridge

The same particles can exist in many forms: an in-memory ``Set[Particle]``, a
STAR file with an MRC stack, a ``SetOfParticles`` in a Scipion project, or a
NumPy array. **Type resolution** converts between these forms automatically.
You state the type you need, and the library finds a chain of converters that
produces it.

.. code-block:: python

    @B.resolve_params
    def mean_defocus(particles: B.Resolve[B.Set[Particle]]) -> float:
        return float(particles["ctf"]["defocus_u"].mean())

    mean_defocus(B.Set[Particle](...))                     # used as is
    mean_defocus(B.ParticleStackProxy("particles"))        # read from disk first

.. contents:: On this page
   :local:
   :depth: 2

Resolvers
---------

A **resolver** is a stateless function that converts a value of one type into
an equivalent value of another type. Register one with ``@B.resolver``. The
annotations of the ``value`` parameter and of the return value define the edge
it adds:

.. code-block:: python

    @B.resolver
    def resolve_particle_stack_proxy(value: B.Set[Particle]) -> B.ParticleStackProxy:
        """Write a Set[Particle] to a temporary STAR file and MRC stack."""
        new_proxy = B.ParticleStackProxy.new_temporary_proxy()
        ...
        return new_proxy

Both annotations are required.

The resolution graph
^^^^^^^^^^^^^^^^^^^^

All resolvers form a directed graph: **nodes are types, and edges are
resolvers**. Resolving ``value`` to type ``T`` searches the **shortest path**
from ``type(value)`` to ``T`` (Dijkstra's algorithm) and applies the resolvers
along it in order.

.. code-block:: text

      spa.Particle (Set)       np.ndarray
              \                  /
               ▼                ▼
          spa.ParticleStackProxy
                    │
                    ▼
                  Path

Subclasses are handled automatically: when no resolver is registered for a
type, the search starts from the nearest base class in its MRO that has one.
If no path exists, ``TypeError`` is raised.

Resolving values
----------------

There are three ways to resolve a value:

.. code-block:: python

    # 1. Directly
    stack = B.resolve(particles, astype=B.ParticleStackProxy)

    # 2. Through annotated parameters
    @B.resolve_params
    def f(stack: B.Resolve[B.ParticleStackProxy]): ...

    # 3. Precompute the path once and reuse it (hot loops)
    to_set = B.find_resolver(Path, B.Set[Particle])
    particles = to_set(Path("particles.star"))

``find_resolver`` returns a ``ComposedResolver``. It runs the graph search
once, so calling it repeatedly costs only the conversions. The Ray runner
precomputes a resolver for every protocol input when it is created, so no
graph search happens while data streams.

Forcing an intermediate type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Resolve[Target, Via]`` (or ``intermediate=Via``) forces the path through a
specific type. This is useful when several paths exist and only one of them
has the side effect you need:

.. code-block:: python

    def run(path: B.Resolve[str, MyProxy]): ...
    B.resolve(value, astype=str, intermediate=MyProxy)

Built-in resolvers
^^^^^^^^^^^^^^^^^^

* ``object -> str`` via ``str(value)``.
* ``tuple -> str``: the elements are resolved and joined with spaces
  (``(1, 2, 3)`` becomes ``"1 2 3"``).
* ``ak.Array -> np.ndarray``.
* ``Set[Particle] <-> ParticleStackProxy``, ``StarfileProxy ->
  ParticleStackProxy``, ``StarfileProxy -> MRCStackProxy``, and related
  conversions in ``scipion_bridge.single_particle``.
* Path and output conversions used by :doc:`proxies <proxies>`.
* With the pyworkflow backend: conversions between Scipion objects (for
  example ``SetOfParticles``) and the array types.

Chunked resolution
------------------

Large inputs should not be loaded at once. A resolver that accepts a ``slice``
keyword argument is **slice-aware**: it can produce just one part of its
output.

.. code-block:: python

    @B.resolver
    def stack_to_particles(
        value: B.ParticleStackProxy,
        *,
        slice: slice | None = None,
    ) -> B.Set[Particle]:
        ...   # read only images [slice] from the memory-mapped MRC stack

When a path contains a slice-aware step, the value can be resolved **in
chunks**:

.. code-block:: python

    for chunk in B.resolve_iter(Path("particles.star"), astype=B.Set[Particle],
                                chunk_size=1_000):
        pipeline.send(particles=chunk)

The steps before the first slice-aware step run once. The rest of the path runs
once per chunk. Without ``chunk_size``, the chunk size is estimated from the
intermediate's ``estimated_item_nbytes`` so that each chunk holds about
``target_bytes`` (32 MiB by default). See ``B.estimate_optimal_chunk_size``.

When several paths have the same length, slice-aware ones are preferred.

Stateful resolver classes
^^^^^^^^^^^^^^^^^^^^^^^^^

A resolver can also be a class with a ``forward`` method. One instance is
created per resolution, and in chunked resolution one per stream, so the
instance can cache work across chunks (for example, a parsed STAR file):

.. code-block:: python

    @B.resolver
    class stack_to_particles:
        def __init__(self):
            self._star_cache = {}

        def forward(self, value: B.ParticleStackProxy, *, slice=None) -> B.Set[Particle]:
            ...

Resolver classes must not implement ``__iter__``. A resolver is a transformer,
not an iterator.

Resolvers may also accept a ``metadata`` keyword argument, which receives the
``metadata=`` value passed to ``resolve``.

Visibility and scoping
----------------------

Resolvers are registered under the module that defines them. A resolution
only considers resolvers whose module is **visible** from the call site:

* the calling module itself;
* modules the calling module has imported;
* the module that defines the value's type;
* the ``scipion_bridge`` package.

When paths have equal length, resolvers from the caller's own module win. This
lets a plugin override a conversion locally without affecting others.

``B.lift_resolvers(module)`` re-registers the resolvers of a submodule under a
parent package, so that importing the package is enough to make them visible.
The library does this for its own built-in resolvers.

Debugging
---------

Set the log level to ``INFO`` to log every resolution and the resolvers it
uses. When a resolver returns a value of the wrong type, the resulting
``TypeError`` lists the full path of resolvers that was taken.
