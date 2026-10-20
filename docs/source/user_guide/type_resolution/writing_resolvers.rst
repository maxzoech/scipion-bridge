Writing Resolvers
=================

.. currentmodule:: scipion_bridge

A resolver adds one edge to the resolution graph. This article describes the
two forms of resolvers, functions and classes, the optional ``metadata`` and
``slice`` arguments, and the resolvers that ship with the library.

.. contents:: On this page
   :local:
   :depth: 2

Function resolvers
------------------

A resolver is a stateless function registered with ``@B.resolver``. The
annotation of its ``value`` parameter is the source type, and its return
annotation is the target type:

.. code-block:: python

    @B.resolver
    def resolve_particle_stack_proxy(value: B.Set[Particle]) -> B.ParticleStackProxy:
        """Write a Set[Particle] to a temporary STAR file and MRC stack."""
        new_proxy = B.ParticleStackProxy.new_temporary_proxy()
        ...
        return new_proxy

Both annotations are required, and the first parameter must be called
``value``. The function should be a pure conversion: the result must be an
equivalent representation of the input, because the resolver may be one step
of a longer path that the caller never sees. A resolver that returns a value
of the wrong type causes a ``TypeError`` at the end of the resolution, which
lists every resolver of the path.

Resolvers can call ``B.resolve`` themselves, for example to delegate part of
a conversion. Nested resolutions use the visibility of the outer call.

Class resolvers
---------------

A resolver can also be a class with a ``forward`` method, whose ``value``
parameter and return annotation define the edge:

.. code-block:: python

    @B.resolver
    class stack_to_particles:
        def __init__(self):
            self._star_cache = {}

        def forward(self, value: B.ParticleStackProxy, *, slice=None) -> B.Set[Particle]:
            ...

One instance is created per resolution, and in chunked resolution one per
stream, so the instance can cache work across the chunks of one input, such
as a parsed STAR file. Use a class when such caching pays off; otherwise a
function is simpler.

Resolver classes must not implement ``__iter__``: a resolver is a
transformer, not an iterator, and an iterable class would be ambiguous in
chunked resolution.

Optional arguments
------------------

A resolver may accept two keyword arguments, which the library passes only to
resolvers that declare them:

``slice``
   Makes the resolver *slice-aware*: it can produce one part of its output.
   See :doc:`chunked_resolution`.

``metadata``
   Receives the ``metadata=`` value passed to ``resolve``. The Scipion backend
   uses it to pass the running Scipion protocol and the name of the output to
   its resolvers, which need them to create output sets in the protocol's
   directory or to read only the new items of an input.

.. code-block:: python

    @B.resolver
    def to_scipion(value: B.Set[Particle], metadata=None) -> SetOfParticles:
        protocol = metadata.protocol
        ...

Opt-in routes
-------------

Resolution takes the route of lowest cost through the graph. Every resolver
costs 0 by default, so when two routes of the same cost lead to the target,
which one is taken is not defined. Give the resolver of the alternative
route a positive ``weight`` to keep it out of default resolutions:

.. code-block:: python

    @B.resolver(weight=1)
    def resolve_xmipp_spider_stack_to_particles(
        value: XmippSpiderParticleStackProxy,
        *,
        slice=None,
    ) -> B.Set[Particle]:
        ...

The route is then taken only when it is requested, e.g.
``B.find_resolver(Path, B.Set[Particle], intermediate=XmippSpiderParticleStackProxy)``,
or for a value that can only take that route.

Resolvers for subclasses
------------------------

Registering a resolver for a type also makes it reachable from subclasses of
that type: the graph contains implicit edges from every registered type to its
base classes. A ``ParticleStackProxy`` subclass therefore resolves to
``Set[Particle]`` through the resolver of ``ParticleStackProxy``.

Specialized containers are separate classes, however. ``Set[FlexParticle]``
is not a subclass of ``Set[Particle]``, and needs resolvers of its own.

Duplicate registrations
-----------------------

Registering a second, different resolver for the same pair of types in the
same module emits a warning and keeps the first one. Resolvers for the same
pair in different modules coexist; which one a resolution uses depends on
visibility (see :doc:`scoping`).

Built-in resolvers
------------------

The library registers these resolvers:

* ``object -> str`` via ``str(value)``;
* ``tuple -> str``: the elements are resolved to strings and joined with
  spaces, so ``(1, 2, 3)`` becomes ``"1 2 3"`` on a command line;
* ``ak.Array -> np.ndarray``;
* ``Path -> P`` for every proxy type ``P``, checking the file extension,
  prefix and suffix that the proxy declares;
* ``Path``, ``str``, proxies and proxy groups to ``FuncParam``, the argument
  representation of shell commands, used by ``@B.proxify``;
* the single particle conversions listed in
  :doc:`/user_guide/structs/single_particle`;
* with the ``pyworkflow`` extra, conversions between Scipion objects
  (``SetOfParticles``, ``SetOfParticlesFlex``, ``SetOfClasses2D``) and the
  array types.
