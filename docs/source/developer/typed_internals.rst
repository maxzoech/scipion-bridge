Type Resolution, Proxies and Commands
=====================================

This article describes the machinery behind type resolution, proxies,
automatic reference counting and shell commands, and the way specialized
generic classes such as ``Set[Particle]`` keep their identity across
processes. The user-facing behavior is described in
:doc:`/user_guide/type_resolution/index`, :doc:`/user_guide/proxies` and
:doc:`/user_guide/shell_commands`.

.. contents:: On this page
   :local:
   :depth: 2

The registry
------------

``core/typed/resolve.py`` keeps a single ``Registry`` (``DEFAULT_REGISTRY``),
holding a ``networkx.DiGraph`` whose nodes are types. Each edge carries:

* ``resolver``: a class with a ``forward`` method;
* ``module``: the namespace the resolver is registered under;
* ``weight``: 0 for registered resolvers;
* ``requires_metadata`` and ``requires_slice``: whether ``forward`` accepts
  those keyword arguments.

``@resolver`` reads the ``value`` and return annotations with
``typing.get_type_hints`` and calls ``add_resolver``. Function resolvers are
wrapped in a generated projector class with a ``forward`` method, so that
both forms are executed the same way. The decorator returns the class for
class resolvers and an instance of the projector for functions, so a decorated
function remains callable.

For both endpoints of every new edge, *downcast* edges to each base class in
the type's MRO are added, with the base's position in the MRO as weight and an
identity resolver. They make resolvers of a base class reachable from its
subclasses, at a small cost that prefers more specific paths.

The namespace of a resolver is its module plus its qualified name without the
last component, with ``<locals>`` removed: a module-level function in
``pkg.mod`` is registered under ``pkg.mod``, a resolver defined inside the
function ``pkg.mod.f`` under ``pkg.mod.f``. Registering a different resolver
for an existing edge in the same namespace warns and keeps the first.

Lookup
------

A lookup (``resolve``, ``find_resolver``, ``resolve_iter``) proceeds as
follows:

1. **Find the caller.** ``_find_calling_frame`` walks the stack to the first
   frame outside the ``scipion_bridge`` package. Its module and the qualified
   name of its code form the *calling namespace*.
2. **Collect visible namespaces.** These are the calling namespace, the
   modules of all objects in the caller's globals, the namespace of the
   value's type, and the ``scipion_bridge`` package, each expanded into its
   prefixes (``a.b.c`` gives ``a``, ``a.b``, ``a.b.c``). The result is
   intersected with the expanded namespaces that have registered resolvers.
3. **Restrict the graph** to edges whose namespace is visible.
4. **Find the origin.** The search starts from the first class in the value
   type's MRO that is a node of the restricted graph.
5. **Search.** A Dijkstra search (``dijkstra.py``) finds the shortest path to
   the target, optionally through a required intermediate type. Ties between
   paths of equal weight are broken by ``ScopedPathfindingContainer``: first
   resolvers whose namespace starts with the calling namespace (resolvers
   local to the calling function), then slice-aware resolvers, then the
   resolver with the longer (more specific) symbol path.
6. **Compose.** The path becomes a ``ComposedResolver``, a list of
   ``ResolveStep`` entries.

Nested resolutions, made by a resolver while it runs, reuse the namespaces of
the outer resolution through a context variable, so a resolver sees the same
resolvers as its caller.

``lift_resolvers(module, target=...)`` rewrites the namespace of every edge
registered under ``module`` to ``target`` (the calling module by default).
The target must be a prefix of the module name.

ComposedResolver
----------------

A ``ComposedResolver`` runs its steps in order, instantiating each resolver
class per call and passing ``metadata`` and ``slice`` to the steps that
accept them. A requested slice is passed to the first slice-aware step only;
later steps receive ``None`` (the *narrowing rule*). A slice requested on a
path without a slice-aware step raises ``TypeError``. The output is checked
against the target type, and a mismatch raises ``TypeError`` listing the
steps.

``iter`` splits the steps at the first slice-aware one. The prefix runs once;
its result must be ``Sized``. The chunk size is given, or estimated from the
intermediate's ``estimated_item_nbytes``. One instance per stream step is
created before the loop, so class resolvers can cache across chunks. When a
``ComposedResolver`` is created, each step's class may implement a
``precompute`` class method, which is called once; proxy group resolvers use
it to find the resolver of their primary proxy ahead of time.

Proxies
-------

``core/typed/proxy.py`` defines ``Proxy`` and ``ProxyGroup``. The metaclass
``ProxyMetaclass`` registers, for every concrete proxy class, a resolver from
``Path`` to it. The generated resolver checks the file extension, prefix and
suffix. For proxy groups, the resolver accepts a base path without extension,
or a path of the primary proxy, which it resolves to the primary proxy type
and on to the group.

``new_temporary_proxy`` allocates a path through the temporary file provider,
registers it with ARC with a count of 0, and constructs a managed proxy, which
increments the count to 1. A proxy group allocates a base path and creates one
managed child per declared field; the group itself is not counted, only its
children.

proxify
^^^^^^^

``@proxify`` wraps a function so that each argument is resolved to a
``FuncParam``: a string for the command line, plus the proxy type and whether
it is managed. ``ResolveProxy[P]`` annotations and ``Output(P)`` values set
``P`` as a required intermediate of the resolution. An ``Output`` is resolved
through a resolver registered when the ``Output`` is created, which allocates
a new temporary proxy.

A ``FuncParam`` for a managed proxy adds an ARC reference to each file of
the proxy for its own lifetime, so the files of temporary inputs outlive the
proxies that created them until the wrapped call has returned. After the
call, the wrapper builds the proxies to return from the ``FuncParam`` of
each ``Output`` parameter.

Automatic reference counting
----------------------------

``core/utils/arc.py`` implements a process-wide ``FileReferenceCounter``
(``manager``):

* ``register_temporary_file`` and ``new_managed_file`` start tracking a path;
* ``add_reference`` and ``remove_reference`` are called when managed proxies
  and ``FuncParam`` objects for the path are created and garbage collected;
* when a count reaches zero, the file is deleted through the
  ``temp_file_provider`` and the path is forgotten.

Adding a reference to a path that is not tracked emits a
``DeprecationWarning``: such paths are usually persistent user files, which
reference counting must not delete. Errors in ``Proxy.__del__`` are logged,
not raised, since exceptions in finalizers cannot be handled by the caller.

Shell commands
--------------

``core/utils/shell.py`` implements ``@shell_command``. At definition time, the
decorator parses the function's source with ``ast`` and requires the body to
be a single ``pass`` statement, records the signature and the parameters
annotated as ``bool``, and rejects boolean positional parameters.

Calling the wrapper first calls the original function with the arguments, so
that Python validates them. The arguments are then bound to the signature,
``None`` values dropped, each parameter translated (``-name`` for positional,
``--name`` for keyword-only, bare flags for booleans, renamed through the
decorator's keyword arguments), ``postprocess_fn`` applied, and the domain
prefix and program name prepended. The injected ``shell_exec`` provider
receives the program name, the domain, the argument list and the subprocess
options.

Calling ``shell_command`` with keyword arguments only returns a decorator
that merges further keyword arguments, which is how a domain-bound decorator
can be specialized.

Specialized generic classes
---------------------------

``Set[Particle]`` must be the same class in every process: output validation
checks instances against it, the resolution graph uses it as a node, and
pickled sets refer to it. ``core/utils/marker.py`` implements this.

Every specialization ``Origin[Args]`` of a ``Marker`` subclass is a real class,
created once and cached. If the origin and all arguments are importable
classes, the specialization gets a ``__qualname__`` that encodes them without
dots, for example ``Set[scipion_bridge/single_particle/particle:Particle]``,
and is stored under that name on the origin's module. Pickling then stores a
reference by module and name. Every module defining a ``Marker`` class gets a
module-level ``__getattr__`` (PEP 562) that rebuilds a specialization from its
encoded name, so a process that has never built ``Set[Particle]`` reconstructs
it on unpickling and obtains the cached class.

Specializations whose origin or arguments cannot be imported by name (defined
in ``__main__`` or a local scope) keep a readable name and are serialized by
value, which requires cloudpickle.
