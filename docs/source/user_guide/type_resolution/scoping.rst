.. _resolver-visibility:

Visibility, Overrides and Debugging
===================================

.. currentmodule:: scipion_bridge

The resolution graph is global, but a resolution does not consider every
resolver in it. This article explains which resolvers are visible from a
call, how a module overrides a conversion locally, how a path can be forced
through a specific type, and how to find out why a resolution behaved as it
did.

.. contents:: On this page
   :local:
   :depth: 2

Visibility
----------

Resolvers are registered under the module that defines them. A resolution
only considers resolvers whose module is **visible** from the call site:

* the calling module itself;
* modules the calling module has imported (any module object, or the module
  of any object, in its global namespace);
* the module that defines the value's type;
* the ``scipion_bridge`` package.

The call site is the first frame outside of the ``scipion_bridge`` package,
so resolutions made by the library on behalf of user code, such as those of
``@B.proxify`` or the Ray runner, see the resolvers visible from the user's
module.

Visibility keeps plugins from interfering with each other: a resolver defined
by one plugin does not change the behavior of another plugin that never
imports it.

Making resolvers visible
------------------------

Import the module that defines resolvers wherever they should apply. For a
package, the simplest way is to import its resolvers module in the package's
``__init__.py``, and to lift the resolvers into the package:

.. code-block:: python

    # my_plugin/__init__.py
    import scipion_bridge as B
    from . import resolvers

    B.lift_resolvers(resolvers)

``B.lift_resolvers(module)`` re-registers the resolvers of a submodule under
the calling package, so that every module of the package, and every module
that imports it, sees them. The library does this for its own built-in
resolvers, which is why ``import scipion_bridge`` is enough to use them.
``lift_resolvers`` accepts several modules and an explicit ``target`` package.

Local overrides
---------------

When paths have equal length, resolvers registered in the caller's local
scope win: a resolver defined inside the function that performs the
resolution takes precedence over resolvers of the same length registered
elsewhere. This allows a single function to replace a conversion for its own
calls, without affecting other code:

.. code-block:: python

    def export(particles):
        @B.resolver
        def write_with_extra_columns(value: B.Set[Particle]) -> B.ParticleStackProxy:
            ...

        # Uses write_with_extra_columns instead of the built-in resolver.
        return B.resolve(particles, astype=B.ParticleStackProxy)

For a module-wide replacement, register the resolver for a distinct proxy
type and force it with an intermediate type, as described next. Relying on
ties between module-level resolvers of different plugins is fragile.

Forcing an intermediate type
----------------------------

``Resolve[Target, Via]`` (or ``intermediate=Via``) forces the path through a
specific type. This is useful when several paths exist and only one of them
has the side effect needed, for example writing a particular file format:

.. code-block:: python

    @B.resolve_params
    def run(path: B.Resolve[str, MyProxy]): ...

    B.resolve(value, astype=str, intermediate=MyProxy)

``B.ResolveProxy[P]`` in ``@B.proxify`` functions works the same way: the
argument is resolved through ``P``.

Debugging
---------

Set the log level to ``INFO`` to log every resolution: the source and target
types, the caller's namespace, and the time spent on the search and on the
conversion. At ``DEBUG``, the visible namespaces and every step of a path are
logged as well:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)

Common problems and their causes:

``'X' could not be resolved as 'Y'``
   No path exists among the visible resolvers. Check that the module defining
   the resolver is imported by the calling module, or lifted into a package
   that is. Remember that ``Set[Sub]`` is not a subclass of ``Set[Base]``.

A different path than expected is taken
   A shorter path exists, or a path of equal length wins a tie. Log at
   ``DEBUG`` to see the steps, and force the intermediate type if necessary.

``The resolved output ... did not match target data type``
   A resolver returned a value of the wrong type. The message lists every
   resolver of the path; the faulty one is usually the last.

``Cannot iterate resolution ... no resolver along the path supports slicing``
   ``resolve_iter`` (or the Ray runner) needs a slice-aware step on the path.
   See :doc:`chunked_resolution`.
