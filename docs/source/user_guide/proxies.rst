Proxies and Temporary Files
===========================

.. currentmodule:: scipion_bridge

External programs read and write **files**, while Python code works with
**objects**. A *proxy* is a typed Python object that stands for a file on
disk. Proxies are built on :doc:`type resolution <type_resolution>`, so any
value that can be resolved into a proxy can be passed where a file is expected.
Proxies that own a temporary file delete it automatically once they are no
longer used.

.. code-block:: python

    # Without proxies: manage paths by hand.
    xmipp_image_resize("/tmp/in.vol", "/tmp/out.vol", dim=128)
    ...
    os.remove("/tmp/out.vol")

    # With proxies: pass and receive objects.
    resized = xmipp_image_resize(volume, dim=128)    # returns a Volume proxy

.. contents:: On this page
   :local:
   :depth: 2

Defining proxy types
--------------------

Subclass ``B.Proxy`` and declare the file extension:

.. code-block:: python

    class Volume(B.Proxy):
        @classmethod
        def file_ext(cls):
            return ".vol"

Or create one in a single line with ``namedproxy``:

.. code-block:: python

    Volume = B.namedproxy("Volume", file_ext=".vol")
    Latents = B.namedproxy("Latents", file_ext=".fmlatents")

A proxy wraps a path (``proxy.path``). Proxies built from an existing path are
**unmanaged**: the library never deletes them.

.. code-block:: python

    vol = Volume("/data/map.vol")

Proxy groups
^^^^^^^^^^^^

Some formats consist of several files. A ``ProxyGroup`` bundles child proxies
that share a base path and names one of them as the *primary* proxy:

.. code-block:: python

    class ParticleStackProxy(B.ProxyGroup):
        metadata: StarfileProxy          # particles.star
        particle_stack: MRCStackProxy    # particles.mrcs

        @property
        def primary_proxy(self):
            return self.metadata

The group's ``path`` is the path of its primary proxy, so a program receives
the STAR file, and the STAR file references the stack.
``scipion_bridge.single_particle`` provides ``StarfileProxy``,
``MRCStackProxy`` and ``ParticleStackProxy``.

``@proxify``
------------

``@B.proxify`` turns a function over paths into a function over proxies. It is
typically stacked on top of ``@B.shell_command``:

.. code-block:: python

    @B.proxify
    @B.shell_command
    def xmipp_image_resize(
        i: B.ResolveProxy[Volume],
        o: B.ResolveProxy[Volume] = B.Output(Volume),
        *,
        dim: int,
    ):
        pass

    resized = xmipp_image_resize(vol, dim=128)
    # runs: xmipp_image_resize -i /data/map.vol -o /tmp/<random>.vol --dim 128
    # resized: <Volume for /tmp/<random>.vol (managed)>

Inputs: ``ResolveProxy[P]``
   The argument is resolved to proxy type ``P``, and the function receives its
   path. You can pass a ``P``, a path or string, or anything with a resolver to
   ``P``, for example an in-memory ``Set[Particle]`` for a
   ``ParticleStackProxy``, which is then written to a temporary file.

Outputs: ``= B.Output(P)``
   A default of ``Output(P)`` makes the wrapper allocate a new **managed**
   temporary file of type ``P`` and pass its path to the function. The proxy
   is returned to the caller. With several outputs, they are returned as a
   tuple.

The temporary directory comes from the injected temporary file provider. Under
the pyworkflow backend, this is the protocol's ``tmp`` directory.

Automatic reference counting
----------------------------

**Managed** proxies own their file. A process-wide reference counter (ARC)
tracks how many live proxies refer to each managed path:

* creating a managed proxy sets the count of its path to 1;
* another proxy for the same path increments it;
* a proxy being garbage collected decrements it.

When the count reaches zero, the file is deleted. Intermediate files therefore
live exactly as long as Python objects refer to them, without ``try`` /
``finally`` blocks or cleanup passes:

.. code-block:: python

    def pipeline(vol):
        filtered = xmipp_transform_filter(vol, fourier="low_pass 0.1")
        return xmipp_image_resize(filtered, dim=128)

    out = pipeline(vol)    # the filtered intermediate is already deleted

Retyping proxies
----------------

``typed(astype=P)`` converts a proxy into a proxy of another type. By default
the file is copied to a path with the new extension:

.. code-block:: python

    text = untyped.typed(astype=TextFile)

Some programs append their own extension to the output path they are given
(they write ``/tmp/x.vol`` when passed ``/tmp/x``). Request an untyped output,
then retype it **without copying**:

.. code-block:: python

    @B.proxify
    @B.shell_command
    def appends_extension(*, output: B.ResolveProxy = B.Output(B.Proxy)):
        pass

    result = appends_extension().typed(astype=Volume, copy_data=False)

With ``copy_data=False`` the proxy is just re-pointed to the path with the
extension.
