Proxies and Temporary Files
===========================

.. currentmodule:: scipion_bridge

External programs read and write **files**, while Python code works with
**objects**. A *proxy* is a typed Python object that stands for a file on
disk. Because proxies are types, they take part in
:doc:`type resolution </user_guide/type_resolution/index>`: any value that can
be resolved into a proxy can be passed where a file is expected. Proxies that
own a temporary file delete it automatically once no Python object refers to
it.

.. code-block:: python

    # Without proxies: paths are managed by hand.
    xmipp_image_resize("/tmp/in.vol", "/tmp/out.vol", dim=128)
    ...
    os.remove("/tmp/out.vol")

    # With proxies: objects are passed and returned.
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

or create one in a single line with ``namedproxy``:

.. code-block:: python

    Volume = B.namedproxy("Volume", file_ext=".vol")
    HalfMap = B.namedproxy("HalfMap", file_ext=".mrc", suffix="_half1")

Besides the extension, a proxy type may declare a ``prefix`` and a ``suffix``
of the file name. Together they describe the file names of the format, and
are used both to check existing paths and to name new temporary files.

A proxy wraps a path, available as ``proxy.path``. Proxies constructed from an
existing path are **unmanaged**: the library never deletes their file.

.. code-block:: python

    vol = Volume("/data/map.vol")

Every proxy type automatically gets a resolver from ``Path``, which checks that
the file name matches the declared extension, prefix and suffix and raises
``TypeError`` otherwise. A ``Path`` can therefore be passed wherever the proxy
type is expected.

Proxies for streamed inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A proxy type that serves as the source of chunked resolution, such as the
input file of a protocol, implements two more members:

* ``__len__``, the number of elements in the file;
* ``estimated_item_nbytes``, the size of one element in bytes, used to choose
  a chunk size. It returns ``None`` by default.

See :doc:`/user_guide/type_resolution/chunked_resolution`.

Proxy groups
------------

Some formats consist of several files: a STAR file and the MRC stack it
references, a map and its half maps. A ``ProxyGroup`` bundles child proxies
that share a base path, and names one of them as the *primary* proxy:

.. code-block:: python

    class ParticleStackProxy(B.ProxyGroup):
        metadata: StarfileProxy          # <base>.star
        particle_stack: MRCStackProxy    # <base>.mrcs

        @property
        def primary_proxy(self):
            return self.metadata

The children are declared as annotations, and their paths are derived from
the base path (which has no extension) by adding each child's prefix, suffix
and extension. The group's ``path`` is the path of its primary proxy, so a
program receives the STAR file, and the STAR file references the stack. A
group behaves like a mapping of its children (``group["metadata"]``,
``group.items()``), and ``len(group)`` requires all children to report the
same length.

A group can be constructed from its base path, or resolved from a path of its
primary proxy: ``Path("particles.star")`` resolves to a
``ParticleStackProxy`` with base ``particles``.

``scipion_bridge.single_particle`` provides ``StarfileProxy``,
``MRCStackProxy`` and ``ParticleStackProxy``.

``@proxify``
------------

``@B.proxify`` turns a function over paths into a function over proxies. It
is typically stacked on top of ``@B.shell_command``:

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

Every argument passed to a proxified function is resolved into a string for
the command line. Annotations decide how:

Inputs: ``B.ResolveProxy[P]``
   The argument is resolved to proxy type ``P``, and the function receives its
   path. A ``P``, a ``pathlib.Path`` to a file of the format, or anything with
   a resolver to ``P`` can be passed, for example an in-memory
   ``Set[Particle]`` for a ``ParticleStackProxy``, which is then written to
   temporary files.

Outputs: ``= B.Output(P)``
   A default of ``Output(P)`` makes the wrapper allocate a new **managed**
   temporary file of type ``P`` and pass its path to the function. After the
   call, the proxy is returned to the caller. With several outputs, they are
   returned as a tuple, in parameter order. Passing a ``Path`` for an output
   parameter writes to that path instead.

Other parameters
   Values are converted to strings through type resolution: a ``tuple`` is
   joined with spaces, everything else uses ``str()``. ``None`` leaves the
   argument out.

The temporary directory comes from the injected temporary file provider: the
system temporary directory in standalone and Ray use, the protocol's ``tmp``
directory under the pyworkflow backend.

Automatic reference counting
----------------------------

**Managed** proxies own their file. A reference counter (ARC) tracks how many
live proxies refer to each managed path:

* creating a managed temporary proxy registers its path with a count of 1;
* another managed proxy for the same path increments the count, and so does
  an argument of a running proxified call that refers to it;
* a proxy being garbage collected decrements it.

When the count reaches zero, the file is deleted through the temporary file
provider. Intermediate files therefore live exactly as long as Python objects
refer to them, without ``try`` / ``finally`` blocks or cleanup passes:

.. code-block:: python

    def pipeline(vol):
        filtered = xmipp_transform_filter(vol, fourier="low_pass 0.1")
        return xmipp_image_resize(filtered, dim=128)

    out = pipeline(vol)    # the filtered intermediate is already deleted

For proxy groups, each child file is counted separately.

Lifetime across processes
^^^^^^^^^^^^^^^^^^^^^^^^^

The reference counts are kept per process. Managed files are best used within
one stage call: create them, run the programs, and read the results back into
memory before the call returns. Returning a managed proxy from a stage hands a
path to another process, possibly on another machine, whose files are not
tracked there.

Unmanaged paths are never deleted. Counting references for a path that was
not created as a temporary file emits a ``DeprecationWarning``, because such
paths usually belong to the user.

Retyping proxies
----------------

``typed(astype=P)`` converts a proxy into a proxy of another type. By default,
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

With ``copy_data=False``, the proxy is re-pointed to the path with the
extension. Only untyped proxies (without an extension) can be retyped.
