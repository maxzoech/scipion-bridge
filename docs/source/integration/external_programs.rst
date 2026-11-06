Step 2: Wrapping External Programs
==================================

.. currentmodule:: scipion_bridge

Many methods are only available as command line programs. This page shows how
to make such a program callable from a protocol as if it were a Python
function that takes and returns in-memory data. Three pieces work together:

1. a **command declaration** (``@B.shell_command``) describing the program's
   arguments;
2. **proxy types** for the file formats the program reads and writes;
3. **resolvers** that convert between those files and the data types of the
   protocol.

If the method is called in-process, skip to :doc:`writing_protocols`. The
mechanics of each piece are described in :doc:`/user_guide/shell_commands`,
:doc:`/user_guide/proxies` and :doc:`/user_guide/type_resolution/index`.

.. contents:: On this page
   :local:
   :depth: 2

Declaring the command
---------------------

The denoiser of the running example is invoked as:

.. code-block:: bash

    cryodenoise predict --input particles.star --weights model.pt \
        --output denoised.mrcs --scores scores.csv

The program name ``cryodenoise`` is followed by a sub-command, so the
program name goes into a ``Domain`` (a command prefix shared by several
programs) and the sub-command becomes the function name:

.. code-block:: python

    # scipion_denoise/programs.py
    import scipion_bridge as B

    denoise_domain = B.Domain("cryodenoise", ["cryodenoise"], isolated=True)


    @B.shell_command(domain=denoise_domain, name="predict", particles="input")
    def cryodenoise_predict(*, particles: str, weights: str, output: str, scores: str):
        pass

The declaration is an interface, not an implementation. Its body must be
exactly ``pass``; any other statement, including a docstring, raises
``RuntimeError`` when the module is imported. The parameters map to arguments
as follows:

* keyword-only parameters (after ``*``) become ``--name VALUE``;
* positional-or-keyword parameters become ``-name VALUE``;
* keyword arguments of the decorator rename parameters: ``particles="input"``
  turns the parameter ``particles`` into ``--input``;
* parameters annotated as ``bool`` become flags, present only when true;
* arguments passed as ``None`` are left out.

Calling ``cryodenoise_predict(particles="in.star", weights="model.pt",
output="out.mrcs", scores="s.csv")`` now runs the program with these paths.
The call blocks until the program exits, and a non-zero exit code raises
``RuntimeError`` with the program's standard error.

.. note::

   The arguments are joined with spaces and run through a shell, without
   quoting. Paths created by the library contain no spaces, but values
   supplied by users should not contain spaces or shell metacharacters.

``isolated=True`` marks programs that live in a conda environment of their
own. The Scipion backend activates that environment before running them (see
:doc:`deployment`). On Ray and in standalone use, the program must be found on
the ``PATH`` of the worker processes; alternatively, the domain's prefix can
select the environment explicitly, for example
``["conda", "run", "-n", "cryodenoise", "cryodenoise"]``.

Passing data instead of paths
-----------------------------

With paths as arguments, the protocol would still have to write each batch to
a STAR file and MRC stack, allocate output paths, read the results back and
delete everything afterwards. ``@B.proxify`` automates this. Stacked on top
of the command, it changes the parameter annotations from paths to *proxy
types*, which stand for files of a given format:

.. code-block:: python

    ScoreTable = B.namedproxy("ScoreTable", file_ext=".csv")


    @B.proxify
    @B.shell_command(domain=denoise_domain, name="predict", particles="input")
    def cryodenoise_predict(
        *,
        particles: B.ResolveProxy[spa.ParticleStackProxy],
        weights: str,
        output: B.ResolveProxy[spa.MRCStackProxy] = B.Output(spa.MRCStackProxy),
        scores: B.ResolveProxy[ScoreTable] = B.Output(ScoreTable),
    ):
        pass

The two kinds of annotations have different roles:

``B.ResolveProxy[P]`` on an input
   The argument is *resolved* to the proxy type ``P`` and the program receives
   the proxy's path. The argument may be a ``P``, a ``pathlib.Path`` to a file
   of that format, or any value for which a resolution path to ``P`` exists.
   For ``ParticleStackProxy``, an in-memory ``Set[Particle]`` qualifies: a
   built-in resolver writes it to a temporary STAR file and MRC stack.

``= B.Output(P)`` as a default
   Before the call, a new temporary file of type ``P`` is allocated and its
   path passed to the program. After the call, the proxy is returned to the
   caller. With several outputs, they are returned as a tuple in the order of
   the parameters.

The call now takes a batch and returns proxies:

.. code-block:: python

    stack, score_table = cryodenoise_predict(particles=batch, weights=weights_path)
    # runs: cryodenoise predict --input /tmp/k3j...star --weights /models/w.pt
    #       --output /tmp/9fq...mrcs --scores /tmp/x0a...csv

The temporary files are *managed*: a reference counter deletes each of them
when the last proxy referring to it is garbage collected. The input STAR file
and stack are deleted as soon as the call returns; ``stack`` and
``score_table`` live as long as the variables refer to them.

Reading results back into memory
--------------------------------

A stage should return in-memory data, not proxies. Temporary files belong to
the process that created them, and the next stage may run in another process
or on another machine. Convert the outputs back into the protocol's types
before the stage returns.

For formats that are read in more than one place, write a resolver. A
resolver is a function from one type to another, registered with
``@B.resolver``; the annotations of the ``value`` parameter and of the return
value define the conversion it provides:

.. code-block:: python

    # scipion_denoise/resolvers.py
    import numpy as np
    import scipion_bridge as B

    from .programs import ScoreTable    # ScoreTable lives next to the command


    @B.resolver
    def read_score_table(value: ScoreTable) -> np.ndarray:
        """Read one score per particle from the program's CSV output."""
        return np.loadtxt(value.path, delimiter=",", ndmin=1)

The MRC stack can be read with ``mrcfile`` directly. Assembling the output
set is plain code in the stage:

.. code-block:: python

    # scipion_denoise/programs.py (continued)
    import mrcfile
    import numpy as np
    from scipion_bridge import single_particle as spa

    from .types import DenoisedParticle


    def denoise_batch(batch: B.Set[spa.Particle], weights: str) -> B.Set[DenoisedParticle]:
        stack, score_table = cryodenoise_predict(particles=batch, weights=weights)

        with mrcfile.open(stack.path) as mrc:
            pixels = np.array(mrc.data, dtype=np.float32)
        scores = B.resolve(score_table, astype=np.ndarray)

        out = B.Set[DenoisedParticle](capacity=len(batch))
        out["pixels"] = pixels
        out["sampling_rate"] = batch["sampling_rate"]
        out["snr"] = scores.reshape(-1, 1)
        return out

Reading a field that the input never provided raises
``UninitializedFieldError``, so copy only the fields that the input is known
to carry.

Making a new input format streamable
------------------------------------

The Ray runner reads protocol inputs from files in chunks. For that, the
resolution path from ``Path`` to the protocol's input type must contain a
*slice-aware* resolver: one that accepts a ``slice`` keyword argument and
produces only that part of the data. The built-in particle types already have
one (STAR file and MRC stack to ``Set[Particle]``). A protocol whose input
comes in another format needs a proxy type and a slice-aware resolver for it.

As an example, take particle images stored as a NumPy ``.npy`` array of shape
``(N, H, W)``:

.. code-block:: python

    import numpy as np
    import scipion_bridge as B
    from scipion_bridge import single_particle as spa


    class NpyImageStack(B.Proxy):
        @classmethod
        def file_ext(cls):
            return ".npy"

        def __len__(self) -> int:
            return np.load(self.path, mmap_mode="r").shape[0]

        @property
        def estimated_item_nbytes(self) -> int:
            stack = np.load(self.path, mmap_mode="r")
            return stack[0].nbytes


    @B.resolver
    def npy_stack_to_particles(
        value: NpyImageStack,
        *,
        slice: slice | None = None,
    ) -> B.Set[spa.Particle]:
        stack = np.load(value.path, mmap_mode="r")
        images = np.asarray(stack if slice is None else stack[slice], dtype=np.float32)

        particles = B.Set[spa.Particle](capacity=len(images))
        particles["pixels"] = images
        return particles

Three details make this work with the runner:

* The runner calls ``len()`` on the proxy to know how many elements there
  are, and reads ``estimated_item_nbytes`` to choose a chunk size of about
  32 MiB. Without the estimate, it falls back to 100 elements per chunk; the
  environment variables ``SCIPION_CHUNK_SIZE`` and ``SCIPION_TARGET_BYTE_SIZE``
  override both.
* The resolver memory-maps the file, so reading a chunk only reads that
  chunk.
* Defining a ``Proxy`` subclass registers a resolver from ``Path`` to it,
  which checks the file extension. A path from ``Path`` to ``Set[Particle]``
  through ``NpyImageStack`` therefore exists as soon as the resolver above is
  registered.

The last point needs care. Type resolution chooses a path by **types**, not by
file names, and the runner precomputes one path per input from the input's
origin type (``Path`` by default). ``Set[Particle]`` can now be reached from
``Path`` both through ``NpyImageStack`` and through the built-in
``ParticleStackProxy``, and the precomputed path is the same for every file,
whatever its extension. When a new format produces a type that another format
already produces, declare the proxy type as the origin of the input and pass
proxies instead of paths:

.. code-block:: python

    runner = RayPipelineRunner(
        Denoise(),
        origin_types={"particles": NpyImageStack},
    )
    runner.run(particles=NpyImageStack("particles.npy"))

When the new format produces a type of its own, the path from ``Path`` is
unique, and inputs can be given as paths, including on the command line.

If a resolver needs to cache work across chunks, for example a parsed
metadata file, write it as a class with a ``forward`` method instead. In
chunked resolution, one instance serves all chunks of a stream (see
:doc:`/user_guide/type_resolution/index`).

Formats made of several files, such as a STAR file with its MRC stack, are
described by a ``ProxyGroup`` (see :doc:`/user_guide/proxies`).

Making resolvers visible
------------------------

Resolvers are registered under the module that defines them, and a resolution
only considers resolvers from modules that are visible at the call site: the
calling module, the modules it imports, the module of the value's type, and
the ``scipion_bridge`` package. Import the resolvers module from the package's
``__init__.py`` so that importing the package is enough:

.. code-block:: python

    # scipion_denoise/__init__.py
    from . import resolvers  # noqa: F401  (registers the resolvers)

To make them visible under the package name itself, lift them:

.. code-block:: python

    B.lift_resolvers(resolvers)

When a conversion does not resolve as expected, set the log level to ``INFO``:
every resolution and the resolvers it uses are logged.

Checklist
---------

* The command declaration's body is ``pass`` and its parameters match the
  program's arguments.
* Inputs that are written from memory use ``B.ResolveProxy[P]``, outputs use
  ``= B.Output(P)``.
* Every output proxy is read back into memory before the stage returns.
* Input formats that the runner reads from disk have a slice-aware resolver.
* The resolvers module is imported by the package.

Next, :doc:`writing_protocols` puts these functions into a protocol.
