Sharp Bits
==========

This page collects behaviors that often surprise new users. Most of them
follow from two facts: **graphs are built once and run elsewhere**, and
**data lives in columns, not in objects**.

.. contents::
   :local:
   :depth: 1

``steps()`` builds a graph; it does not process data
----------------------------------------------------

``steps()`` runs once, on the driver, when the pipeline is compiled. Code in
it does not see any data:

.. code-block:: python

    def steps(self):
        print("hello")                 # printed once, at compile time
        return self.particles.map(self._process)

    def _process(self, batch):
        print(len(batch))              # printed for every batch, on a worker

Put per-batch logic in the functions you pass to ``map``.

Stage functions run in other processes
--------------------------------------

On Ray, ``map`` functions run in worker processes, possibly on other machines.
They and everything they capture (including ``self`` for bound methods) are
pickled with ``cloudpickle``. As a consequence:

* captured objects must be picklable (no open files, locks, or CUDA handles);
* changes to ``self`` or to globals inside a stage are **not** visible to other
  stages or to the driver;
* expensive objects such as models belong in a ``B.Resource``, which is built
  once per process (or once per cluster), not in ``__init__``.

The protocol's last stage must return a dict
--------------------------------------------

The final ``map`` of ``steps()`` must return a ``dict`` whose keys and types
match ``outputs()``. Returning the batch itself raises ``ValidationError``:

.. code-block:: python

    .map(lambda batch: batch)                     # ValidationError
    .map(lambda batch: {"particles": batch})      # OK

Every protocol attribute needs an annotation
--------------------------------------------

.. code-block:: python

    class P(B.Protocol):
        threshold = 0.5             # TypeError at class definition
        threshold: float = 0.5      # OK (state)
        threshold: B.Field[float] = B.Field(default=0.5)   # OK (parameter)

Structs have the same rule.

Scalar columns have a trailing dimension
----------------------------------------

Scalar fields are arrays of shape ``(1,)``, so a column read from a set has
shape ``(N, 1)``, not ``(N,)``:

.. code-block:: python

    particles["sampling_rate"].shape          # (N, 1)
    particles["sampling_rate"].ravel()        # (N,)

Boolean masks must be one-dimensional, so ``ravel()`` before filtering:

.. code-block:: python

    good = particles[particles["ctf"]["resolution"].ravel() < 4.0]

Dynamic dimensions can give Awkward arrays
------------------------------------------

A field declared with ``None`` dimensions is a NumPy array while all its
elements have the same shape. Once they differ, for example after concatenating
sets with images of different sizes, or when some elements were never written,
the field is returned as an Awkward array instead. Code that expects NumPy
should convert explicitly; this works for both types:

.. code-block:: python

    import awkward as ak
    pixels = ak.to_numpy(particles["pixels"])   # raises if the images differ in shape

Views share memory
------------------

Slices, index arrays, masks, row access, and field access all return **views**
of the same storage. Writing through a view changes the original:

.. code-block:: python

    first = particles[:10]
    first["sampling_rate"] = np.ones((10, 1))   # also changes particles

Slices with a step other than 1 (``particles[::2]``) are not supported. Use an
index array instead: ``particles[np.arange(0, len(particles), 2)]``.

``send`` does not accept lists, and ``flatten`` does not accept Sets
--------------------------------------------------------------------

Batches travel between stages as ``Set`` objects, which serialize as Arrow
buffers. Python lists of structs would be pickled object by object and are
rejected by ``Pipeline.send``. For the same reason, ``flatten()`` rejects a
``Set``: use ``chunk(n)`` to change the batch size instead.

``combine_latest`` drops items at the end of the stream
-------------------------------------------------------

Items that arrive before the other stream has produced a value are buffered.
If the other stream never produces one (for example because ``collect(n)``
received fewer than ``n`` elements and its map failed), the buffered items are
dropped at ``FLUSH``. Make sure the side stream always emits.

``chunk`` emits a partial last batch
------------------------------------

On ``FLUSH``, ``chunk(n)`` emits the remainder as a smaller set. Models
compiled for a fixed batch size (``torch.compile``, ``jax.jit``) may recompile
for it. Pass ``drop_last=True`` to discard it, or pad it in your map.

The Ray runner discards outputs without a sink
----------------------------------------------

``RayPipelineRunner`` validates the protocol's outputs but does nothing else
with them unless you pass ``sink=``. To keep the results, pass a
``SinkWriter`` or a callback.

The sink receives the protocol's output **dictionaries**, not the containers
in them. ``TensorStoreSinkWriter`` writes single containers and rejects a
dictionary with ``TypeError``; wrap it in a writer that dispatches each output,
as shown in :ref:`ray-persisting-outputs`.

Branches that do not reach a sink do not run
--------------------------------------------

A pipeline consists of the nodes upstream of the sinks it is compiled from.
A branch of ``steps()`` that does not lead to the returned node, such as a
``map`` whose result is unused, is never compiled. The same applies to
``checkpoint(writer)``: its sink is downstream of the stream, so it runs only
if it is passed to ``Pipeline.from_sink`` explicitly. Protocol runners compile
only the protocol's output, so a ``checkpoint`` inside ``steps()`` has no
effect.

GPU calls of ephemeral protocols run in fresh processes
-------------------------------------------------------

With the default ``TaskType.EPHEMERAL``, every map call that claims a GPU runs
in a new worker process, because CUDA binds a process to its first GPU. A
``B.Resource`` with the default process scope is therefore built again for
every such call: a model is loaded for every batch. Declare protocols that
keep a model on the GPU with ``task=B.TaskType.LONG_RUNNING``.

``Set[Sub]`` is not a ``Set[Base]``
-----------------------------------

Containers are specialized per element type. ``Set[FlexParticle]`` is a class
of its own, not a subclass of ``Set[Particle]``. Resolvers written for
``Set[Particle]`` do not apply to it, and an output declared as one cannot be
satisfied by the other.

Resolution is by type, not by file name
---------------------------------------

Type resolution chooses its path from the types involved. A ``Path`` is a
``Path``, whatever its extension, so when two formats both resolve from
``Path`` to the same type, the path the Ray runner precomputes for an input is
the same for every file. Give such inputs as proxies of the intended format,
with ``origin_types`` (see :doc:`/integration/external_programs`).

Command arguments are not quoted
--------------------------------

``@B.shell_command`` joins the arguments with spaces and runs the command
through a shell. Values with spaces or shell metacharacters are split or
interpreted by the shell.

Fields with defaults are required on the command line
-----------------------------------------------------

``B.Field(default=...)`` makes a field non-optional, and the command line
parser generated by ``launch_as_terminal_application`` treats non-optional
fields as required options, even when they have a default. Pass
``optional=True`` to make such an option optional.

Protocols are sealed once their pipeline is built
-------------------------------------------------

``get_pipeline()`` runs ``validate_protocol_configuration()``, ``setup()`` and
``steps()``, then seals the protocol: setting or deleting an attribute raises
``B.SealedProtocolError``, and ``list``, ``dict``, ``set`` and ``bytearray``
attributes are rejected. The functions of the stages run as copies of the
protocol in other processes, so state kept in a map would differ between the
copies. State across items belongs in a ``.scan`` stage, expensive objects in
a ``B.Resource`` (see :ref:`where-state-belongs`).

The seal does not see inside other objects, class attributes or module
globals. Changing those in a stage is just as wrong, but not caught. A
protocol builds its pipeline once; create a new instance to build another.

The Ray backend turns off NumPy's transparent hugepages
-------------------------------------------------------

On Linux, NumPy asks the kernel to back every array of 4 MiB or more with
transparent hugepages. On a long-running, shared host this can stall a
pipeline progressively: stages moving large stacks of particles slow from
about a second to minutes per item, memory pressure (``full`` in the cgroup's
``memory.pressure``) is high, and ``compact_stall`` keeps growing in
``/proc/vmstat``.

The cause is the kernel setting ``defrag=madvise``
(``/sys/kernel/mm/transparent_hugepage/defrag``) combined with fragmented
physical memory. A page fault in such an array makes the kernel compact memory
synchronously to build a 2 MiB page. When the Ray object store in
``/dev/shm``, the page cache and pinned CUDA memory have fragmented the RAM,
``/proc/buddyinfo`` shows no free 2 MiB blocks, and most compactions fail
after stalling the process.

The Ray backend therefore sets ``NUMPY_MADVISE_HUGEPAGE=0`` in the
``runtime_env`` of the Ray job and of every actor and task it creates. In the
driver, where NumPy is already imported, ``RayBackend`` switches the hint off
at runtime. The standalone backend leaves NumPy unchanged.

To keep hugepages, set the variable before starting the driver; the backend
passes its value on to every worker:

.. code-block:: bash

    NUMPY_MADVISE_HUGEPAGE=1 python run_pipeline.py

The pickled ``Set[T]`` needs an importable ``T``
------------------------------------------------

Sets are pickled with a reference to their class. The standard ``pickle``
module cannot find element types that were defined in ``__main__`` or in a
notebook. Define your structs in a module, or use ``cloudpickle``, which Ray
uses automatically.

Shell command bodies must be ``pass``
-------------------------------------

A function decorated with ``@B.shell_command`` is an interface. A docstring,
a ``return``, or any other statement in its body raises ``RuntimeError`` when
the function is defined. Keep documentation in comments above the function.
