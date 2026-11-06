Declarations
============

.. currentmodule:: scipion_bridge

Everything a backend needs to know about a protocol without running it is
declared as annotated class attributes. This article describes each kind of
declaration, how backends use it, and the validation of outputs.

.. contents:: On this page
   :local:
   :depth: 2

How annotations are classified
------------------------------

When a protocol class is defined, every annotated attribute is classified by
its annotation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Annotation
     - Kind
   * - ``B.Input[T]``
     - An input stream of items of type ``T``.
   * - ``B.Field[T]``
     - A user-facing parameter of type ``T``.
   * - ``B.Resource[T]``
     - Lazily built, read-only state of type ``T``.
   * - anything else
     - Plain instance state.

Every class-level attribute of a protocol must be annotated. ``state = 42``
raises ``TypeError`` when the class is defined. The declaration order of the
annotations is kept and used for the order of form fields and command line
options.

Inputs
------

.. code-block:: python

    particles: B.Input[B.Set[spa.Particle]]
    particles: B.Input[B.Set[spa.Particle]] = B.Input(label="Particles", help="...")

An ``Input[T]`` declares a stream of items of type ``T``. On an instance,
``self.particles`` is a ``Source`` named after the attribute, so operations
are chained onto it directly. All accesses to the same input refer to the same
source, which fans out to every branch.

``B.Input(...)`` accepts ``default``, ``optional``, ``label`` and ``help``.
The label and help text appear in the Scipion form and in ``--help``.

The input type determines where data can come from. The Ray runner resolves
inputs from files, which requires a resolution path from ``Path`` to ``T``
with a slice-aware step; the Scipion backend requires a resolver from a
Scipion object to ``T`` (see :doc:`/integration/deployment`).

Fields
------

.. code-block:: python

    chunk_size: B.Field[int]
    chunk_size: B.Field[int] = B.Field(default=256, label="Chunk size", help="...")

A ``Field[T]`` is a parameter. Its value is read with ``.value``:

.. code-block:: python

    self.chunk_size.value

Values are supplied by the backend: by ``RayPipelineRunner(parameters=...)``,
by command line arguments, or by the Scipion form. A parameter without a value
falls back to its default. ``.value`` works on the driver inside ``steps()``
and inside every stage, because each backend serves the parameter values to
its worker processes. It does not work at import time or in ``__init__``,
before a backend has provided the values.

``B.Field(...)`` accepts ``default``, ``optional``, ``label``, ``group`` and
``help``. A field declared by its annotation alone is required. With
``B.Field(...)``, ``optional`` defaults to ``True`` when no default is given,
and to ``False`` otherwise. The Ray runner refuses to compile a protocol whose
required fields have neither a value nor a default.

Supported types and their interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Type
     - Scipion form
     - Command line
   * - ``int``, ``float``, ``str``
     - Int / Float / String parameter
     - ``--name VALUE``
   * - ``bool``
     - Boolean parameter
     - ``--name`` / ``--no-name``
   * - ``Enum`` subclass
     - Drop-down of the enum *values*
     - ``--name MEMBER`` (choices are the member *names*)

On the command line, field names are converted to kebab case
(``chunk_size`` becomes ``--chunk-size``). A field that is not optional is a
required option, even when it has a default.

Resources
---------

.. code-block:: python

    model: B.Resource[torch.nn.Module] = B.Resource(
        builder=load_model,
        scope=B.ResourceScope.PROCESS,
    )

A ``Resource[T]`` is **read-only state** that is expensive to create: network
weights, large lookup tables, thread pools. ``builder`` is called with the
protocol instance the first time the attribute is read, and the result is
cached. Because the builder receives the instance, it can read fields:

.. code-block:: python

    def load_model(protocol: "FoundationModelInference") -> torch.nn.Module:
        return load_weights(protocol.model_type.value).eval().cuda()

The scope decides where the cache lives:

``ResourceScope.PROCESS`` (default)
   Built once per worker process, the first time a stage on that worker reads
   it. Note that the process of a call depends on the compute requirements:
   GPU calls of ephemeral protocols each run in a fresh process (see
   :doc:`compute`).

``ResourceScope.SHARED``
   Built once for the **whole Ray cluster**, by a task holding a share of the
   cluster's CPU cores while it builds, and placed in Ray's object store.
   Every worker fetches the same object. Use it for large, immutable data
   such as a reference volume.

If the annotation names a class, the built value is checked against it.
Assigning to a resource raises ``AttributeError``. Resources need a backend
that provides a resource provider; the standalone and Ray backends do.

State
-----

Any other annotated class attribute is plain instance state:

.. code-block:: python

    tracker_handle: Any = None

State is configuration that is not a user-facing parameter. Set it in
``__init__()``, ``setup()`` or ``steps()``; once ``get_pipeline()`` has built
the pipeline, the protocol is **sealed** and its attributes cannot change.
Setting or deleting an attribute of a sealed protocol raises
``B.SealedProtocolError``.

The functions of the stages run as copies of the protocol, pickled into the
worker processes together with the bound methods used in ``map``, and with
executor pools as several copies at once. A change made by one call would not
be seen by the other calls, so sealing turns it into an error. State must
therefore be picklable and immutable. When the protocol is sealed:

* a ``list``, ``dict``, ``set`` or ``bytearray`` attribute, also inside a
  tuple, raises ``B.SealedProtocolError``. Use a ``tuple``, a ``frozenset``, a
  tuple of ``(key, value)`` pairs or a frozen dataclass instead.
  ``types.MappingProxyType`` does not work: it cannot be pickled;
* NumPy arrays are made read-only, also in the copies of the workers;
* other objects are accepted as they are. The seal cannot see inside them, so
  do not change them in a stage either.

.. _where-state-belongs:

Where state belongs
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - State
     - Where it belongs
   * - Accumulated over items: counts, running means, the best class so far
     - A ``.scan(func, initial=..., on_flush=...)`` stage, which folds the
       stream (see :doc:`../streaming/stateful_operations`).
   * - Expensive to build: models, compiled programs, lookup tables
     - A ``B.Resource(builder=...)``, built once per process or cluster.
   * - Configuration
     - An attribute set in ``__init__()``, ``setup()`` or ``steps()``, or a
       ``B.Field`` if users should set it.

A count kept in a map is the typical mistake:

.. code-block:: python

    # Wrong: raises SealedProtocolError on the first item. Without the seal,
    # every copy of the protocol would count only the items it handled.
    def _count(self, particles):
        self.seen += len(particles)
        return particles

Fold the counts into a stage of their own instead. ``scan`` emits the running
total after every item; ``on_flush=`` would emit one more item at the end of
the stream:

.. code-block:: python

    def steps(self):
        return (
            self.particles.map(len)
            .scan(lambda total, n: total + n, initial=lambda: 0)
            .map(lambda total: {"count": total})
        )

Python cannot stop a deliberate bypass of the seal, through
``object.__setattr__``, ``__dict__``, class attributes or module globals. Do
not bypass it: the results would depend on which copy of the protocol handled
an item.

Outputs and validation
----------------------

``outputs()`` maps output names to types:

.. code-block:: python

    def outputs(self):
        return {"particles": B.Set[spa.FlexParticle]}

The last stage of ``steps()`` must return a dictionary. Every dictionary is
validated by a final map that the protocol appends to its graph, before it
reaches the sink:

* a result that is not a dictionary raises ``ValidationError``;
* a key that is not declared raises ``ValidationError``;
* a value that is not an instance of the declared type raises
  ``ValidationError``.

The type check is exact for specialized containers: a ``Set[Particle]`` does
not pass as ``Set[FlexParticle]``, nor the reverse. Not every declared output
has to be present in every dictionary, which allows a protocol to emit
different outputs at different times.

Hooks
-----

``Protocol`` defines two optional hooks, ``setup()`` and
``validate_protocol_configuration()``. ``get_pipeline()`` runs them on every
backend, in this order and before ``steps()``, all on the driver before any
data flows. ``setup()`` may set attributes; afterwards ``steps()`` builds the
pipeline and the protocol is sealed (see `State`_). The pipeline of a
protocol can be built only once: create a new instance to build another one.
