Step 1: Modelling the Data
==========================

.. currentmodule:: scipion_bridge

A protocol consumes and produces streams of typed items. Before writing any
processing code, decide what those items are. The types are more than
documentation: they determine how inputs are converted from files and Scipion
objects, how outputs are validated, how batches are split and combined, how
results are persisted, and which Scipion objects a generated form accepts.

This page explains how to choose and define the types of a protocol. The
mechanics of the containers are described in :doc:`/user_guide/structs/index`.

.. contents:: On this page
   :local:
   :depth: 2

Elements and batches
--------------------

A protocol deals with two levels of data:

* an **element**, such as one particle, described by a ``Struct``;
* a **batch** of elements, a ``Set[T]``, which is the item that actually
  travels through the stream.

Streams carry batches rather than single elements because every item that
crosses a process boundary costs a serialization and a transfer. A batch of
thousands of particles costs roughly as much to send as a single particle, so
the per-item overhead is amortized. Operations such as ``chunk`` exist to
control the batch size; the element type stays the same.

Inputs and outputs of protocols are therefore declared as sets:

.. code-block:: python

    particles: B.Input[B.Set[spa.Particle]]

    def outputs(self):
        return {"particles": B.Set[DenoisedParticle]}

A ``Collection[T]`` is the other container that commonly appears as an item:
a fixed number of slots, each holding one ``T``, for example the classes of a
classification.

Start from the built-in types
-----------------------------

``scipion_bridge.single_particle`` defines the types of single particle
analysis. They are already connected to STAR/MRC files and to Scipion objects
through resolvers, so a protocol that uses them can read its inputs from disk
and from Scipion without further work.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Struct
     - Fields
     - Converts to and from
   * - ``Particle``
     - ``pixels`` (2D, dynamic), ``ctf``, ``coordinate``, ``sampling_rate``,
       ``acquisition``
     - STAR + MRC stack, ``SetOfParticles``
   * - ``FlexParticle``
     - ``Particle`` plus ``embeddings`` (1D, dynamic)
     - ``SetOfParticlesFlex``
   * - ``CTF``, ``Coordinate``, ``Acquisition``
     - Nested parts of ``Particle``
     - (as part of ``Particle``)
   * - ``Class2D``
     - ``particles: Set[Particle]``, ``class_id``, ``representative``
     - ``SetOfClasses2D`` (as ``Collection[Class2D]``)

Use these types for inputs whenever possible, and define new types only for
what the method adds.

Defining a new struct
---------------------

The denoising method of the running example produces, for every particle, a
denoised image and an estimated signal-to-noise ratio. The denoised particle
is still a particle, so the output type extends ``Particle``:

.. code-block:: python

    # scipion_denoise/types.py
    import scipion_bridge as B
    from scipion_bridge import single_particle as spa


    class DenoisedParticle(spa.Particle):
        snr: float

A subclass inherits all fields of its base and adds its own. Every
class-level attribute of a struct must be annotated or be a schema object such
as ``B.Array``; a plain assignment raises ``TypeError`` when the class is
defined.

.. important::

   Containers are specialized per element type. ``Set[DenoisedParticle]`` is
   a class of its own and **not** a subclass of ``Set[Particle]``, even though
   ``DenoisedParticle`` is a subclass of ``Particle``. Resolvers written for
   ``Set[Particle]`` therefore do not apply to ``Set[DenoisedParticle]``, and
   a protocol that declares ``Set[DenoisedParticle]`` as an output must return
   exactly that type. :doc:`external_programs` shows how to add resolvers for
   a new type.

Choosing field types
--------------------

Each annotation of a struct becomes one or more columns. The choice of field
type determines how the data is stored, and therefore how fast it can be
sliced, concatenated and serialized.

**Scalars** (``float``, ``int``, ``bool``, NumPy scalar types) become columns
of shape ``(N, 1)``. Use them for per-element metadata such as scores, ids
and physical parameters.

**Arrays** are declared with a fixed rank and a shape per dimension:

.. code-block:: python

    class Volume(B.Struct):
        voxels: B.Array[np.float32] = B.Array(shape=(None, None, None))

    class Pose(B.Struct):
        rotation: B.Array[np.float64] = B.Array(shape=(3, 3))

A dimension given as a number is *static*: every element has that size. A
dimension given as ``None`` is *dynamic*: elements may differ in that
dimension. Prefer static dimensions when the size is known in advance. A
static column is always a single NumPy array, and missing elements read as
zeros. A dynamic column is a NumPy array while all elements share a shape, and
becomes an Awkward Array once they differ or once some are missing (see
:ref:`ragged-arrays`). ``Particle.pixels`` is dynamic so that one type serves
datasets with different box sizes; within a batch, the images usually share
a shape and the column stays a NumPy array.

**Nested structs** group related fields, as ``Particle.ctf`` groups the CTF
parameters. They add no storage overhead; each leaf is still one column.

**Nested sets** (``particles: B.Set[Particle]``) give each element a
variable number of members, such as the particles of a class.

**Nested collections** (``B.Collection[T](size=n)``) give each element a fixed
number of slots.

Named dimensions (``B.Dim``) tie the sizes of several arrays together, for
example the number of corners of a box shared by two arrays.

Designing outputs for streaming
-------------------------------

A streaming protocol emits its outputs in pieces: one dictionary per batch
that reaches the last stage. Whoever consumes those pieces (a sink writing to
Zarr, the Scipion backend appending to an output set) combines them according
to the :ref:`reduction semantics <reduction-semantics>` of the type system:

* a ``Set`` output is **appended**: each emitted batch adds rows;
* a ``Struct`` or plain array output is **replaced**: the last emission wins.

The rules nest. A ``Collection[Class2D]`` emitted repeatedly appends new
particles to each class's ``particles`` set and replaces each class's
``representative``.

Design outputs so that each emission is meaningful under these rules. A
denoising protocol emits the denoised particles of each batch, which are
appended to the output. A classification protocol emits, per batch, the
particles newly assigned to each class together with the updated
representative, rather than the whole class history.

Where to define types
---------------------

Define structs in a module of an importable package, never in a script's
``__main__`` or in a notebook cell:

* Ray workers unpickle batches by importing the element type by module and
  name. The Ray backend adds the working directory and ``src`` to the
  workers' ``PYTHONPATH``; anything else must be installed or on the path.
* Specialized containers such as ``Set[DenoisedParticle]`` are pickled by
  reference to their element type, which only works for importable classes.
* The Scipion backend imports the protocol's module to register the generated
  protocol class.

Checklist
---------

* The input types are built-in types, or types with resolvers from ``Path``
  (and from Scipion objects, if the protocol is to run in Scipion).
* The output types reflect what each batch adds, under the reduction rules.
* Dimensions with a known size are static.
* All structs are defined in an importable module.

Next, :doc:`external_programs` describes how to wrap a command line program
and connect its file formats to these types. If the method is called
in-process, continue with :doc:`writing_protocols`.
