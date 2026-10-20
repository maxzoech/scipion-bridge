Single Particle Types
=====================

.. currentmodule:: scipion_bridge.single_particle

``scipion_bridge.single_particle`` (imported by convention as ``spa``) defines
the types of single particle analysis, the proxies for their file formats,
and the resolvers that connect them. It is both a ready-to-use domain module
and the reference for how a domain is built from the parts of the library.

.. code-block:: python

    from scipion_bridge import single_particle as spa

.. contents:: On this page
   :local:
   :depth: 2

Structs
-------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Struct
     - Fields
   * - ``Particle``
     - ``pixels`` (2D float32, dynamic), ``ctf: CTF``,
       ``coordinate: Coordinate``, ``sampling_rate``,
       ``acquisition: Acquisition``
   * - ``FlexParticle``
     - All fields of ``Particle``, plus ``embeddings`` (1D, dynamic)
   * - ``CTF``
     - ``defocus_u``, ``defocus_v``, ``defocus_angle``, ``phase_shift``,
       ``resolution``, ``fit_quality``
   * - ``Coordinate``
     - ``x``, ``y``
   * - ``Acquisition``
     - ``magnification``, ``voltage``, ``spherical_aberration``,
       ``amplitude_contrast``, ``dose_initial``, ``dose_per_frame``
   * - ``Class2D``
     - ``particles: Set[Particle]``, ``class_id``, ``representative: Particle``

All scalar fields are ``float64``, except ``class_id``, which is an integer.
Fields are filled only when the source provides them; check with
``is_initialized`` before relying on optional metadata such as the CTF.

Proxies
-------

The proxies describe the RELION-style file formats for particle stacks:

``StarfileProxy``
   A ``.star`` metadata file. ``read()`` parses it with ``starfile`` and caches
   the result; ``len()`` is the number of particles.

``MRCStackProxy``
   An MRC image stack (``.mrcs``, also accepts ``.mrc``). ``len()`` and
   ``estimated_item_nbytes`` read only the header. Constructed from a ``.star``
   path, it finds the stack through the ``rlnImageName`` column.
   ``read_images(selection)`` returns a contiguous range of images as
   ``float32``.

``SpiderStackProxy``
   A little-endian SPIDER stack of 2D images (``.stk``), as written by Xmipp.
   ``len()``, ``estimated_item_nbytes`` and ``read_images(selection)`` work
   as for ``MRCStackProxy``. A file whose size does not match its header,
   e.g. a big-endian one, raises ``ValueError``.

Both stack proxies read a range of images with one sequential read of the
file. Copying from a memory map instead faults the pages in one by one,
which reads large stacks on spinning disks at a fraction of their speed.

``ParticleStackProxy``
   A ``ProxyGroup`` of a ``StarfileProxy`` (``metadata``, the primary proxy)
   and an ``MRCStackProxy`` (``particle_stack``) sharing a base path. Programs
   receive the path of the STAR file.

These three are also exported from the top-level namespace as
``B.StarfileProxy``, ``B.MRCStackProxy`` and ``B.ParticleStackProxy``.

Particles extracted by Xmipp, for example by ``XmippProtExtractParticles`` in
Scipion, come as one ``<stem>.mrcs`` stack and one ``<stem>.xmd`` metadata
file per micrograph:

``XmdProxy``
   An ``.xmd`` file: a STAR file with Xmipp labels (``_ctfDefocusU``,
   ``_xcoor``, ...). It reads like a ``StarfileProxy``.

``XmippParticleStackProxy``
   A ``ProxyGroup`` of an ``XmdProxy`` (``metadata``, the primary proxy) and
   an ``MRCStackProxy`` (``particle_stack``) sharing a base path. The
   ``_image`` column names the stack relative to the Scipion project, so the
   stack is found by its stem instead.

``XmippSpiderParticleStackProxy``
   The same for Xmipp stacks stored as SPIDER: ``<stem>.xmd`` next to
   ``<stem>.stk``.

Resolvers
---------

The module registers these conversions:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Conversion
     - Behavior
   * - ``Set[Particle]`` → ``ParticleStackProxy``
     - Writes a temporary MRC stack and a STAR file with the image names and
       the initialized metadata columns (coordinates, CTF, acquisition, pixel
       size).
   * - ``Set[Particle]`` → ``MRCStackProxy``
     - Writes the images only.
   * - ``ParticleStackProxy`` → ``Set[Particle]``
     - Slice-aware. Memory-maps the stack and reads the requested range of
       images, and the matching rows of the STAR file, including values from
       its optics block. Caches the parsed STAR file across chunks.
   * - ``StarfileProxy`` → ``ParticleStackProxy``, ``MRCStackProxy``
     - Locates the stack referenced by the STAR file.
   * - ``XmippParticleStackProxy``, ``XmippSpiderParticleStackProxy`` →
       ``Set[Particle]``
     - Slice-aware. Reads the requested range of images and fills the CTF
       (``_ctfDefocusU/V/Angle``, ``_ctfPhaseShift``, ``_ctfCritMaxFreq`` as
       resolution, ``_ctfCritFitting`` as fit quality), the coordinates
       (``_xcoor``, ``_ycoor``), the acquisition (``_ctfVoltage``,
       ``_ctfSphericalAberration``, ``_ctfQ0`` as amplitude contrast,
       ``_magnification``) and the pixel size (``_samplingRate``,
       ``_ctfSamplingRate``) from the labels present. Drops the particles
       with ``_enabled`` other than 1, and raises ``ValueError`` if the
       ``_image`` indices are not in stack order.
   * - ``XmdProxy`` → ``XmippParticleStackProxy``,
       ``XmippSpiderParticleStackProxy``
     - Pairs the metadata with the stack of the same stem.

Together with the resolver that every proxy type gets from ``Path``, these
give a slice-aware path from a STAR file on disk to ``Set[Particle]``, which
is what lets the Ray runner stream particle inputs in chunks.

``ParticleStackProxy`` and both Xmipp proxies lead from ``Path`` to
``Set[Particle]``. The Xmipp resolvers are registered with ``weight=1``
(see ``@resolver``), so without further information a path is read as a
RELION STAR file. Read ``.xmd`` files by resolving through the Xmipp proxy
that matches the stack format:

.. code-block:: python

    resolver = B.find_resolver(
        Path,
        B.Set[spa.Particle],
        intermediate=spa.XmippParticleStackProxy,
    )
    particles = resolver(Path("extra/micrograph_fractions.xmd"))

Scipion keeps the pixel size, magnification and dose of extracted particles
in the set (``particles.sqlite``), not in the ``.xmd`` files, so these fields
stay unset.

With the ``pyworkflow`` extra, further resolvers convert between these types
and Scipion objects: ``SetOfParticles`` and ``Set[Particle]``,
``SetOfParticlesFlex`` and ``Set[FlexParticle]``, ``SetOfClasses2D`` and
``Collection[Class2D]``.

Using the module as a template
------------------------------

A new domain, for example tomography or atomic models, follows the same
structure: structs in one module, proxies for its file formats in another,
resolvers in a third, and an ``__init__.py`` that imports the resolvers and
lifts them into the package with ``lift_resolvers``. See
:doc:`/integration/external_programs` for the individual steps.
