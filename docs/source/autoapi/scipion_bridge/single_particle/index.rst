scipion_bridge.single_particle
==============================

.. py:module:: scipion_bridge.single_particle


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/single_particle/particle/index
   /autoapi/scipion_bridge/single_particle/proxies/index
   /autoapi/scipion_bridge/single_particle/resolvers/index


Classes
-------

.. autoapisummary::

   scipion_bridge.single_particle.Particle
   scipion_bridge.single_particle.FlexParticle
   scipion_bridge.single_particle.CTF
   scipion_bridge.single_particle.Coordinate
   scipion_bridge.single_particle.Class2D
   scipion_bridge.single_particle.Acquisition
   scipion_bridge.single_particle.ParticleStackProxy
   scipion_bridge.single_particle.StarfileProxy
   scipion_bridge.single_particle.MRCStackProxy


Package Contents
----------------

.. py:class:: Particle(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.struct.Struct`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: pixels
      :type:  scipion_bridge.core.struct.Array[numpy.float32]


   .. py:attribute:: ctf
      :type:  CTF


   .. py:attribute:: coordinate
      :type:  Coordinate


   .. py:attribute:: sampling_rate
      :type:  float


   .. py:attribute:: acquisition
      :type:  Acquisition


.. py:class:: FlexParticle(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`Particle`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: embeddings
      :type:  scipion_bridge.core.struct.Array[float]


.. py:class:: CTF(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.struct.Struct`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: defocus_u
      :type:  float


   .. py:attribute:: defocus_v
      :type:  float


   .. py:attribute:: defocus_angle
      :type:  float


   .. py:attribute:: phase_shift
      :type:  float


   .. py:attribute:: resolution
      :type:  float


   .. py:attribute:: fit_quality
      :type:  float


.. py:class:: Coordinate(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.struct.Struct`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: x
      :type:  float


   .. py:attribute:: y
      :type:  float


.. py:class:: Class2D(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.struct.Struct`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: particles
      :type:  scipion_bridge.core.struct.Set[Particle]


   .. py:attribute:: class_id
      :type:  int


   .. py:attribute:: representative
      :type:  Particle


.. py:class:: Acquisition(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.struct.Struct`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: magnification
      :type:  float


   .. py:attribute:: voltage
      :type:  float


   .. py:attribute:: spherical_aberration
      :type:  float


   .. py:attribute:: amplitude_contrast
      :type:  float


   .. py:attribute:: dose_initial
      :type:  float


   .. py:attribute:: dose_per_frame
      :type:  float


.. py:class:: ParticleStackProxy(base_path: os.PathLike, managed: bool = False, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.typed.proxy.ProxyGroup`


   Abstract base class representing a grouped collection of Proxy objects.

   Subclasses must annotate child proxy fields with Proxy types and implement
   the primary_proxy abstract property.


   .. py:attribute:: metadata
      :type:  StarfileProxy


   .. py:attribute:: particle_stack
      :type:  MRCStackProxy


   .. py:property:: primary_proxy
      :type: scipion_bridge.core.typed.proxy.Proxy


      Return the primary proxy instance for this group.


   .. py:property:: num_particles
      :type: int



   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated total size in bytes for a single logical item across all child proxies.

      Returns the sum of all child proxy estimates, or None if any child cannot estimate its size.


.. py:class:: StarfileProxy(path: os.PathLike, managed=False, *args, **kwargs)

   Bases: :py:obj:`scipion_bridge.core.typed.proxy.Proxy`


   .. py:method:: read() -> Any


   .. py:method:: file_ext()
      :classmethod:



.. py:class:: MRCStackProxy(path: os.PathLike, managed: bool = False, metadata_path: Optional[os.PathLike] = None, *args, **kwargs)

   Bases: :py:obj:`scipion_bridge.core.typed.proxy.Proxy`


   .. py:attribute:: metadata_path
      :type:  Optional[pathlib.Path]


   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated size in bytes for a single logical item contained in this proxy.

      Returns None if size cannot be determined without reading the data.


   .. py:method:: file_ext() -> Optional[str]
      :classmethod:



   .. py:method:: extensions() -> Optional[tuple[str, Ellipsis]]
      :classmethod:



   .. py:method:: find_mrc_path_from_star(star_path: pathlib.Path) -> pathlib.Path
      :classmethod:


      Find the MRC data file referenced by a .star file.



