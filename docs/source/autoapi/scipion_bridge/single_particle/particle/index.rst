scipion_bridge.single_particle.particle
=======================================

.. py:module:: scipion_bridge.single_particle.particle


Classes
-------

.. autoapisummary::

   scipion_bridge.single_particle.particle.CTF
   scipion_bridge.single_particle.particle.Coordinate
   scipion_bridge.single_particle.particle.Acquisition
   scipion_bridge.single_particle.particle.Particle
   scipion_bridge.single_particle.particle.FlexParticle
   scipion_bridge.single_particle.particle.Class2D


Module Contents
---------------

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


.. py:class:: Class2D(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.struct.Struct`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:attribute:: particles
      :type:  scipion_bridge.core.struct.Set[Particle]


   .. py:attribute:: class_id
      :type:  int


   .. py:attribute:: representative
      :type:  Particle


