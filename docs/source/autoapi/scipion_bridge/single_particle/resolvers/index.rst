scipion_bridge.single_particle.resolvers
========================================

.. py:module:: scipion_bridge.single_particle.resolvers


Classes
-------

.. autoapisummary::

   scipion_bridge.single_particle.resolvers.resolve_particle_stack_to_particles


Functions
---------

.. autoapisummary::

   scipion_bridge.single_particle.resolvers.resolve_mrc_stack_proxy
   scipion_bridge.single_particle.resolvers.resolve_starfile_to_mrc_stack
   scipion_bridge.single_particle.resolvers.resolve_starfile_to_particle_stack
   scipion_bridge.single_particle.resolvers.resolve_particle_stack_proxy


Module Contents
---------------

.. py:function:: resolve_mrc_stack_proxy(value: scipion_bridge.core.struct.Set[scipion_bridge.single_particle.particle.Particle]) -> scipion_bridge.single_particle.proxies.MRCStackProxy

   Resolve an MRCStackProxy from a Particle Set.


.. py:function:: resolve_starfile_to_mrc_stack(value: scipion_bridge.single_particle.proxies.StarfileProxy) -> scipion_bridge.single_particle.proxies.MRCStackProxy

   Resolve an MRCStackProxy from a StarfileProxy.


.. py:function:: resolve_starfile_to_particle_stack(value: scipion_bridge.single_particle.proxies.StarfileProxy) -> scipion_bridge.single_particle.proxies.ParticleStackProxy

   Resolve a ParticleStackProxy from a StarfileProxy.


.. py:function:: resolve_particle_stack_proxy(value: scipion_bridge.core.struct.Set[scipion_bridge.single_particle.particle.Particle]) -> scipion_bridge.single_particle.proxies.ParticleStackProxy

   Resolve a ParticleStackProxy from a Particle object.


.. py:class:: resolve_particle_stack_to_particles

   Resolve a Set[Particle] from a ParticleStackProxy.

   When used in iterating resolution, the instance is created once before the
   chunk loop, so the parsed STAR metadata is cached and reused across chunks.


   .. py:method:: forward(value: scipion_bridge.single_particle.proxies.ParticleStackProxy, *, slice: Optional[resolve_particle_stack_to_particles.forward.slice] = None) -> scipion_bridge.core.struct.Set[scipion_bridge.single_particle.particle.Particle]


