scipion_bridge.single_particle.proxies
======================================

.. py:module:: scipion_bridge.single_particle.proxies


Classes
-------

.. autoapisummary::

   scipion_bridge.single_particle.proxies.StarfileProxy
   scipion_bridge.single_particle.proxies.MRCStackProxy
   scipion_bridge.single_particle.proxies.ParticleStackProxy


Module Contents
---------------

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


