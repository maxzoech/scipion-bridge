scipion_bridge.core.environment.resource_provider
=================================================

.. py:module:: scipion_bridge.core.environment.resource_provider


Classes
-------

.. autoapisummary::

   scipion_bridge.core.environment.resource_provider.ResourceScope
   scipion_bridge.core.environment.resource_provider.ResourceProvider
   scipion_bridge.core.environment.resource_provider.DefaultResourceProvider


Module Contents
---------------

.. py:class:: ResourceScope

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   Lifecycle and sharing scope for protocol resources.


   .. py:attribute:: PROCESS
      :value: 'process'



   .. py:attribute:: SHARED
      :value: 'shared'



.. py:class:: ResourceProvider

   Bases: :py:obj:`abc.ABC`


   Abstract provider for managing lifecycle and caching of Protocol resources.


   .. py:method:: get_resource(name: str, builder: Callable[[Any], Any], instance: Any, scope: ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None) -> Any
      :abstractmethod:


      Retrieve or build a resource for a given protocol instance.



.. py:class:: DefaultResourceProvider

   Bases: :py:obj:`ResourceProvider`


   Process-local provider that lazily constructs resources and caches them.


   .. py:method:: get_resource(name: str, builder: Callable[[Any], Any], instance: Any, scope: ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None) -> Any

      Retrieve or build a resource for a given protocol instance.



