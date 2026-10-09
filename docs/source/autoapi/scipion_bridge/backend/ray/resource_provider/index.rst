scipion_bridge.backend.ray.resource_provider
============================================

.. py:module:: scipion_bridge.backend.ray.resource_provider


Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.ray.resource_provider.SHARED_BUILD_CPU_FRACTION


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.resource_provider.RayResourceCoordinator
   scipion_bridge.backend.ray.resource_provider.RayResourceProvider


Module Contents
---------------

.. py:data:: SHARED_BUILD_CPU_FRACTION
   :value: 0.35


.. py:class:: RayResourceCoordinator

   Cluster-wide coordinator managing shared ObjectRefs in Ray Plasma store.

   Every resource is built once, by a task holding a share of the cluster's
   CPU cores only while it builds; the coordinator itself holds none.


   .. py:method:: get_or_build(key: str, builder: Callable[[Any], Any], instance: Any) -> ray.ObjectRef


   .. py:method:: clear() -> None


.. py:class:: RayResourceProvider

   Bases: :py:obj:`scipion_bridge.core.environment.resource_provider.ResourceProvider`


   Ray-aware resource provider supporting worker-local and cluster-shared resources.


   .. py:method:: clear() -> None

      Clear the worker-local resource cache.



   .. py:method:: reset_coordinator() -> None
      :classmethod:


      Reset the cluster coordinator's cached object references if active.



   .. py:method:: get_resource(name: str, builder: Callable[[Any], Any], instance: Any, scope: scipion_bridge.core.environment.resource_provider.ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None) -> Any

      Retrieve or build a resource for a given protocol instance.



