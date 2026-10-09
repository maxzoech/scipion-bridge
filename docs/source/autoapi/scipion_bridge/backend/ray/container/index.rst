scipion_bridge.backend.ray.container
====================================

.. py:module:: scipion_bridge.backend.ray.container


Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.ray.container.configure_ray_container


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.ray.container.RayContainer


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.ray.container.configure_ray_env


Module Contents
---------------

.. py:function:: configure_ray_env(modules=None, packages=None, parameters: Optional[Mapping[str, Any]] = None)

   Configure and wire RayContainer.

   :param parameters: Values of the protocol parameters, served to ``Field.value``.


.. py:class:: RayContainer

   Bases: :py:obj:`dependency_injector.containers.DeclarativeContainer`


   .. py:attribute:: config


   .. py:attribute:: shell_exec


   .. py:attribute:: temp_file_provider


   .. py:attribute:: storage_provider


   .. py:attribute:: parameters


   .. py:attribute:: protocol_config_provider


   .. py:attribute:: resource_provider


   .. py:attribute:: streaming_backend


.. py:data:: configure_ray_container

