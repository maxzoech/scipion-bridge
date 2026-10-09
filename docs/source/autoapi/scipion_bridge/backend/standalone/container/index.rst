scipion_bridge.backend.standalone.container
===========================================

.. py:module:: scipion_bridge.backend.standalone.container


Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.standalone.container.default_streaming_backend


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.standalone.container.Container


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.standalone.container.configure_default_env


Module Contents
---------------

.. py:data:: default_streaming_backend

.. py:class:: Container

   Bases: :py:obj:`dependency_injector.containers.DeclarativeContainer`


   .. py:attribute:: config


   .. py:attribute:: shell_exec


   .. py:attribute:: temp_file_provider


   .. py:attribute:: storage_provider


   .. py:attribute:: protocol_config_provider


   .. py:attribute:: resource_provider


   .. py:attribute:: streaming_backend


.. py:function:: configure_default_env(modules=None, packages=None)

