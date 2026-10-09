scipion_bridge.backend
======================

.. py:module:: scipion_bridge.backend


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/backend/pyworkflow/index
   /autoapi/scipion_bridge/backend/ray/index
   /autoapi/scipion_bridge/backend/standalone/index


Classes
-------

.. autoapisummary::

   scipion_bridge.backend.Container


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.configure_default_env
   scipion_bridge.backend.configure_pyworkflow_env


Package Contents
----------------

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

.. py:function:: configure_pyworkflow_env(backend, *, conda_env: str, configuration: Optional[Union[scipion_bridge.core.protocol.protocol_base._ProtocolTypeConfiguration, scipion_bridge.core.protocol.protocol_base.ProtocolConfiguration, Dict[str, Type]]] = None, modules=None, packages=None)

