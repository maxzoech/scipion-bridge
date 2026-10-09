scipion_bridge.backend.pyworkflow
=================================

.. py:module:: scipion_bridge.backend.pyworkflow


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/backend/pyworkflow/resolvers/index
   /autoapi/scipion_bridge/backend/pyworkflow/scipion3_protocol/index
   /autoapi/scipion_bridge/backend/pyworkflow/utils/index
   /autoapi/scipion_bridge/backend/pyworkflow/workflow_container/index


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.pyworkflow.configure_pyworkflow_env
   scipion_bridge.backend.pyworkflow.convert_protocol_to_scipion3_protocol


Package Contents
----------------

.. py:function:: configure_pyworkflow_env(backend, *, conda_env: str, configuration: Optional[Union[scipion_bridge.core.protocol.protocol_base._ProtocolTypeConfiguration, scipion_bridge.core.protocol.protocol_base.ProtocolConfiguration, Dict[str, Type]]] = None, modules=None, packages=None)

.. py:function:: convert_protocol_to_scipion3_protocol(protocol: scipion_bridge.core.protocol.Protocol, *, label: str, conda_env: str)

