scipion_bridge.backend.pyworkflow.workflow_container
====================================================

.. py:module:: scipion_bridge.backend.pyworkflow.workflow_container


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.pyworkflow.workflow_container.convert_scipion_to_python
   scipion_bridge.backend.pyworkflow.workflow_container.configure_pyworkflow_env


Module Contents
---------------

.. py:function:: convert_scipion_to_python(val: Any, dtype: Type) -> Any

   Converts a value retrieved from a PyWorkflow/Scipion Param to its declared Python type.


.. py:function:: configure_pyworkflow_env(backend, *, conda_env: str, configuration: Optional[Union[scipion_bridge.core.protocol.protocol_base._ProtocolTypeConfiguration, scipion_bridge.core.protocol.protocol_base.ProtocolConfiguration, Dict[str, Type]]] = None, modules=None, packages=None)

