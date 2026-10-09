scipion_bridge.backend.pyworkflow.scipion3_protocol
===================================================

.. py:module:: scipion_bridge.backend.pyworkflow.scipion3_protocol


Attributes
----------

.. autoapisummary::

   scipion_bridge.backend.pyworkflow.scipion3_protocol.HAS_PWEM


Functions
---------

.. autoapisummary::

   scipion_bridge.backend.pyworkflow.scipion3_protocol.reduce_minibatch_to_persistent_output
   scipion_bridge.backend.pyworkflow.scipion3_protocol.convert_protocol_to_scipion3_protocol


Module Contents
---------------

.. py:data:: HAS_PWEM
   :value: True


.. py:function:: reduce_minibatch_to_persistent_output(protocol: Any, key: str, minibatch_obj: Any) -> Any

   Reduce a stateless minibatch Scipion object into the protocol's persistent on-disk output set.


.. py:function:: convert_protocol_to_scipion3_protocol(protocol: scipion_bridge.core.protocol.Protocol, *, label: str, conda_env: str)

