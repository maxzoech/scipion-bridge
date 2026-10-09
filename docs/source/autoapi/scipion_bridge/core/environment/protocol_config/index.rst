scipion_bridge.core.environment.protocol_config
===============================================

.. py:module:: scipion_bridge.core.environment.protocol_config


Classes
-------

.. autoapisummary::

   scipion_bridge.core.environment.protocol_config.ProtocolConfigurationProvider
   scipion_bridge.core.environment.protocol_config.StaticProtocolConfigurationProvider


Module Contents
---------------

.. py:class:: ProtocolConfigurationProvider

   Bases: :py:obj:`abc.ABC`


   Abstract provider for managing field values of Protocol instances.


   .. py:method:: get_value(name: str, default: Any = None) -> Any
      :abstractmethod:


      Retrieve the value of a field for a given protocol instance.



.. py:class:: StaticProtocolConfigurationProvider(values: Mapping[str, Any])

   Bases: :py:obj:`ProtocolConfigurationProvider`


   Provider serving field values from a fixed mapping.

   Fields without an entry in the mapping resolve to their declared default.


   .. py:attribute:: values


   .. py:method:: get_value(name: str, default: Any = None) -> Any

      Retrieve the value of a field for a given protocol instance.



