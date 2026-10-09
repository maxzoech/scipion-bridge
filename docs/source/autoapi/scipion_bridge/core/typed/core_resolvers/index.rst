scipion_bridge.core.typed.core_resolvers
========================================

.. py:module:: scipion_bridge.core.typed.core_resolvers


Functions
---------

.. autoapisummary::

   scipion_bridge.core.typed.core_resolvers.resolve_any_to_str
   scipion_bridge.core.typed.core_resolvers.resolve_tuple_to_str
   scipion_bridge.core.typed.core_resolvers.resolve_awkward_to_ndarray


Module Contents
---------------

.. py:function:: resolve_any_to_str(value: object) -> str

.. py:function:: resolve_tuple_to_str(value: tuple) -> str

.. py:function:: resolve_awkward_to_ndarray(value: awkward.Array) -> numpy.ndarray

   Resolve an Awkward array directly into a NumPy ndarray.


