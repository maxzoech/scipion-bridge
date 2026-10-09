scipion_bridge.core.typed.array
===============================

.. py:module:: scipion_bridge.core.typed.array


Classes
-------

.. autoapisummary::

   scipion_bridge.core.typed.array.ArrayConvertable


Functions
---------

.. autoapisummary::

   scipion_bridge.core.typed.array.resolve_output_to_proxy


Module Contents
---------------

.. py:class:: ArrayConvertable

   .. py:method:: to_numpy()
      :abstractmethod:



   .. py:method:: from_numpy(data: numpy.ndarray)
      :classmethod:

      :abstractmethod:



.. py:function:: resolve_output_to_proxy(value: numpy.ndarray, cls: Type[ArrayConvertable]) -> ArrayConvertable

