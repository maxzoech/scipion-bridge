scipion_bridge.core.environment.storage
=======================================

.. py:module:: scipion_bridge.core.environment.storage

.. autoapi-nested-parse::

   Storage providers for schema-driven Struct and Set data containers.



Classes
-------

.. autoapisummary::

   scipion_bridge.core.environment.storage.NumPyStorageArray
   scipion_bridge.core.environment.storage.NumPyStorageGroup
   scipion_bridge.core.environment.storage.ArrayStorageProvider
   scipion_bridge.core.environment.storage.ArrowStorageProvider
   scipion_bridge.core.environment.storage.NumPyStorageProvider


Module Contents
---------------

.. py:class:: NumPyStorageArray(shape: Tuple[int, Ellipsis], dtype: Any, data: Optional[numpy.ndarray] = None)

   Storage array backed directly by numpy.ndarray.


   .. py:attribute:: attrs
      :type:  Dict[str, Any]


   .. py:property:: shape
      :type: Tuple[int, Ellipsis]



   .. py:property:: ndim
      :type: int



   .. py:property:: dtype
      :type: Any



   .. py:property:: size
      :type: int



.. py:class:: NumPyStorageGroup

   Group container mimicking a zarr.Group interface backed by NumPy storage.


   .. py:method:: create_dataset(name: str, shape: Tuple[int, Ellipsis], dtype: Any) -> NumPyStorageArray


   .. py:attribute:: create_array


   .. py:method:: debug_print() -> None

      Print all stored arrays along with their shapes and data types.



.. py:class:: ArrayStorageProvider

   Bases: :py:obj:`abc.ABC`


   Abstract base class for array storage backend providers.


   .. py:method:: create_group(shape_prefix: Tuple[int, Ellipsis] = ()) -> Any
      :abstractmethod:


      Create and return a new array storage group instance.



   .. py:method:: concat(arrays: Sequence[Any], axis: int = 0) -> Any
      :abstractmethod:


      Concatenate a sequence of backend storage arrays along an axis.



.. py:class:: ArrowStorageProvider

   Bases: :py:obj:`ArrayStorageProvider`


   Array storage provider backed by Apache Arrow and NumPy staging structures.


   .. py:method:: create_group(shape_prefix: Tuple[int, Ellipsis] = ()) -> Any

      Create and return a new array storage group instance.



   .. py:method:: concat(arrays: Sequence[Any], axis: int = 0) -> Any

      Concatenate a sequence of backend storage arrays along an axis.



.. py:class:: NumPyStorageProvider

   Bases: :py:obj:`ArrayStorageProvider`


   Default array storage provider backed by NumPy in-memory data structures.


   .. py:method:: create_group(shape_prefix: Tuple[int, Ellipsis] = ()) -> NumPyStorageGroup

      Create and return a new array storage group instance.



   .. py:method:: concat(arrays: Sequence[Any], axis: int = 0) -> Any

      Concatenate a sequence of backend storage arrays along an axis.



