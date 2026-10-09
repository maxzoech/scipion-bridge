scipion_bridge.core.utils.arc
=============================

.. py:module:: scipion_bridge.core.utils.arc


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.utils.arc.manager


Classes
-------

.. autoapisummary::

   scipion_bridge.core.utils.arc.FileReferenceCounter


Module Contents
---------------

.. py:class:: FileReferenceCounter

   .. py:attribute:: references
      :type:  Dict[os.PathLike, int]


   .. py:method:: new_managed_file(file_ext: Optional[str], temp_file_provider: scipion_bridge.core.environment.temp_files.TemporaryFilesProvider = Provide[Container.temp_file_provider]) -> pathlib.Path


   .. py:method:: register_temporary_file(path: os.PathLike) -> pathlib.Path


   .. py:method:: add_reference(path: os.PathLike)


   .. py:method:: remove_reference(path: os.PathLike, temp_file_provider: scipion_bridge.core.environment.temp_files.TemporaryFilesProvider = Provide[Container.temp_file_provider])


   .. py:method:: is_tracked(path: os.PathLike)


   .. py:method:: get_count(path: os.PathLike)


.. py:data:: manager

