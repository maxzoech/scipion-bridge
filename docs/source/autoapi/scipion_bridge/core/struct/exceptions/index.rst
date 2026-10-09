scipion_bridge.core.struct.exceptions
=====================================

.. py:module:: scipion_bridge.core.struct.exceptions

.. autoapi-nested-parse::

   Exceptions for the struct module.



Exceptions
----------

.. autoapisummary::

   scipion_bridge.core.struct.exceptions.UninitializedFieldError


Module Contents
---------------

.. py:exception:: UninitializedFieldError

   Bases: :py:obj:`AttributeError`, :py:obj:`ValueError`


   Raised when attempting to access a field or element that has not been initialized or is null.

   Inherits from both AttributeError and ValueError so callers expecting either standard Python
   attribute lookup failure or Arrow null-value errors catch this consistently.


