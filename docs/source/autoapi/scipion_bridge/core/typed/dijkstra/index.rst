scipion_bridge.core.typed.dijkstra
==================================

.. py:module:: scipion_bridge.core.typed.dijkstra


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.typed.dijkstra.T


Classes
-------

.. autoapisummary::

   scipion_bridge.core.typed.dijkstra.PathfindingContainer


Functions
---------

.. autoapisummary::

   scipion_bridge.core.typed.dijkstra.build_default_container
   scipion_bridge.core.typed.dijkstra.find_shortest_path


Module Contents
---------------

.. py:data:: T

.. py:class:: PathfindingContainer(value: T, previous: Optional[T], weight: int)

   Bases: :py:obj:`Generic`\ [\ :py:obj:`T`\ ]


   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default


   .. py:attribute:: value


   .. py:attribute:: previous


   .. py:attribute:: weight


.. py:function:: build_default_container(graph: networkx.DiGraph, value: T, previous: Optional[T], weight: int)

.. py:function:: find_shortest_path(graph: networkx.DiGraph, origin: T, destination: T, intermediate: Optional[Any] = None, container_builder: Callable[[networkx.DiGraph, T, Optional[T], int], PathfindingContainer] = build_default_container, weight: str = 'weight')

