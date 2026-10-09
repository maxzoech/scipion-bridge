scipion_bridge.core.typed.resolve
=================================

.. py:module:: scipion_bridge.core.typed.resolve


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.typed.resolve.Target
   scipion_bridge.core.typed.resolve.Origin
   scipion_bridge.core.typed.resolve.Intermediate
   scipion_bridge.core.typed.resolve.Resolve
   scipion_bridge.core.typed.resolve.DEFAULT_REGISTRY


Classes
-------

.. autoapisummary::

   scipion_bridge.core.typed.resolve.ResolveStep
   scipion_bridge.core.typed.resolve.ResolveContext
   scipion_bridge.core.typed.resolve.ScopedPathfindingContainer
   scipion_bridge.core.typed.resolve.ComposedResolver
   scipion_bridge.core.typed.resolve.Registry


Functions
---------

.. autoapisummary::

   scipion_bridge.core.typed.resolve.build_default_container
   scipion_bridge.core.typed.resolve.resolution_context
   scipion_bridge.core.typed.resolve.estimate_optimal_chunk_size
   scipion_bridge.core.typed.resolve.current_registry
   scipion_bridge.core.typed.resolve.resolver
   scipion_bridge.core.typed.resolve.resolve
   scipion_bridge.core.typed.resolve.resolve_iter
   scipion_bridge.core.typed.resolve.find_resolver
   scipion_bridge.core.typed.resolve.lift_resolvers
   scipion_bridge.core.typed.resolve.resolve_params


Module Contents
---------------

.. py:class:: ResolveStep

   Bases: :py:obj:`tuple`


   .. py:attribute:: cls


   .. py:attribute:: requires_metadata


   .. py:attribute:: requires_slice


   .. py:attribute:: description


.. py:class:: ResolveContext

   Bases: :py:obj:`tuple`


   .. py:attribute:: registry


   .. py:attribute:: namespaces


   .. py:attribute:: caller_namespace


   .. py:attribute:: recursion_level


.. py:data:: Target

.. py:data:: Origin

.. py:data:: Intermediate

.. py:data:: Resolve

.. py:class:: ScopedPathfindingContainer(value: Optional[Any], previous: Optional[Any], weight: int, incoming_edge_attributes: Optional[ResolverNode], local_scope_name: str)

   Bases: :py:obj:`scipion_bridge.core.typed.dijkstra.PathfindingContainer`


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


   .. py:class:: ResolverNode

      Bases: :py:obj:`tuple`


      .. py:attribute:: resolver_fn


      .. py:attribute:: module


      .. py:attribute:: requires_slice



   .. py:attribute:: edge_attributes


   .. py:attribute:: local_scope_name


   .. py:property:: is_local_scope


   .. py:property:: resolution_priority
      :type: int



   .. py:property:: slice_priority
      :type: int



.. py:function:: build_default_container(graph: networkx.DiGraph, value: Type, previous: Optional[Type], weight: int, local_scope_name: str)

.. py:function:: resolution_context(registry: Registry, namespace: Set[str], caller_namespace: str) -> collections.abc.Generator[ResolveContext]

.. py:function:: estimate_optimal_chunk_size(item_nbytes: Optional[int], target_bytes: Optional[int] = None, default_chunk_size: int = 100) -> int

   Calculate an optimal batch/chunk size targeting a memory footprint (default 32MB).

   If item_nbytes is None or <= 0, returns default_chunk_size.
   Otherwise returns max(1, target_bytes // item_nbytes).


.. py:class:: ComposedResolver(origin: Type[Origin], target: Type[Target], steps: List[ResolveStep])

   Bases: :py:obj:`Generic`\ [\ :py:obj:`Origin`\ , :py:obj:`Target`\ ]


   An independent execution pipeline composed of resolved transformation steps.


   .. py:attribute:: origin


   .. py:attribute:: target


   .. py:attribute:: steps


   .. py:method:: iter(value: Origin, *, chunk_size: Optional[int] = None, target_bytes: Optional[int] = None, metadata: Optional[Any] = None) -> Iterator[Target]

      Iteratively resolve value in chunks by slicing the first slice-aware step.

      If chunk_size is None, optimal chunk size is estimated based on intermediate.estimated_item_nbytes
      targeting target_bytes (defaults to 32MB).

      :raises TypeError: If value doesn't match origin, if no step in the path supports slicing,
          or if the intermediate object before the slice step is not Sized.
      :raises ValueError: If chunk_size or target_bytes <= 0.



.. py:class:: Registry

   .. py:attribute:: graph
      :type:  networkx.DiGraph[Type[Hashable]]


   .. py:method:: get_registered_modules() -> Set[str]


   .. py:method:: add_resolver(origin: Type[Origin], target: Type[Target], resolver: Union[type, Callable], namespace: Optional[str] = None, requires_metadata: Optional[bool] = None, requires_slice: Optional[bool] = None)


   .. py:method:: find_resolve_func(namespace: Set[str], origin: Type[Origin], target: Type[Target], intermediate: Optional[Type[Intermediate]] = None, local_scope_name: Optional[str] = None) -> ComposedResolver[Origin, Target]


   .. py:method:: find_resolver(origin: Union[Type[Origin], Origin], target: Type[Target], intermediate: Optional[Type[Intermediate]] = None) -> ComposedResolver[Origin, Target]

      Precompute and return a ComposedResolver for the given origin and target types.



   .. py:method:: resolve(value, astype: Type[Target], intermediate: Optional[Type[Intermediate]] = None, metadata: Optional[Any] = None, slice: Optional[slice] = None) -> Target


   .. py:method:: resolve_iter(value, astype: Type[Target], intermediate: Optional[Type[Intermediate]] = None, chunk_size: Optional[int] = None, target_bytes: Optional[int] = None, metadata: Optional[Any] = None) -> Iterator[Target]


   .. py:method:: lift_resolvers(origin_module_name: str, target_module_name: str)


.. py:data:: DEFAULT_REGISTRY

.. py:function:: current_registry() -> Registry

.. py:function:: resolver(target: Any) -> Any

.. py:function:: resolve(value, astype: Type[Target], intermediate: Optional[Type[Intermediate]] = None, metadata: Optional[Any] = None, slice: Optional[slice] = None) -> Target

.. py:function:: resolve_iter(value, astype: Type[Target], intermediate: Optional[Type[Intermediate]] = None, chunk_size: Optional[int] = None, target_bytes: Optional[int] = None, metadata: Optional[Any] = None) -> Iterator[Target]

.. py:function:: find_resolver(origin: Union[Type[Origin], Origin], target: Type[Target], intermediate: Optional[Type[Intermediate]] = None) -> ComposedResolver[Origin, Target]

   Precompute and return a ComposedResolver from the current registry.


.. py:function:: lift_resolvers(*modules: types.ModuleType, target: Optional[types.ModuleType] = None)

.. py:function:: resolve_params(f: Callable)

