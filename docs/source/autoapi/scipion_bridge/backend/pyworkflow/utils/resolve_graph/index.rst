scipion_bridge.backend.pyworkflow.utils.resolve_graph
=====================================================

.. py:module:: scipion_bridge.backend.pyworkflow.utils.resolve_graph

.. autoapi-nested-parse::

   Scipion pointer class resolution helpers.



Functions
---------

.. autoapisummary::

   scipion_bridge.backend.pyworkflow.utils.resolve_graph.find_pointer_class
   scipion_bridge.backend.pyworkflow.utils.resolve_graph.find_output_pointer_class
   scipion_bridge.backend.pyworkflow.utils.resolve_graph.find_pyworkflow_object_for_type


Module Contents
---------------

.. py:function:: find_pointer_class(target_type: Type) -> Optional[str]

   Inspect the type resolution graph to find the appropriate Scipion input
   container class name (e.g. 'SetOfParticles') for a given target type.

   Traverses predecessors (Scipion -> Bridge) in the resolution graph.


.. py:function:: find_output_pointer_class(target_type: Type, include_itemwise: bool = False) -> Optional[Type]

   Inspect the type resolution graph to find the appropriate Scipion output
   container Python type (e.g. SetOfParticlesFlex) for a given target type.

   Traverses successors (Bridge -> Scipion) in the resolution graph.


.. py:function:: find_pyworkflow_object_for_type(target_type: Type) -> Optional[str]

   Inspect the type resolution graph to find the appropriate PyWorkflow
   object class for a given target type (defaults to input lookup).


