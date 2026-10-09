scipion_bridge.core.utils.marker
================================

.. py:module:: scipion_bridge.core.utils.marker

.. autoapi-nested-parse::

   Generic marker classes that keep their identity across processes.

   Every specialization ``Origin[Args]`` of a :class:`Marker` is a real class. To
   make such classes picklable *by name* (so that stdlib ``pickle``,
   ``cloudpickle``, ``ray.cloudpickle`` and others send a reference instead of
   re-creating the class), each specialization gets a ``__qualname__`` that fully
   encodes its origin and type arguments without using dots, e.g.::

       Set[scipion_bridge/single_particle/particle:Particle]

   The specialization is stored under that name on the module of its origin, and
   every module defining a marker class gets a module-level ``__getattr__``
   (PEP 562) that rebuilds a specialization from its encoded name. A process that
   unpickles a specialization it has never built therefore resolves it through
   ``Origin[Args]`` and the shared ``_generic_cache``.

   Specializations whose origin or arguments cannot be imported by name (classes
   defined in ``__main__`` or in a local scope, non-class arguments) keep their
   readable name and are serialized by value, as before.



Attributes
----------

.. autoapisummary::

   scipion_bridge.core.utils.marker.T


Classes
-------

.. autoapisummary::

   scipion_bridge.core.utils.marker.MarkerMeta
   scipion_bridge.core.utils.marker.Marker


Module Contents
---------------

.. py:data:: T

.. py:class:: MarkerMeta(name: str, bases: Tuple[type, Ellipsis], namespace: Dict[str, Any], **kwargs: Any)

   Bases: :py:obj:`abc.ABCMeta`


   Metaclass making Marker specializations resolvable by module and qualname.


.. py:class:: Marker(dtype: Optional[Any] = None, **kwargs: Any)

   Bases: :py:obj:`Generic`\ [\ :py:obj:`T`\ ]


   Generic base class supporting static typing and runtime annotation introspection.


   .. py:attribute:: name
      :type:  Optional[str]
      :value: None



   .. py:attribute:: options


   .. py:property:: dtype
      :type: Optional[Any]



