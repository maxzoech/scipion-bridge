scipion_bridge.core.protocol.fields
===================================

.. py:module:: scipion_bridge.core.protocol.fields


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.protocol.fields.T
   scipion_bridge.core.protocol.fields.FieldSelf
   scipion_bridge.core.protocol.fields.InputSelf


Classes
-------

.. autoapisummary::

   scipion_bridge.core.protocol.fields.BoundField
   scipion_bridge.core.protocol.fields.Field
   scipion_bridge.core.protocol.fields.BoundInput
   scipion_bridge.core.protocol.fields.Input
   scipion_bridge.core.protocol.fields.Resource


Module Contents
---------------

.. py:data:: T

.. py:data:: FieldSelf

.. py:data:: InputSelf

.. py:class:: BoundField(name: str, *, dtype: Optional[Any] = None, default: Optional[T] = None, optional: Optional[bool] = None, label: Optional[str] = None, group: Optional[str] = None, help: Optional[str] = None)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ]


   .. py:attribute:: name


   .. py:attribute:: default
      :value: None



   .. py:attribute:: optional
      :value: None



   .. py:attribute:: label
      :value: None



   .. py:attribute:: group
      :value: None



   .. py:attribute:: help
      :value: None



   .. py:property:: value
      :type: T


      Return the value of the field, or None if not set.


.. py:class:: Field(*, dtype: Optional[Any] = None, default: Optional[T] = None, optional: Optional[bool] = None, label: Optional[str] = None, group: Optional[str] = None, help: Optional[str] = None)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ]


   .. py:attribute:: default
      :value: None



   .. py:attribute:: optional
      :value: None



   .. py:attribute:: label
      :value: None



   .. py:attribute:: group
      :value: None



   .. py:attribute:: help
      :value: None



.. py:class:: BoundInput(name: str, *, dtype: Optional[Any] = None, default: Optional[T] = None, optional: Optional[bool] = None, label: Optional[str] = None, help: Optional[str] = None)

   Bases: :py:obj:`BoundField`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.streaming.ops.Source`


   Entry point input stream node.


.. py:class:: Input(*, dtype: Optional[Any] = None, default: Optional[T] = None, optional: Optional[bool] = None, label: Optional[str] = None, help: Optional[str] = None)

   Bases: :py:obj:`Field`\ [\ :py:obj:`T`\ ]


.. py:class:: Resource(*, builder: Callable[[Any], T], scope: scipion_bridge.core.environment.resource_provider.ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ]


   Marker and descriptor for actor-scoped protocol resources.


   .. py:attribute:: builder


   .. py:attribute:: scope


   .. py:attribute:: name
      :type:  Optional[str]
      :value: None



