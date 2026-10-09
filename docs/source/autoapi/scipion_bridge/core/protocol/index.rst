scipion_bridge.core.protocol
============================

.. py:module:: scipion_bridge.core.protocol


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/core/protocol/chain_utils/index
   /autoapi/scipion_bridge/core/protocol/fields/index
   /autoapi/scipion_bridge/core/protocol/protocol_base/index


Exceptions
----------

.. autoapisummary::

   scipion_bridge.core.protocol.ValidationError


Classes
-------

.. autoapisummary::

   scipion_bridge.core.protocol.ChainedProtocol
   scipion_bridge.core.protocol.Protocol
   scipion_bridge.core.protocol.Field
   scipion_bridge.core.protocol.Input
   scipion_bridge.core.protocol.Resource
   scipion_bridge.core.protocol.ResourceScope


Functions
---------

.. autoapisummary::

   scipion_bridge.core.protocol.resources


Package Contents
----------------

.. py:class:: ChainedProtocol(first: Protocol, second: Protocol, *, mapping: Optional[Mapping[str, str]] = None)

   Bases: :py:obj:`Protocol`


   Two protocols executed as one: the outputs of ``first`` feed ``second``.

   Created with ``first | second`` or ``first.pipe(second)``. The steps of both
   protocols are merged into a single streaming pipeline; each batch emitted by
   ``first`` is passed as one item to the inputs of ``second``. Inputs, parameters
   and resources of both protocols are merged; the outputs are those of ``second``.


   .. py:attribute:: first


   .. py:attribute:: second


   .. py:property:: configuration
      :type: ProtocolConfiguration



   .. py:method:: setup() -> None

      Optional setup method for protocol initialization.



   .. py:method:: validate_protocol_configuration() -> None


   .. py:method:: outputs() -> Dict[str, Type]


   .. py:method:: steps() -> scipion_bridge.core.streaming.ops.Op

      Return the streaming pipeline of the protocol, built from its inputs.

      Each ``.map()`` runs as a separate pipeline stage. Split GPU work and
      CPU post-processing into separate maps so that they overlap on
      consecutive batches::

          def steps(self):
              return (
                  self.particles.chunk(256)
                  .map(self._forward)
                  .map(self._build_metadata)
              )



.. py:class:: Protocol(protocol_id: Optional[str] = None)

   .. py:attribute:: compute_resources
      :type:  ClassVar[Optional[scipion_bridge.core.environment.compute.ComputeResources]]
      :value: None



   .. py:property:: configuration
      :type: ProtocolConfiguration



   .. py:attribute:: protocol_id
      :type:  str
      :value: '00000000000000000000000000000000'



   .. py:method:: setup()

      Optional setup method for protocol initialization.



   .. py:method:: get_pipeline() -> scipion_bridge.core.streaming.ops.Op

      Return the pipeline of operations for this protocol.



   .. py:method:: outputs() -> Dict[str, Type]
      :abstractmethod:



   .. py:method:: steps() -> scipion_bridge.core.streaming.ops.Op
      :abstractmethod:


      Return the streaming pipeline of the protocol, built from its inputs.

      Each ``.map()`` runs as a separate pipeline stage. Split GPU work and
      CPU post-processing into separate maps so that they overlap on
      consecutive batches::

          def steps(self):
              return (
                  self.particles.chunk(256)
                  .map(self._forward)
                  .map(self._build_metadata)
              )



   .. py:method:: validate_protocol_configuration()


   .. py:method:: pipe(other: Protocol, mapping: Optional[Mapping[str, str]] = None) -> ChainedProtocol

      Chain ``other`` after this protocol.

      The outputs of this protocol feed the inputs of ``other``. Unless
      ``mapping`` (output name -> input name of ``other``) is given, outputs
      are matched to inputs by name, or by type if this protocol has a single
      output. The result is again a protocol, whose inputs are the inputs of
      this protocol plus the unwired inputs of ``other``.



.. py:exception:: ValidationError

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:function:: resources(*, gpus: float = 0, min_vram: Optional[float] = None, cpus: Optional[float] = None, task: scipion_bridge.core.environment.compute.TaskType = TaskType.EPHEMERAL) -> Callable[[ProtocolT], ProtocolT]

   Declare the compute resources of a protocol's stages.

   Every call of the protocol's ``map`` and ``map_element`` functions
   requires the resources::

       @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
       class Inference(B.Protocol): ...

   See :class:`ComputeResources` for the arguments.


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



.. py:class:: ResourceScope

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   Lifecycle and sharing scope for protocol resources.


   .. py:attribute:: PROCESS
      :value: 'process'



   .. py:attribute:: SHARED
      :value: 'shared'



