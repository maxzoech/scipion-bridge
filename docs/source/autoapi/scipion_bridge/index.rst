scipion_bridge
==============

.. py:module:: scipion_bridge


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/backend/index
   /autoapi/scipion_bridge/core/index
   /autoapi/scipion_bridge/single_particle/index
   /autoapi/scipion_bridge/visualize/index


Attributes
----------

.. autoapisummary::

   scipion_bridge.Resolve
   scipion_bridge.ResolveProxy
   scipion_bridge.Dim
   scipion_bridge.FLUSH


Classes
-------

.. autoapisummary::

   scipion_bridge.ComposedResolver
   scipion_bridge.Proxy
   scipion_bridge.ProxyGroup
   scipion_bridge.Output
   scipion_bridge.ParticleStackProxy
   scipion_bridge.StarfileProxy
   scipion_bridge.MRCStackProxy
   scipion_bridge.Domain
   scipion_bridge.ComputeResources
   scipion_bridge.TaskType
   scipion_bridge.Protocol
   scipion_bridge.Field
   scipion_bridge.Input
   scipion_bridge.Resource
   scipion_bridge.ResourceScope
   scipion_bridge.ChainedProtocol
   scipion_bridge.ResourceProvider
   scipion_bridge.DefaultResourceProvider
   scipion_bridge.Struct
   scipion_bridge.Set
   scipion_bridge.Collection
   scipion_bridge.Array
   scipion_bridge.Arg
   scipion_bridge.Op
   scipion_bridge.FlushSignal


Functions
---------

.. autoapisummary::

   scipion_bridge.resolver
   scipion_bridge.resolve_params
   scipion_bridge.resolve
   scipion_bridge.resolve_iter
   scipion_bridge.find_resolver
   scipion_bridge.proxify
   scipion_bridge.namedproxy
   scipion_bridge.estimate_optimal_chunk_size
   scipion_bridge.shell_command
   scipion_bridge.resources
   scipion_bridge.concat


Package Contents
----------------

.. py:function:: resolver(target: Any) -> Any

.. py:function:: resolve_params(f: Callable)

.. py:function:: resolve(value, astype: Type[Target], intermediate: Optional[Type[Intermediate]] = None, metadata: Optional[Any] = None, slice: Optional[slice] = None) -> Target

.. py:function:: resolve_iter(value, astype: Type[Target], intermediate: Optional[Type[Intermediate]] = None, chunk_size: Optional[int] = None, target_bytes: Optional[int] = None, metadata: Optional[Any] = None) -> Iterator[Target]

.. py:function:: find_resolver(origin: Union[Type[Origin], Origin], target: Type[Target], intermediate: Optional[Type[Intermediate]] = None) -> ComposedResolver[Origin, Target]

   Precompute and return a ComposedResolver from the current registry.


.. py:data:: Resolve

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



.. py:function:: proxify(f: Callable[Ellipsis, Any]) -> Callable[Ellipsis, Any]

.. py:class:: Proxy(path: os.PathLike, managed=False, *args, **kwargs)

   .. py:attribute:: managed
      :value: False



   .. py:property:: path
      :type: pathlib.Path



   .. py:method:: file_ext() -> Optional[str]
      :classmethod:



   .. py:method:: extensions() -> Optional[tuple[str, Ellipsis]]
      :classmethod:



   .. py:method:: prefix() -> Optional[str]
      :classmethod:



   .. py:method:: suffix() -> Optional[str]
      :classmethod:



   .. py:method:: extension() -> Optional[str]
      :classmethod:



   .. py:method:: get_referenced_paths(path_str: str) -> List[pathlib.Path]
      :classmethod:


      Return the list of filesystem paths associated with this proxy class.



   .. py:method:: from_func_param(param: FuncParam) -> Proxy
      :classmethod:


      Instantiate a proxy from a FuncParam.



   .. py:method:: from_proxy(source: Proxy, copy_data: bool = True) -> Casted
      :classmethod:


      Create a typed proxy instance from another proxy instance.



   .. py:method:: new_temporary_proxy(base_path: Optional[os.PathLike] = None, temp_file_provider: scipion_bridge.core.environment.temp_files.TemporaryFilesProvider = Provide[Container.temp_file_provider]) -> Proxy
      :classmethod:



   .. py:method:: typed(*, astype: Type[Casted], copy_data=True) -> Casted


   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated size in bytes for a single logical item contained in this proxy.

      Returns None if size cannot be determined without reading the data.


.. py:class:: ProxyGroup(base_path: os.PathLike, managed: bool = False, **kwargs: Any)

   Bases: :py:obj:`Proxy`, :py:obj:`abc.ABC`


   Abstract base class representing a grouped collection of Proxy objects.

   Subclasses must annotate child proxy fields with Proxy types and implement
   the primary_proxy abstract property.


   .. py:attribute:: base_path


   .. py:attribute:: managed
      :value: False



   .. py:method:: get_proxy_fields() -> Dict[str, Type[Proxy]]
      :classmethod:


      Return a dictionary mapping child proxy field names to their Proxy subclass types.



   .. py:method:: get_field_paths(base_path: os.PathLike) -> Dict[str, pathlib.Path]
      :classmethod:


      Return a dictionary mapping each proxy field name to its corresponding Path.



   .. py:method:: primary_proxy_type() -> Optional[Type[Proxy]]
      :classmethod:


      Return the Proxy class corresponding to primary_proxy.



   .. py:method:: extensions() -> Optional[tuple[str, Ellipsis]]
      :classmethod:



   .. py:method:: extract_base_path(primary_path: os.PathLike) -> pathlib.Path
      :classmethod:


      Extract the canonical base_path from a primary proxy path by stripping
      group/child prefixes, suffixes, and extensions.



   .. py:method:: get_referenced_paths(path_str: str) -> List[pathlib.Path]
      :classmethod:


      Return the list of child proxy paths associated with this ProxyGroup class.



   .. py:method:: from_func_param(param: FuncParam) -> ProxyGroup
      :classmethod:


      Instantiate a ProxyGroup from a FuncParam.



   .. py:method:: from_proxy(source: Proxy, copy_data: bool = True) -> Casted
      :classmethod:


      Create a typed proxy instance from another proxy instance.



   .. py:property:: primary_proxy
      :type: Proxy

      :abstractmethod:


      Return the primary proxy instance for this group.


   .. py:property:: path
      :type: pathlib.Path



   .. py:method:: new_temporary_proxy(base_path: Optional[os.PathLike] = None, temp_file_provider: scipion_bridge.core.environment.temp_files.TemporaryFilesProvider = Provide[Container.temp_file_provider]) -> ProxyGroup
      :classmethod:



   .. py:method:: keys()


   .. py:method:: values()


   .. py:method:: items()


   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated total size in bytes for a single logical item across all child proxies.

      Returns the sum of all child proxy estimates, or None if any child cannot estimate its size.


.. py:class:: Output(dtype: Type[T])

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


   .. py:attribute:: dtype


.. py:type:: ResolveProxy
   :canonical: Union[Output[Intermediate], Intermediate, Origin]


.. py:function:: namedproxy(typename: str, *, file_ext: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> Type[ProxyProtocol]

.. py:function:: estimate_optimal_chunk_size(item_nbytes: Optional[int], target_bytes: Optional[int] = None, default_chunk_size: int = 100) -> int

   Calculate an optimal batch/chunk size targeting a memory footprint (default 32MB).

   If item_nbytes is None or <= 0, returns default_chunk_size.
   Otherwise returns max(1, target_bytes // item_nbytes).


.. py:class:: ParticleStackProxy(base_path: os.PathLike, managed: bool = False, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.typed.proxy.ProxyGroup`


   Abstract base class representing a grouped collection of Proxy objects.

   Subclasses must annotate child proxy fields with Proxy types and implement
   the primary_proxy abstract property.


   .. py:attribute:: metadata
      :type:  StarfileProxy


   .. py:attribute:: particle_stack
      :type:  MRCStackProxy


   .. py:property:: primary_proxy
      :type: scipion_bridge.core.typed.proxy.Proxy


      Return the primary proxy instance for this group.


   .. py:property:: num_particles
      :type: int



   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated total size in bytes for a single logical item across all child proxies.

      Returns the sum of all child proxy estimates, or None if any child cannot estimate its size.


.. py:class:: StarfileProxy(path: os.PathLike, managed=False, *args, **kwargs)

   Bases: :py:obj:`scipion_bridge.core.typed.proxy.Proxy`


   .. py:method:: read() -> Any


   .. py:method:: file_ext()
      :classmethod:



.. py:class:: MRCStackProxy(path: os.PathLike, managed: bool = False, metadata_path: Optional[os.PathLike] = None, *args, **kwargs)

   Bases: :py:obj:`scipion_bridge.core.typed.proxy.Proxy`


   .. py:attribute:: metadata_path
      :type:  Optional[pathlib.Path]


   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated size in bytes for a single logical item contained in this proxy.

      Returns None if size cannot be determined without reading the data.


   .. py:method:: file_ext() -> Optional[str]
      :classmethod:



   .. py:method:: extensions() -> Optional[tuple[str, Ellipsis]]
      :classmethod:



   .. py:method:: find_mrc_path_from_star(star_path: pathlib.Path) -> pathlib.Path
      :classmethod:


      Find the MRC data file referenced by a .star file.



.. py:class:: Domain

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: command
      :type:  List[str]


   .. py:attribute:: isolated
      :type:  bool
      :value: False



   .. py:method:: default() -> Domain
      :classmethod:



.. py:class:: ComputeResources

   Resources required by every call of the stages of a protocol.

   Only the Ray backend schedules by these resources; the standalone and
   pyworkflow backends ignore them.

   .. attribute:: gpus

      Number of GPUs; fractions let calls share a GPU.

   .. attribute:: min_vram

      GPU memory in GiB a call needs on each of its GPUs. The
      backend claims the share of a GPU that holds it (see
      :func:`gpu_claim`), or ``gpus`` if that is more. GPU memory is
      not enforced: calls sharing a GPU must stay within their claim.

   .. attribute:: cpus

      Number of CPU cores reserved for a call. ``None`` reserves
      none: the calls may use all cores of the machine and the OS
      scheduler shares them.

   .. attribute:: task

      Whether the resources are taken per call or held across calls.


   .. py:attribute:: gpus
      :type:  float
      :value: 0



   .. py:attribute:: min_vram
      :type:  Optional[float]
      :value: None



   .. py:attribute:: cpus
      :type:  Optional[float]
      :value: None



   .. py:attribute:: task
      :type:  TaskType


.. py:class:: TaskType

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   How the resources of a protocol are held while its stages run.


   .. py:attribute:: EPHEMERAL
      :value: 'ephemeral'



   .. py:attribute:: LONG_RUNNING
      :value: 'long_running'



.. py:function:: shell_command(f: F, *, domain: scipion_bridge.core.environment.domain.Domain = Domain.default(), name: Optional[str] = None, postprocess_fn: Optional[Callable] = None, **args_map) -> F
                 shell_command(f: None = None, *, domain: scipion_bridge.core.environment.domain.Domain = Domain.default(), name: Optional[str] = None, postprocess_fn: Optional[Callable] = None, **args_map) -> ShellDecoratorProtocol

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



.. py:function:: resources(*, gpus: float = 0, min_vram: Optional[float] = None, cpus: Optional[float] = None, task: scipion_bridge.core.environment.compute.TaskType = TaskType.EPHEMERAL) -> Callable[[ProtocolT], ProtocolT]

   Declare the compute resources of a protocol's stages.

   Every call of the protocol's ``map`` and ``map_element`` functions
   requires the resources::

       @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
       class Inference(B.Protocol): ...

   See :class:`ComputeResources` for the arguments.


.. py:class:: ResourceProvider

   Bases: :py:obj:`abc.ABC`


   Abstract provider for managing lifecycle and caching of Protocol resources.


   .. py:method:: get_resource(name: str, builder: Callable[[Any], Any], instance: Any, scope: ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None) -> Any
      :abstractmethod:


      Retrieve or build a resource for a given protocol instance.



.. py:class:: DefaultResourceProvider

   Bases: :py:obj:`ResourceProvider`


   Process-local provider that lazily constructs resources and caches them.


   .. py:method:: get_resource(name: str, builder: Callable[[Any], Any], instance: Any, scope: ResourceScope = ResourceScope.PROCESS, dtype: Optional[Any] = None) -> Any

      Retrieve or build a resource for a given protocol instance.



.. py:class:: Struct(storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`Trait`, :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Materialization layer: builds the finalized Schema and binds array storage.


   .. py:method:: default() -> Struct
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:



   .. py:property:: storage
      :type: scipion_bridge.core.struct.storage._BaseStorage



   .. py:method:: is_initialized(field_name: str) -> bool

      Check if a field has been initialized in storage.



   .. py:method:: initialized_fields() -> list[str]

      Return the names of all fields that have been initialized.



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the initialized fields to a RecordBatch with a single row.

      For a view (a Set element, a Collection item or a nested field), only
      the data of the view is exported. Nested Sets become nested list
      columns.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct a Struct owning its storage from a batch of :meth:`to_arrow`.



.. py:class:: Set(items: Sequence[T] = (), capacity: Optional[Union[int, scipion_bridge.core.struct.struct.Arg]] = None, storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Sequence container for Struct instances backed by Apache Arrow columnar storage.


   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:



   .. py:method:: item_type() -> Type[scipion_bridge.core.struct.struct.Struct]
      :classmethod:


      Return the element Struct type of the Set.



   .. py:property:: capacity
      :type: Optional[int]



   .. py:property:: storage
      :type: scipion_bridge.core.struct.storage._BaseStorage



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



   .. py:method:: default() -> Set[T]
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the visible rows of the Set to an Apache Arrow RecordBatch.

      Every top-level field becomes a column; nested Structs and Sets become
      StructArray columns. Uninitialized fields are omitted. For a view, only
      the rows of the view are exported, and contiguous buffers are wrapped
      without copying.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct a Set from a RecordBatch produced by :meth:`to_arrow`.

      Column buffers are adopted without copying where Arrow allows it; such
      buffers may be read-only and are copied on the first in-place write.



   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


.. py:class:: Collection(size: int, items: Optional[Union[Sequence[T], Dict[int, T]]] = None, storage: Optional[scipion_bridge.core.struct.storage._BaseStorage] = None, dtype: Optional[Any] = None, **kwargs: Any)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Statically indexed sequence container for Struct items backed by columnar storage.


   .. py:attribute:: size


   .. py:property:: storage
      :type: scipion_bridge.core.struct.storage._BaseStorage



   .. py:method:: is_initialized(index: int) -> bool


   .. py:method:: initialized_indices() -> list[int]


   .. py:method:: default() -> scipion_bridge.core.struct.schema.SchemaConvertible
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: item_type() -> Type[scipion_bridge.core.struct.struct.Struct]
      :classmethod:


      Return the element Struct type of the Collection.



   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:

      :abstractmethod:



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



   .. py:method:: items() -> Iterator[Tuple[int, T]]


   .. py:method:: keys() -> list[int]


   .. py:method:: values() -> list[T]


   .. py:method:: to_set() -> scipion_bridge.core.struct.set.Set[T]


   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the Collection to a RecordBatch with one row per slot.

      Uninitialized slots, and fields not initialized in a slot, are null.
      The size is stored in the batch metadata.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct a Collection from a RecordBatch produced by :meth:`to_arrow`.



.. py:function:: concat(sets: Sequence[Set[T]]) -> Set[T]

   Concatenate multiple Sets with identical schemas along axis 0.

   :param sets: A non-empty sequence of Set instances to concatenate.

   :returns: A new Set containing the concatenated data.


.. py:class:: Array(dtype: Optional[Union[numpy.dtype, type, str]] = None, *, shape: Optional[Union[Tuple[Union[Dim, int, None], Ellipsis], list]] = None, is_scalar: bool = False)

   Bases: :py:obj:`scipion_bridge.core.utils.marker.Marker`\ [\ :py:obj:`T`\ ], :py:obj:`scipion_bridge.core.struct.schema.SchemaConvertible`


   Descriptor and schema representation for array attributes on Struct classes.


   .. py:attribute:: is_scalar
      :value: False



   .. py:attribute:: shape_spec
      :type:  Tuple[Dim, Ellipsis]
      :value: ()



   .. py:property:: shape
      :type: Tuple[Optional[int], Ellipsis]



   .. py:property:: dtype
      :type: Optional[numpy.dtype]



   .. py:method:: default() -> scipion_bridge.core.struct.schema.SchemaConvertible
      :classmethod:


      Create a default, unspecialized instance from the type.



   .. py:method:: schema() -> scipion_bridge.core.struct.schema.Schema
      :classmethod:

      :abstractmethod:



   .. py:property:: is_descriptor
      :type: bool


      True for a field declared on a Struct class, which holds no data.


   .. py:method:: to_arrow() -> pyarrow.RecordBatch

      Export the initialized data to an Apache Arrow RecordBatch.



   .. py:method:: from_arrow(batch: pyarrow.RecordBatch) -> Self
      :classmethod:


      Construct an instance from a RecordBatch produced by :meth:`to_arrow`.



   .. py:method:: convert_to_entry() -> scipion_bridge.core.struct.schema.Entry

      Convert this instance into a schema Entry tree representation.



.. py:class:: Arg(value: Optional[Union[int, Arg]] = None, *, name: Optional[str] = None)

   Class-level dimension specification or named parameter.


   .. py:attribute:: name
      :value: None



   .. py:method:: new(value: Optional[Union[Arg, int]] = None, *, name: Optional[str] = None) -> Arg
      :classmethod:



   .. py:property:: value
      :type: Optional[int]



   .. py:method:: validate(other: Any) -> None


   .. py:property:: is_static
      :type: bool



.. py:type:: Dim
   :canonical: Arg


.. py:class:: Op(upstream: Optional[List[Node]] = None)

   Bases: :py:obj:`scipion_bridge.core.streaming.node.Node`


   Intermediate operation node that allows chaining downstream operations.


   .. py:method:: op(node: _NodeT) -> _NodeT

      Connect a downstream node to this op.



   .. py:method:: map_batch(func: Callable[[Any], Any], *, cpu_only: bool = False) -> MapOp

      Transform entire incoming stream item / batch (1:1).

      Every map is executed as its own pipeline stage, and consecutive stages
      process different items concurrently. Splitting a step into separate
      maps therefore overlaps its parts, e.g. CPU post-processing of one
      batch with the GPU forward pass of the next::

          particles.chunk(256).map(forward).map(build_metadata)

      :param func: Function applied to each item.
      :param cpu_only: Run without the GPUs of the protocol's compute
                       resources (``@resources``), e.g. for cheap bookkeeping maps
                       that should not wait for a GPU.



   .. py:method:: map(func: Callable[[Any], Any], *, cpu_only: bool = False) -> MapOp

      Alias for map_batch.



   .. py:method:: map_element(func: Callable[[Any], Any], *, workers: scipion_bridge.core.streaming.element_mapper.Workers = 'auto', executor: scipion_bridge.core.streaming.element_mapper.Executor = 'thread', start_method: Optional[scipion_bridge.core.streaming.element_mapper.StartMethod] = None, chunksize: Optional[int] = None, cpu_only: bool = False) -> MapElementOp

      Apply ``func`` to every element of the incoming collections, in parallel.

      For each collection ``col`` (any collection with ``__len__``,
      ``__getitem__`` and ``__setitem__``) this runs
      ``col[i] = func(col[i])`` for all ``i`` on a thread pool and forwards
      the same, modified collection. ``func`` may modify the element in
      place (e.g. a Set row view) or return a new value. Every index is
      processed by exactly one worker.

      Use it after ``.chunk(n)`` to preprocess elements on the CPU in a
      stage of its own::

          particles.chunk(256).map_element(preprocess).map(forward)

      :param func: Function applied to each element.
      :param workers: Number of workers, or ``"auto"`` for one worker per
                      element capped at the available CPUs.
      :param executor: ``"thread"`` (default) or ``"process"``. Threads are safe
                       next to CUDA/JAX and scale when ``func`` releases the GIL
                       (NumPy, PyTorch, OpenCV); processes help for pure-Python work.
      :param start_method: Process start method (``"spawn"`` by default; only
                           with ``executor="process"``). Avoid ``"fork"`` in processes that
                           initialized CUDA or JAX.
      :param chunksize: Consecutive indices per pool task.
      :param cpu_only: Run without the GPUs of the protocol's compute
                       resources (``@resources``).



   .. py:method:: chunk(n: int, drop_last: bool = False) -> ChunkOp

      Accumulate Set[T] instances into batches of target size `n`.

      :param n: Target chunk size (number of elements in the output Set). Must be > 0.
      :param drop_last: If True, any partial remainder Set smaller than `n` upon
                        stream completion (FlushSignal) is dropped. This prevents downstream
                        JIT-compiled models from triggering recompilations for a non-standard
                        batch size.



   .. py:method:: collect(n: int) -> CollectOp

      Collect the first `n` elements of a stream of Set[T] into a single Set.

      The collected Set is emitted once, as soon as `n` elements have arrived;
      all later items are ignored. If the stream ends (FlushSignal) before `n`
      elements arrived, the elements collected so far are emitted instead.

      Use it to train a model on an initial sample of the stream::

          model = particles.collect(5_000).map(train)

      :param n: Number of elements to collect. Must be > 0.



   .. py:method:: flatten() -> FlattenOp

      Emit every element of incoming iterables as a separate item (1:N).

      Accepts any iterable, e.g. lists, tuples, generators or a Collection,
      which yields its initialized items in index order. Strings, bytes and
      mappings are rejected, as iterating them yields characters or keys.

      A Set is a batch of rows, not an iterable, and is rejected as well:
      sending its rows as separate items would be much more expensive than
      sending the batch. Use ``chunk`` to change batch sizes instead.



   .. py:method:: combine_latest(other: Op) -> CombineLatestOp

      Pair every item of this stream with the latest item of `other`.

      Emits `(item, latest)` for every item of this stream, where `latest` is
      the most recent item of `other`. Items arriving before `other` produced
      its first item are buffered and emitted once it has. Items of `other`
      only update `latest` and emit nothing themselves. Items still buffered
      at the end of the stream (FlushSignal) are dropped, as `other` never
      produced an item to pair them with.

      Use it to apply a model trained on a sample of the stream to the whole
      stream::

          model = particles.collect(5_000).map(train)
          particles.chunk(256).combine_latest(model).map(predict)

      :param other: Stream providing the latest value. Must be a different stream
                    than this one.



   .. py:method:: group_by(key: int | str, pipeline: Callable[[Op], Op], *, max_keys: Optional[int] = None, workers: scipion_bridge.core.streaming.ir.GroupByWorkers = WorkersFrom.BACKEND) -> KeyedOp[Any]
                  group_by(key: Callable[[Any], K], pipeline: Callable[[Op], Op], *, max_keys: Optional[int] = None, workers: scipion_bridge.core.streaming.ir.GroupByWorkers = WorkersFrom.BACKEND) -> KeyedOp[K]

      Run ``pipeline`` separately on the items of every key (demux).

      Every item is routed by its key into the pipeline, so that stateful
      operations (``chunk``, ``collect``, ``combine_latest``) only see the
      items of one key. The keys share a few copies of the pipeline
      (``workers``): every key is assigned to one when its first item
      arrives, in turn, and the stages of a copy keep a state per key. A
      slow key delays the other keys of its copy. Results are emitted as
      ``Keyed(key, result)``; call ``unkey()`` to continue with the merged
      stream::

          classes.flatten()
              .group_by(lambda cls: cls.class_id, pipeline=refine)
              .unkey()
              .map(write_class)

      :param key: Index or field name selecting the key of an item
                  (``item[key]``), or a function computing it. Keys must be
                  hashable.
      :param pipeline: Builds the pipeline of one key from its input stream.
                       It may only consume that input, and every branch must lead to
                       the stream it returns.
      :param max_keys: Maximum number of keys; a further key fails the
                       pipeline.
      :param workers: Number of copies of the pipeline the keys share, each
                      with stages (processes) of its own. ``None`` gives every key
                      a copy of its own. Defaults to the backend's setting.



   .. py:method:: write_to(writer: scipion_bridge.core.streaming.sink_writer.SinkWriter) -> scipion_bridge.core.streaming.sink.Sink

      Attach a terminal SinkWriter.



   .. py:method:: checkpoint(writer: scipion_bridge.core.streaming.sink_writer.SinkWriter) -> Op

      Attach an asynchronous persistence checkpoint without cutting off the stream.



   .. py:method:: sink(callback: Callable[[Any], Any]) -> scipion_bridge.core.streaming.sink.Sink

      Attach a callback-based sink (convenience for testing/debugging).



.. py:class:: FlushSignal

   Sentinel object emitted through the stream graph to trigger state flushing.


.. py:data:: FLUSH

