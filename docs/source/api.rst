API Overview
============

This page lists the public API, grouped by topic. Everything in the first
sections is available from the top-level namespace:

.. code-block:: python

    import scipion_bridge as B

The full, generated reference of every module is in the
:doc:`autoapi/index`.

Array types
-----------

.. list-table::
   :widths: 30 70

   * - :py:obj:`B.Struct <scipion_bridge.core.struct.struct.Struct>`
     - Base class for element types. Annotated attributes become fields.
   * - :py:obj:`B.Set[T] <scipion_bridge.core.struct.set.Set>`
     - A column-wise batch of ``T`` structs. Supports slicing, masks, index
       arrays, and field access.
   * - :py:obj:`B.Collection[T] <scipion_bridge.core.struct.collection.Collection>`
     - A fixed-size, row-wise sequence of ``T`` structs.
   * - :py:obj:`B.Array[dtype] <scipion_bridge.core.struct.struct.Array>`
     - An array field with fixed rank: ``B.Array(shape=(None, None))``.
   * - :py:obj:`B.Dim / B.Arg <scipion_bridge.core.struct.struct.Arg>`
     - A named dimension, shared between array shapes.
   * - :py:obj:`B.concat <scipion_bridge.core.struct.set.concat>`
     - Concatenate sets with identical schemas.

Common methods: ``schema()``, ``print_schema()``, ``to_arrow()``,
``from_arrow(batch)``, ``is_initialized(...)``.

Protocols
---------

.. list-table::
   :widths: 30 70

   * - :py:obj:`B.Protocol <scipion_bridge.core.protocol.protocol_base.Protocol>`
     - Base class. Implement ``steps()`` and ``outputs()``.
   * - :py:obj:`B.Input[T] <scipion_bridge.core.protocol.fields.Input>`
     - Declares an input stream. Bound inputs act as a ``Source``.
   * - :py:obj:`B.Field[T] <scipion_bridge.core.protocol.fields.Field>`
     - Declares a parameter. Read it with ``.value``.
   * - :py:obj:`B.Resource[T] <scipion_bridge.core.protocol.fields.Resource>`
     - Declares lazily built, read-only state.
   * - :py:obj:`B.ResourceScope <scipion_bridge.core.environment.resource_provider.ResourceScope>`
     - ``PROCESS`` (per worker) or ``SHARED`` (per cluster).
   * - :py:obj:`B.resources <scipion_bridge.core.protocol.protocol_base.resources>`
     - Class decorator declaring GPUs, VRAM, CPUs and the task type.
   * - :py:obj:`B.ComputeResources <scipion_bridge.core.environment.compute.ComputeResources>`,
       :py:obj:`B.TaskType <scipion_bridge.core.environment.compute.TaskType>`
     - The compute declaration and ``EPHEMERAL`` / ``LONG_RUNNING``.
   * - :py:obj:`B.ChainedProtocol <scipion_bridge.core.protocol.protocol_base.ChainedProtocol>`
     - The result of ``first | second`` or ``first.pipe(second, mapping=...)``.

Streaming
---------

.. list-table::
   :widths: 30 70

   * - :py:obj:`B.Op <scipion_bridge.core.streaming.ops.Op>`
     - A stream. Methods: ``map``, ``map_batch``, ``map_element``, ``chunk``,
       ``collect``, ``flatten``, ``combine_latest``, ``group_by``, ``sink``,
       ``write_to``, ``checkpoint``.
   * - :py:obj:`B.FLUSH <scipion_bridge.core.streaming.node.FLUSH>`,
       :py:obj:`B.FlushSignal <scipion_bridge.core.streaming.node.FlushSignal>`
     - The end-of-stream signal.

From ``scipion_bridge.core.streaming``:

.. list-table::
   :widths: 30 70

   * - :py:obj:`Source <scipion_bridge.core.streaming.ops.Source>`
     - A named entry point of a graph.
   * - :py:obj:`Pipeline <scipion_bridge.core.streaming.pipeline.Pipeline>`
     - ``Pipeline.from_sink(*sinks, backend=...)``, then ``send``, ``flush``,
       ``stats``, ``close``.
   * - :py:obj:`SinkWriter <scipion_bridge.core.streaming.sink_writer.SinkWriter>`
     - The protocol for asynchronous output writers (``write``, ``finalize``).
   * - :py:obj:`Keyed <scipion_bridge.core.streaming.ir.Keyed>`
     - A ``(key, value)`` result of ``group_by``.
   * - :py:obj:`StageStats <scipion_bridge.core.streaming.backend.StageStats>`
     - Per-stage execution metrics.
   * - :py:obj:`StreamingBackendProvider <scipion_bridge.core.streaming.backend.StreamingBackendProvider>`,
       :py:obj:`CompiledPipeline <scipion_bridge.core.streaming.backend.CompiledPipeline>`
     - Interfaces for implementing backends.

Shell commands and proxies
--------------------------

.. list-table::
   :widths: 30 70

   * - :py:obj:`B.shell_command <scipion_bridge.core.utils.shell.shell_command>`
     - Declare a command line program as a Python function.
   * - :py:obj:`B.Domain <scipion_bridge.core.environment.domain.Domain>`
     - A command prefix shared by several programs.
   * - :py:obj:`B.proxify <scipion_bridge.core.typed.proxy.proxify>`
     - Resolve ``ResolveProxy`` arguments and allocate ``Output`` files.
   * - :py:obj:`B.Proxy <scipion_bridge.core.typed.proxy.Proxy>`,
       :py:obj:`B.ProxyGroup <scipion_bridge.core.typed.proxy.ProxyGroup>`
     - A typed file, and a bundle of files.
   * - :py:obj:`B.namedproxy <scipion_bridge.core.typed.proxy.namedproxy>`
     - Create a proxy type from a name and an extension.
   * - :py:obj:`B.ResolveProxy[P] <scipion_bridge.core.typed.proxy.ResolveProxy>`
     - A parameter annotation: resolve to proxy ``P`` and pass its path.
   * - :py:obj:`B.Output(P) <scipion_bridge.core.typed.proxy.Output>`
     - A parameter default: allocate a managed temporary ``P`` and return it.
   * - ``B.ParticleStackProxy``, ``B.StarfileProxy``, ``B.MRCStackProxy``
     - Proxies for single particle data.

Type resolution
---------------

.. list-table::
   :widths: 30 70

   * - :py:obj:`B.resolver <scipion_bridge.core.typed.resolve.resolver>`
     - Register a converter function or class.
   * - :py:obj:`B.resolve <scipion_bridge.core.typed.resolve.resolve>`
     - Resolve a value: ``B.resolve(value, astype=T)``.
   * - :py:obj:`B.resolve_iter <scipion_bridge.core.typed.resolve.resolve_iter>`
     - Resolve a value in chunks through a slice-aware resolver.
   * - :py:obj:`B.find_resolver <scipion_bridge.core.typed.resolve.find_resolver>`
     - Precompute a :py:obj:`ComposedResolver <scipion_bridge.core.typed.resolve.ComposedResolver>`.
   * - :py:obj:`B.resolve_params <scipion_bridge.core.typed.resolve.resolve_params>`,
       ``B.Resolve[T]``
     - Resolve annotated function parameters on every call.
   * - :py:obj:`B.lift_resolvers <scipion_bridge.core.typed.resolve.lift_resolvers>`
     - Make the resolvers of a submodule visible from a parent package.
   * - :py:obj:`B.estimate_optimal_chunk_size <scipion_bridge.core.typed.resolve.estimate_optimal_chunk_size>`
     - The chunk size that targets a byte budget.

Single particle analysis
------------------------

``scipion_bridge.single_particle`` provides ``Particle``, ``FlexParticle``,
``CTF``, ``Coordinate``, ``Acquisition`` and ``Class2D``, the single particle
proxies, and the resolvers between them. See :doc:`user_guide/structs`.

Backends
--------

.. list-table::
   :widths: 40 60

   * - ``scipion_bridge.backend.configure_default_env``
     - Wire the standalone container.
   * - ``scipion_bridge.backend.ray.RayPipelineRunner``
     - Run a protocol on Ray, from Python or as a command line program.
   * - ``scipion_bridge.backend.ray.RayBackend``
     - The Ray streaming backend.
   * - ``scipion_bridge.backend.ray.sinks.TensorStoreSinkWriter``
     - Persist results to Zarr through TensorStore.
   * - ``scipion_bridge.backend.pyworkflow.convert_protocol_to_scipion3_protocol``
     - Generate a Scipion 3 protocol class.
