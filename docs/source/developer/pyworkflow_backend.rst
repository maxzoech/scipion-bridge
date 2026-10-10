PyWorkflow Backend Internals
============================

``backend/pyworkflow`` turns a protocol into a Scipion 3 streaming protocol.
This article describes how the generated class is built, how it streams data
from Scipion sets into the pipeline and back, how conversions between Scipion
objects and the array types are implemented, and how support for a new
Scipion type is added. The user-facing behavior is described in
:doc:`/user_guide/backends/pyworkflow`.

.. contents:: On this page
   :local:
   :depth: 2

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Contents
   * - ``scipion3_protocol.py``
     - ``convert_protocol_to_scipion3_protocol``, the generated
       ``ScipionProtocolWrapper``, and the reduction of output batches into
       persistent Scipion sets
   * - ``workflow_container.py``
     - The Scipion implementations of the services and
       ``configure_pyworkflow_env``
   * - ``resolvers.py``
     - Resolvers between Scipion objects and the single particle types, and
       ``PyWorkflowResolutionContext``
   * - ``utils/resolve_graph.py``
     - ``find_pointer_class`` and ``find_output_pointer_class``, which derive
       Scipion classes from the resolution graph

The module imports ``pyworkflow`` and ``pwem`` only when they are available;
the resolvers are registered only if ``pwem`` can be imported, and are lifted
into the ``scipion_bridge.backend.pyworkflow`` package.

Generating the class
--------------------

``convert_protocol_to_scipion3_protocol(protocol, label=..., conda_env=...)``
imports the protocol's module, determines the Scipion classes of the outputs
with ``find_output_pointer_class``, and defines ``ScipionProtocolWrapper``, a
subclass of ``ProtProcessParticles``, ``ProtFlexBase`` and
``ProtStreamingBase`` with parallel step execution. The class takes the
``__name__``, ``__qualname__`` and ``__module__`` of the original protocol, so
Scipion's plugin discovery registers it.

``_defineParams`` builds the form from ``protocol.configuration``:

* inputs become ``PointerParam`` entries whose ``pointerClass`` is found by
  ``find_pointer_class``;
* fields become ``IntParam``, ``FloatParam``, ``BooleanParam``,
  ``StringParam`` or ``EnumParam`` (with the enum values as choices);
* a *Streaming* section adds the polling interval. Its attribute name is
  mangled (``_ScipionProtocolWrapper__scipion_bridge_param_polling_freq``)
  because the class's module is rewritten.

Deriving Scipion classes from the graph
---------------------------------------

``find_pointer_class(T)`` collects the predecessors of ``T`` in the
resolution graph (and, for ``Set`` and ``Collection`` types, of their element
type) and maps each to a Scipion set class: a Scipion set is used directly, a
Scipion item class (``EMObject``) is mapped to the sets whose ``ITEM_TYPE`` it
is. If several candidates remain, subclasses are discarded in favor of their
bases and the first by name is chosen, with a warning.

``find_output_pointer_class(T)`` does the same with the successors of ``T``.
Only direct edges are considered, which is why an output type needs a resolver
whose source is exactly that type.

Streaming execution
-------------------

The generated protocol defines its steps dynamically in
``stepsGeneratorStep``:

1. It wires the Scipion container (``configure_pyworkflow_env``) into the
   library and the protocol's module, with the Scipion protocol instance as
   backend: ``_PyWorkflowExecProvider`` runs commands through ``runJob``
   (prefixed with the conda activation for isolated domains, and with Xmipp's
   environment for the ``XMIPP`` domain), ``_PyWorkflowTempFileProvider``
   creates files in the protocol's ``tmp`` directory, and
   ``_PyWorkflowProtocolConfigurationProvider`` reads parameter values from
   the protocol's attributes, converting them to the declared types
   (``convert_scipion_to_python``).
2. It compiles the protocol's pipeline with ``Pipeline.from_sink``, with a sink
   calling ``_writeOutputDataHandler``.
3. It inserts the validation, setup and conversion steps.
4. It polls. In each iteration, it reads the ids of every input set, inserts a
   ``_submitDataStep`` for the ids not read before, sleeps for the polling
   interval, and reloads the properties of open input sets. When every input
   set is closed and all its ids have been read, it inserts
   ``_finalizeOutput`` and stops.

``_submitDataStep`` resolves the input set into the declared input type with
a ``PyWorkflowResolutionContext`` carrying the unprocessed ids, so the
resolver reads only the new items (with an SQL ``WHERE`` clause on the set's
database), and sends the result into the pipeline.

``_finalizeOutput`` flushes the pipeline and closes the output sets.

Writing outputs
---------------

``_writeOutputDataHandler`` receives each output dictionary. Under the
protocol's lock, it resolves each value into its Scipion output class, with a
context carrying the output name, and passes the result to
``reduce_minibatch_to_persistent_output``, which merges it into the
protocol's persistent output:

* ``SetOfParticles`` and ``SetOfParticlesFlex``: the items of the batch are
  cloned and appended to the persistent set, created on the first batch with
  metadata copied from the input set; sampling rate, acquisition and
  dimensions are filled from the batch where missing;
* ``SetOfClasses2D``: classes are matched by id; particles are appended to
  existing classes and representatives replaced, new classes are created;
* other ``Set`` objects replace the output; other objects are defined as
  outputs.

The output set is updated in streaming state (``STREAM_OPEN``), and a source
relation to every input is defined.

Resolvers and the resolution context
------------------------------------

``PyWorkflowResolutionContext`` is passed as ``metadata`` and carries:

* ``protocol``: the running Scipion protocol, needed to create sets in its
  directory (``_createSetOfParticles``) and to find its inputs;
* ``output_name``: the output being written, used in set file names;
* ``unprocessed_ids``: the ids of the input items to convert;
* ``append``: reserved for appending to existing sets.

Input resolvers read the set's SQLite database directly for the requested
ids and fill whole columns of a preallocated ``Set``, rather than iterating
Scipion objects, which keeps conversion fast for large batches. Output
resolvers create a new set per batch, which the reduction step merges into the
persistent output.

Adding support for a Scipion type
---------------------------------

To let protocols consume and produce a new kind of Scipion object:

1. Define or choose the array type (a struct and its ``Set`` or
   ``Collection``).
2. Write an input resolver from the Scipion set to the array type, accepting
   ``metadata: Optional[PyWorkflowResolutionContext]`` and converting only
   ``metadata.unprocessed_ids`` when given.
3. Write an output resolver from the array type to the Scipion set, creating
   the set through ``metadata.protocol`` so that it is stored in the
   protocol's directory.
4. If the Scipion set needs merging other than appending items, extend
   ``reduce_minibatch_to_persistent_output``.
5. Register the resolvers in ``resolvers.py`` (inside the ``HAS_PWEM`` guard)
   or in the plugin that needs them.

``find_pointer_class`` and ``find_output_pointer_class`` pick up the new
resolvers automatically.

Testing
-------

Most of the backend can be tested without Scipion by passing mock objects
that implement the methods the code calls. The configuration provider, for
example, is tested with a mock protocol whose attributes have a ``get``
method (``tests/scipion_bridge/protocols/test_protocol.py``). Resolver tests
that need ``pwem`` live in ``tests/scipion_bridge/backend/pyworkflow/``; they
run in an environment with the ``pyworkflow`` extra and are skipped
elsewhere.
