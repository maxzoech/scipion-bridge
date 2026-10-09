scipion_bridge.core.protocol.chain_utils
========================================

.. py:module:: scipion_bridge.core.protocol.chain_utils

.. autoapi-nested-parse::

   Helpers to merge two protocols into a chain (``first | second``).



Exceptions
----------

.. autoapisummary::

   scipion_bridge.core.protocol.chain_utils.ChainError


Classes
-------

.. autoapisummary::

   scipion_bridge.core.protocol.chain_utils.Select


Functions
---------

.. autoapisummary::

   scipion_bridge.core.protocol.chain_utils.types_compatible
   scipion_bridge.core.protocol.chain_utils.resolve_wires
   scipion_bridge.core.protocol.chain_utils.merge_inputs
   scipion_bridge.core.protocol.chain_utils.merge_parameters
   scipion_bridge.core.protocol.chain_utils.merge_resources
   scipion_bridge.core.protocol.chain_utils.merge_pipelines


Module Contents
---------------

.. py:exception:: ChainError

   Bases: :py:obj:`TypeError`


   Raised when two protocols cannot be chained.


.. py:class:: Select(key: str)

   Picklable stage selecting one key from a protocol output dictionary.


   .. py:attribute:: key


.. py:function:: types_compatible(produced: Any, expected: Any) -> bool

   Return whether an output of type ``produced`` can feed an input of type ``expected``.


.. py:function:: resolve_wires(output_types: Mapping[str, Any], inputs: Mapping[str, scipion_bridge.core.protocol.fields.Input], mapping: Optional[Mapping[str, str]] = None) -> Dict[str, str]

   Decide which outputs of the first protocol feed which inputs of the second.

   :returns: Mapping from input name of the second protocol to the output key of the
             first protocol that feeds it.

   Resolution order: an explicit ``mapping``; else outputs and inputs with the
   same name; else, if the first protocol has a single output, the single
   type-compatible input of the second protocol.


.. py:function:: merge_inputs(first: Mapping[str, scipion_bridge.core.protocol.fields.Input], second: Mapping[str, scipion_bridge.core.protocol.fields.Input], wired: Mapping[str, str]) -> OrderedDict[str, Input]

   Inputs of the chain: those of ``first`` plus the unwired ones of ``second``.


.. py:function:: merge_parameters(first: Mapping[str, scipion_bridge.core.protocol.fields.Field], second: Mapping[str, scipion_bridge.core.protocol.fields.Field]) -> OrderedDict[str, Field]

.. py:function:: merge_resources(first: Mapping[str, scipion_bridge.core.protocol.fields.Resource], second: Mapping[str, scipion_bridge.core.protocol.fields.Resource]) -> OrderedDict[str, Resource]

.. py:function:: merge_pipelines(first_out: scipion_bridge.core.streaming.ops.Op, second_out: scipion_bridge.core.streaming.ops.Op, wires: Mapping[str, str]) -> scipion_bridge.core.streaming.ops.Op

   Connect the pipeline of the first protocol into the one of the second.

   Every source of ``second_out`` named in ``wires`` is replaced by a stage
   selecting the corresponding key from ``first_out``. Afterwards, sources with
   the same name are unified so that shared inputs are fed only once.


