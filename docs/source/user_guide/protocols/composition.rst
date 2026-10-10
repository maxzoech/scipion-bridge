.. _chaining-protocols:

Composing Protocols
===================

.. currentmodule:: scipion_bridge

Protocols are most useful when each does one thing. Composition joins them
into a single pipeline, so that a workflow such as *embed, then classify* runs
as one streaming job without intermediate files. This article describes how
protocols are chained, how their declarations are merged, and how a
protocol's declarations can be inspected.

.. contents:: On this page
   :local:
   :depth: 2

Chaining
--------

Two protocols compose into one with ``|`` or ``pipe``:

.. code-block:: python

    pipeline = FoundationModelInference() | ClusterLatents()

    pipeline = FoundationModelInference().pipe(
        ClusterLatents(),
        mapping={"particles": "latents"},      # output of first -> input of second
    )

The result is a ``ChainedProtocol``, itself a protocol, which runs on every
backend and can be chained further. Its steps are merged into a single
streaming graph: every output dictionary of the first protocol is validated,
the wired outputs are selected from it, and each one becomes an item of the
corresponding input of the second protocol.

Wiring outputs to inputs
^^^^^^^^^^^^^^^^^^^^^^^^

Outputs of the first protocol are connected to inputs of the second in this
order of precedence:

1. by the explicit ``mapping`` (output name to input name), if one is given;
2. otherwise, by matching names;
3. otherwise, if the first protocol has a single output, to the single input
   of the second protocol whose type is compatible.

An output feeds an input only if their types are compatible: the same type, or
a subclass with the same element type. Ambiguous or impossible wirings raise
``ChainError`` (a ``TypeError``) when the chain is created, naming the outputs
and inputs involved.

Merged declarations
^^^^^^^^^^^^^^^^^^^

* **Inputs** of the chain are the inputs of the first protocol plus the
  unwired inputs of the second. An input that both declare with the same type
  is shared.
* **Parameters** and **resources** of both protocols are merged. A name that
  both declare must have the same definition in both, in which case it is
  shared; otherwise the chain raises ``ChainError``, and one of them must be
  renamed.
* **Outputs** are those of the second protocol.
* **Compute resources** stay per protocol: the maps of each protocol keep
  their own ``@B.resources`` declaration, so a GPU protocol and a CPU protocol
  can be chained without the CPU stages claiming GPUs.

Because parameters are merged by name, two protocols with a ``chunk_size``
field of different defaults cannot be chained as they are. Give such fields
protocol-specific names (``embedding_chunk_size``) when protocols are meant to
be combined.

Introspection
-------------

``protocol.configuration`` returns the bound declarations, in declaration
order:

.. code-block:: python

    config = FoundationModelInference().configuration
    config.inputs["particles"]          # the bound Input, with dtype Set[Particle]
    config.parameters["chunk_size"]     # the bound Field, with default 10000
    config.resources["model"]           # the Resource

    config.parameters["chunk_size"].dtype       # int
    config.parameters["chunk_size"].default     # 10000

Backends use the configuration to build Scipion forms and command line
parsers, to precompute input resolvers, and to check parameter values. It is
equally useful in tests and in tools that generate documentation for
protocols.
