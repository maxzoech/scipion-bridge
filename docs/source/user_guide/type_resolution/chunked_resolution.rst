Chunked Resolution
==================

.. currentmodule:: scipion_bridge

Inputs of cryo-EM protocols are often far larger than memory. Chunked
resolution reads such inputs piece by piece: the conversion that produces the
data runs once per chunk, on only that chunk. This article explains
slice-aware resolvers, how a path is split into a part that runs once and a
part that runs per chunk, and how the chunk size is chosen.

.. contents:: On this page
   :local:
   :depth: 2

Slice-aware resolvers
---------------------

A resolver that accepts a ``slice`` keyword argument is **slice-aware**: it
can produce just one part of its output.

.. code-block:: python

    @B.resolver
    def stack_to_particles(
        value: B.ParticleStackProxy,
        *,
        slice: slice | None = None,
    ) -> B.Set[Particle]:
        ...   # read only images [slice] from the memory-mapped MRC stack

When ``slice`` is ``None``, the resolver produces the whole output. A
slice-aware resolver should read only the requested part from its source,
for example by memory-mapping a file, so that the cost of a chunk is
proportional to its size.

Resolving in chunks
-------------------

When a resolution path contains a slice-aware step, the value can be resolved
in chunks with ``B.resolve_iter``:

.. code-block:: python

    for chunk in B.resolve_iter(
        Path("particles.star"),
        astype=B.Set[Particle],
        chunk_size=1_000,
    ):
        pipeline.send(particles=chunk)

The path is split at its first slice-aware step:

.. code-block:: text

    Path ──► StarfileProxy ──► ParticleStackProxy ══► Set[Particle]
    └──────────── runs once ────────────────────┘ └── per chunk ──┘

* The steps before the first slice-aware step run once. Their result, the
  *intermediate*, must support ``len()``, which gives the number of elements.
* The slice-aware step and the steps after it run once per chunk. The
  slice-aware step receives ``slice(k, k + chunk_size)``; later steps receive
  the chunk without a slice.

A ``ComposedResolver`` from ``B.find_resolver`` offers the same through its
``iter`` method. Resolving a path without a slice-aware step in chunks raises
``TypeError``.

When several paths have the same length, slice-aware ones are preferred, so
adding a slice-aware resolver does not require removing an existing,
non-slicing one.

Choosing the chunk size
-----------------------

Without an explicit ``chunk_size``, the chunk size is estimated so that each
chunk holds about ``target_bytes`` of data (32 MiB by default). The estimate
uses the ``estimated_item_nbytes`` property of the intermediate: the size of
one element in bytes. ``MRCStackProxy`` computes it from the MRC header, and
``ParticleStackProxy`` forwards it. If the intermediate has no estimate,
chunks of 100 elements are used.

``B.estimate_optimal_chunk_size(item_nbytes, target_bytes=None)`` exposes the
same calculation.

The Ray runner resolves its inputs in chunks and reads two environment
variables: ``SCIPION_CHUNK_SIZE`` sets the chunk size directly, and
``SCIPION_TARGET_BYTE_SIZE`` sets the target size of a chunk.

The input chunk size is independent of the batch sizes inside a protocol.
Chunks are only the unit in which data is read and sent into the pipeline;
``chunk(n)`` in ``steps()`` re-batches them to the size the computation needs.
Chunks of a few tens of megabytes keep the per-item overhead of the pipeline
low without holding much data in flight.

Caching across chunks
---------------------

In chunked resolution, one instance of each resolver class serves all
chunks of a stream. A class resolver can therefore parse a metadata file once
and reuse it for every chunk:

.. code-block:: python

    @B.resolver
    class stack_to_particles:
        def __init__(self) -> None:
            self._star_cache: dict[Path, Any] = {}

        def forward(self, value: B.ParticleStackProxy, *, slice=None) -> B.Set[Particle]:
            star = self._star_cache.setdefault(value.metadata.path, read_star(value))
            ...

The built-in resolver from ``ParticleStackProxy`` to ``Set[Particle]`` works
this way.
