Tutorial: Streaming 2D Classification
=====================================

This tutorial builds a streaming 2D classification workflow that uses a
foundation model to speed up classification. It uses most of the library:
array types, resources, stateful streaming operations, and shell commands.

.. code-block:: text

    Compute embeddings      a neural network embeds every particle
           │
           ▼
         Cluster            cluster the embeddings into classes
           │
           ▼
         AlignPCA           align and average the particles of each class

The goal is to classify tens of millions of particles in a streaming
workflow: particles are processed as they arrive, and the classes are refined
continuously.

.. note::

   Function bodies that are specific to the model or the clustering method are
   elided (``...``). The focus is on how the protocol is structured.

.. contents:: Steps
   :local:
   :depth: 1

Step 1: Embedding particles
---------------------------

The first protocol embeds every particle with a foundation model. Particles
arrive as a stream of ``Set[Particle]`` batches of arbitrary size. ``chunk``
turns them into batches of the size the GPU wants.

.. code-block:: python

    from enum import Enum

    import numpy as np
    import torch
    import scipion_bridge as B
    from scipion_bridge import single_particle as spa


    def load_model(protocol: "FoundationModelInference") -> torch.nn.Module:
        model_type = protocol.model_type.value
        model = ...                    # load the weights for model_type
        return model.eval().cuda()


    @B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
    class FoundationModelInference(B.Protocol):
        """Embed particles with a foundation model."""

        class ModelType(Enum):
            CRYO_IEF_SMALL = "Cryo-IEF (Base)"

        # Inputs
        particles: B.Input[B.Set[spa.Particle]]

        # Fields
        model_type: B.Field[ModelType] = B.Field(default=ModelType.CRYO_IEF_SMALL)
        chunk_size: B.Field[int] = B.Field(default=256)

        # Resources: loaded once per worker process
        model: B.Resource[torch.nn.Module] = B.Resource(builder=load_model)

        def outputs(self):
            return {"particles": B.Set[spa.FlexParticle]}

        def steps(self) -> B.Op:
            return (
                self.particles
                .chunk(self.chunk_size.value)
                .map(self._compute_latents)
                .map(self._map_output, cpu_only=True)
            )

        def _compute_latents(self, batch: B.Set[spa.Particle]):
            images = torch.as_tensor(np.asarray(batch["pixels"])).cuda()
            with torch.no_grad():
                latents = self.model(images).cpu().numpy()
            return batch, latents

        def _map_output(self, batch_and_latents):
            batch, latents = batch_and_latents
            flex = B.Set[spa.FlexParticle](capacity=len(batch))
            ...                          # copy the particle fields, set embeddings
            flex["embeddings"] = latents
            return {"particles": flex}

Notes on this protocol:

* ``@B.resources(gpus=1, task=LONG_RUNNING)`` makes the Ray backend hold one GPU
  for the protocol's stages for the whole stream, so the model is loaded once.
* ``model`` is a ``Resource``. Its builder runs lazily, inside the worker that
  first reads ``self.model``, not on the driver.
* The two ``map`` calls are separate stages. While the GPU embeds batch
  *n + 1*, the CPU builds the output of batch *n*. ``cpu_only=True`` keeps the
  second stage from waiting for the GPU.

Calling the model as an external program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the model is only available as a command line tool, declare it with
``@B.shell_command`` and let ``@B.proxify`` handle the files:

.. code-block:: python

    Latents = B.namedproxy("Latents", file_ext=".fmlatents")
    fm_domain = B.Domain("foundation-models", ["fm"])


    @B.proxify
    @B.shell_command(domain=fm_domain, name="inference", architecture="model-type")
    def compute_latents(
        *,
        particles: B.ResolveProxy[spa.ParticleStackProxy],
        weights: str,
        architecture: str,
        output: B.ResolveProxy[Latents] = B.Output(Latents),
    ):
        pass

Inside ``_compute_latents``, pass the in-memory batch directly. It is written to
a temporary STAR file and MRC stack, which are deleted again after the call:

.. code-block:: python

    def _compute_latents(self, batch):
        model_type = self.model_type.value
        latents = compute_latents(
            particles=batch,
            weights=model_type.weights,
            architecture=model_type.architecture_name,
        )
        return batch, read_latents(latents.path)

With this variant, every batch round-trips through files, so prefer larger
chunks (for example ``chunk_size=10_000``).

Step 2: Training a classifier on a sample
-----------------------------------------

Clustering needs a model trained on a representative sample before it can
classify anything. This is a **stateful** stream operation:

1. ``collect(n)`` gathers the first ``n`` embedded particles into a single set
   and emits it once.
2. ``map(self._cluster_latents)`` trains the classifier on that sample.
3. ``combine_latest(classifier)`` pairs every later chunk with the most recent
   classifier. Chunks that arrive before training finishes are buffered.

.. code-block:: python

    from sklearn.cluster import KMeans


    class ClusterLatents(B.Protocol):
        particles: B.Input[B.Set[spa.FlexParticle]]

        num_classes: B.Field[int] = B.Field(default=50)
        cluster_sample_size: B.Field[int] = B.Field(default=50_000)
        chunk_size: B.Field[int] = B.Field(default=10_000)

        def outputs(self):
            return {"classes": B.Set[spa.Class2D]}

        def steps(self) -> B.Op:
            classifier_training = (
                self.particles
                .collect(self.cluster_sample_size.value)
                .map(self._cluster_latents)
            )

            return (
                self.particles
                .chunk(self.chunk_size.value)
                .combine_latest(classifier_training)
                .map(self._classify_chunk)
                .flatten()
                .group_by(lambda cls: cls.class_id, pipeline=self._refine_class)
                .unkey()
                .map(self._to_output)
            )

        def _cluster_latents(self, sample: B.Set[spa.FlexParticle]):
            embeddings = np.asarray(sample["embeddings"])
            return KMeans(n_clusters=self.num_classes.value).fit(embeddings)

``self.particles`` is used twice. Both uses refer to the same input, which
**fans out** to the training branch and the classification branch.

Step 3: Splitting the stream by class
-------------------------------------

``_classify_chunk`` receives ``(chunk, classifier)``. It assigns every particle
to a class and returns one ``Class2D`` per class present in the chunk.
``flatten()`` then emits each of them as a separate item:

.. code-block:: python

        def _classify_chunk(self, chunk_and_model):
            chunk, model = chunk_and_model
            labels = model.predict(np.asarray(chunk["embeddings"]))
            return [
                spa.Class2D(particles=chunk[labels == k], class_id=int(k))
                for k in np.unique(labels)
            ]

``group_by`` routes each ``Class2D`` into a sub-pipeline **for its class**, so
stateful operations inside the sub-pipeline only see the particles of one class.
The sub-pipeline for a class is created when its first particles arrive:

.. code-block:: python

        def _refine_class(self, class_stream: B.Op) -> B.Op:
            return (
                class_stream
                .map(lambda cls: cls.particles)
                .chunk(2_000)
                .map(self._align_pca)
            )

        def _align_pca(self, particles: B.Set[spa.FlexParticle]) -> spa.Class2D:
            ...   # align the particles, compute the representative image
            return spa.Class2D(particles=particles, class_id=..., representative=...)

The groups are processed in parallel. ``unkey()`` merges them back into one
stream of ``Keyed(class_id, Class2D)`` items:

.. code-block:: python

        def _to_output(self, keyed):
            return {"classes": B.Set[spa.Class2D]([keyed.value])}

Step 4: Chaining and running
----------------------------

Chain the two protocols into one pipeline. The ``particles`` output of the
first is wired to the ``particles`` input of the second by name:

.. code-block:: python

    classification = FoundationModelInference() | ClusterLatents()

Run it on Ray, persisting the classes to Zarr. Set columns are appended as
classes are refined, following the type system's reduction semantics:

.. code-block:: python

    from pathlib import Path
    from scipion_bridge.backend.ray import RayPipelineRunner
    from scipion_bridge.backend.ray.sinks import TensorStoreSinkWriter

    with RayPipelineRunner(
        classification,
        parameters={"num_classes": 100, "cluster_sample_size": 100_000},
        sink=TensorStoreSinkWriter("classes.zarr"),
    ) as runner:
        runner.run(particles=Path("particles.star"))

At the end, the runner prints per-stage statistics. If the embedding stage
shows a high ``process_s`` while the stages after it are mostly ``idle_s``, the
GPU is the bottleneck: add GPUs or use a lower precision.

To run either protocol inside Scipion instead, convert it with
``convert_protocol_to_scipion3_protocol`` (see :ref:`scipion-backend`).

Results
-------

The prototype of this workflow produced the 2D classes of a synthetic dataset
with protocols of no more than about 200 lines of code each. With the Ray
backend, the embedding step ran at 76 particles/s in FP32 and 300 particles/s
in FP16.
