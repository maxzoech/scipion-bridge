# Scipion Bridge

Scipion Bridge is a Python library for writing streaming cryo-EM protocols.
A protocol written with it runs in a notebook, as a command line program,
distributed across a [Ray](https://www.ray.io) cluster, or as a native
protocol of [Scipion 3](https://scipion.i2pc.es), without changes to its code.

```python
import scipion_bridge as B
from scipion_bridge import single_particle as spa


@B.resources(gpus=1, task=B.TaskType.LONG_RUNNING)
class FoundationModelInference(B.Protocol):
    """Embed particles with a foundation model."""

    particles: B.Input[B.Set[spa.Particle]]
    chunk_size: B.Field[int] = B.Field(default=256)
    model: B.Resource[Model] = B.Resource(builder=load_model)

    def outputs(self):
        return {"particles": B.Set[spa.FlexParticle]}

    def steps(self) -> B.Op:
        return (
            self.particles
            .chunk(self.chunk_size.value)
            .map(self._compute_latents)
            .map(self._to_output, cpu_only=True)
        )
```

## What it provides

- **Array types** (`Struct`, `Set`, `Collection`): typed, columnar containers
  backed by NumPy, Awkward Array and Apache Arrow, with defined rules for
  splitting and combining data.
- **Streaming**: a declarative API (`map`, `chunk`, `collect`,
  `combine_latest`, `group_by`, ...) lowered to a small intermediate
  representation and executed by a backend.
- **Type resolution**: a graph of converter functions between in-memory
  types, files and Scipion objects.
- **Proxies and shell commands**: typed handles for files with automatic
  cleanup, and command line programs declared as Python functions.
- **Backends**: standalone, Ray, and Scipion 3 (pyworkflow), selected through
  dependency injection.

## Installation

Scipion Bridge requires Python 3.11 or 3.12.

```bash
conda create -n scipion-bridge python=3.11
conda activate scipion-bridge
pip install -e ".[ray]"            # add ",pyworkflow" for the Scipion backend
```

## Documentation

The documentation in `docs/` covers installation, a quickstart, a
step-by-step guide for integrating protocols, the concepts of each building
block, and a developer guide. Build it with:

```bash
pip install sphinx sphinx-book-theme sphinx-autoapi nbsphinx
cd docs
make html                          # output in docs/build/html
```

## Development

```bash
pip install pytest pytest-mock pytest-cov pytest-xdist pyright black flake8
pytest -m "not ray" tests          # fast, in-process tests
pytest tests                       # full suite, including Ray tests
```

Conventions for contributors are described in `AGENTS.md` and in the
developer guide of the documentation.
