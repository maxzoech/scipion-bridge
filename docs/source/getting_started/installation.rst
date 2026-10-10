Installation
============

Scipion Bridge is a pure Python package. It requires **Python 3.11 or 3.12**
and is installed from a checkout of its repository. Which optional
dependencies you need depends on where your protocols will run, so this page
first describes the parts of the installation and then the typical setups.

Parts of the installation
-------------------------

The package is split into a core and two optional extras.

**The core** contains everything needed to define and test protocols: the
array types, the streaming API, type resolution, proxies, shell commands and
the single particle types. It depends on NumPy, Awkward Array, PyArrow,
``networkx``, ``dependency-injector``, ``mrcfile`` and ``starfile``, among
others.

**The extras** add execution backends:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Extra
     - Installs
     - Needed to
   * - ``ray``
     - ``ray``, ``tensorstore``
     - Run streaming pipelines in parallel on one machine or distributed on a
       Ray cluster, and write results to Zarr with ``TensorStoreSinkWriter``.
       Without Ray, a pipeline can be defined but not executed.
   * - ``pyworkflow``
     - ``scipion-pyworkflow``, ``scipion-em``, ``zarr``, ``xmipp-metadata``
     - Convert protocols into Scipion 3 protocols, and convert data between
       Scipion objects (``SetOfParticles``, ``SetOfClasses2D``, ...) and the
       array types.

Ray is the only streaming backend that ships with the library. Install the
``ray`` extra unless you only use the array types, type resolution and shell
commands.

Setting up a development environment
------------------------------------

For writing and testing protocols, create a dedicated conda environment (any
other virtual environment works as well) and install an editable copy of the
repository with the Ray extra:

.. code-block:: bash

    conda create -n scipion-bridge python=3.11
    conda activate scipion-bridge

    git clone <repository URL> scipion-bridge
    cd scipion-bridge
    pip install -e ".[ray]"

An editable install (``-e``) makes changes to the source tree visible without
reinstalling, which is convenient both for protocol authors who keep their
protocols next to the library and for library developers.

Installing into a Scipion 3 environment
---------------------------------------

To run protocols inside Scipion, the library must be importable from the
Python environment that Scipion uses, usually a conda environment named
``scipion3``. Install it there together with the ``pyworkflow`` extra, and
with the ``ray`` extra if the protocol's pipeline should run on Ray:

.. code-block:: bash

    conda activate scipion3
    pip install -e ".[ray,pyworkflow]"

.. note::

   Scipion 3 installations use Python 3.8 by default, but Scipion Bridge
   requires Python 3.11 or later. The Scipion environment must therefore be
   created with, or upgraded to, a newer Python version before the library can
   be installed into it.

The external programs that a protocol calls (for example a foundation model
command line tool) usually live in conda environments of their own. They do
not need Scipion Bridge installed. How the Scipion backend activates those
environments is described in :doc:`/integration/deployment`.

Verifying the installation
--------------------------

Import the library and print the layout of one of the built-in types:

.. code-block:: python

    >>> import scipion_bridge as B
    >>> from scipion_bridge import single_particle as spa
    >>> spa.Particle.print_schema()
    /
    ├── pixels: Array[float32], shape: [None, None]
    ├── ctf (struct)
    │   ├── defocus_u: Array[float64], shape: [1]
    ...

To check that the Ray backend can start, compile and run a trivial pipeline:

.. code-block:: python

    from scipion_bridge.backend.ray import RayBackend
    from scipion_bridge.core.streaming import Pipeline, Source

    out = Source("numbers").map(lambda x: x * 2).sink(print)
    with Pipeline.from_sink(out, backend=RayBackend()) as pipeline:
        pipeline.send(numbers=21)       # prints 42
    pipeline.close()

``RayBackend`` starts a local Ray instance on first use. Ray needs local
networking for its processes to find each other; in a sandbox or container
that blocks it, ``ray.init()`` times out during startup.

Tools for development
---------------------

Contributors and protocol authors who write tests need the test and quality
tools:

.. code-block:: bash

    pip install pytest pytest-mock pytest-cov pytest-xdist pyright black flake8

The code is type checked with ``pyright``, formatted with ``black`` and
linted with ``flake8``. :doc:`/developer/testing` explains how the test suite
is organized and which subset of it to run for a change.

Building this documentation
---------------------------

The documentation is built with Sphinx:

.. code-block:: bash

    pip install sphinx sphinx-book-theme sphinx-autoapi nbsphinx
    cd docs
    make html        # output in docs/build/html

The module reference is generated from the docstrings by ``sphinx-autoapi``.
The example notebooks are rendered but not executed during the build.

Next steps
----------

Continue with the :doc:`quickstart`, which builds a first streaming protocol,
and then read the :doc:`/user_guide/overview` for the architecture behind it.
