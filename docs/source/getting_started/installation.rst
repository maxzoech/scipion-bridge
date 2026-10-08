Installation
============

Scipion Bridge requires **Python 3.11 or 3.12**. Install it in a conda
environment (or any other virtual environment).

Core library
------------

The core library contains the array types, type resolution, proxies, shell
commands, and the streaming API. Install an editable copy from a checkout of the
repository:

.. code-block:: bash

    conda create -n scipion-bridge python=3.11
    conda activate scipion-bridge
    pip install -e .

Optional extras
---------------

Backends are optional extras. Install the ones you need:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Installs
     - Use it to
   * - ``ray``
     - ``ray``, ``tensorstore``
     - Run streaming protocols in parallel on one machine or distributed on a
       Ray cluster, and write results to Zarr with ``TensorStoreSinkWriter``.
   * - ``pyworkflow``
     - ``scipion-pyworkflow``, ``scipion-em``, ``zarr``, ``xmipp-metadata``
     - Convert protocols into Scipion 3 protocols.

.. code-block:: bash

    pip install -e ".[ray]"
    pip install -e ".[ray,pyworkflow]"

Using the pyworkflow backend
----------------------------

The ``pyworkflow`` backend must be installed into the conda environment of
Scipion 3, usually called ``scipion3``.

.. note::

   The Scipion 3 environment must use Python 3.11 or later, as required by
   this library.

Development tools
-----------------

The test suite uses ``pytest``. The code is type checked with ``pyright``,
formatted with ``black``, and linted with ``flake8``:

.. code-block:: bash

    pip install pytest pytest-mock pytest-xdist pyright black flake8

    pytest tests/
    pyright
    black .
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

Building this documentation
---------------------------

.. code-block:: bash

    pip install sphinx sphinx-book-theme sphinx-autoapi nbsphinx
    cd docs
    make html        # output in docs/build/html

Verify the installation
-----------------------

.. code-block:: python

    >>> import scipion_bridge as B
    >>> from scipion_bridge import single_particle as spa
    >>> spa.Particle.print_schema()
    /
    ├── pixels: Array[float32], shape: [None, None]
    ├── ctf (struct)
    │   ├── defocus_u: Array[float64], shape: [1]
    ...
