Standalone Backend
==================

.. currentmodule:: scipion_bridge

The standalone backend is the default container. It is what scripts,
notebooks and tests use when they work with the library directly, outside of a
runner or of Scipion.

.. code-block:: python

    from scipion_bridge.backend import configure_default_env

    container = configure_default_env()     # wires the scipion_bridge package

Services
--------

* **Commands** run with ``subprocess.Popen`` in a shell. The command is printed
  before it runs, and a non-zero exit code raises ``RuntimeError`` with the
  program's standard error.
* **Temporary files** are created in the system temporary directory.
* **Resources** are built on first access and cached per process.
* **Parameter values** have no provider: the container leaves
  ``protocol_config_provider`` abstract, because outside of a runner there are
  no parameter values to serve. Code that reads ``Field.value`` needs one; see
  below.
* **Pipelines** are compiled with the Ray backend if Ray is installed.

Wiring
------

Wiring replaces the ``Provide[...]`` declarations in the library's modules
with the container's services. ``configure_default_env`` wires the
``scipion_bridge`` package; pass ``modules=[...]`` or ``packages=[...]`` to
wire your own code as well, if it declares services with ``Provide``.

Overriding services
-------------------

Every service is a provider of the container and can be overridden, for the
duration of a ``with`` block or permanently. Tests use this to check the
commands a function would run without running them:

.. code-block:: python

    with container.shell_exec.override(exec_mock):
        my_command("/in", value=42)

and to provide parameter values to protocol code:

.. code-block:: python

    from dependency_injector import providers
    from scipion_bridge.core.environment import StaticProtocolConfigurationProvider

    values = StaticProtocolConfigurationProvider({"chunk_size": 256})
    with container.protocol_config_provider.override(providers.Object(values)):
        protocol.chunk_size.value         # 256

:doc:`/integration/testing` builds a test setup on these overrides.
