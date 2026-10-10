Dependency Injection
====================

All side effects of the library go through services provided by a
``dependency_injector`` container. This article explains how library code
declares the services it needs, how containers provide them, how the Ray
backend makes them available in worker processes, and how to add a service.

.. contents:: On this page
   :local:
   :depth: 2

Declaring a dependency
----------------------

Library code declares what it needs with ``Provide["service_name"]`` as the
default of a parameter, and ``@inject`` on the function:

.. code-block:: python

    from dependency_injector.wiring import Provide, inject


    @inject
    def _get_value(
        self,
        config_provider: ProtocolConfigurationProvider = Provide["protocol_config_provider"],
    ):
        return config_provider.get_value(self.name, default=self.default)

The service interfaces live in ``core/environment``:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Service
     - Interface
     - Used by
   * - ``shell_exec``
     - ``ShellExecProvider``
     - ``@shell_command``
   * - ``temp_file_provider``
     - ``TemporaryFilesProvider``
     - proxies, ARC
   * - ``storage_provider``
     - ``ArrayStorageProvider``
     - array groups (not the containers' own data)
   * - ``protocol_config_provider``
     - ``ProtocolConfigurationProvider``
     - ``Field.value``
   * - ``resource_provider``
     - ``ResourceProvider``
     - ``Resource``
   * - ``streaming_backend``
     - ``StreamingBackendProvider``
     - ``Pipeline.from_sink``

Containers and wiring
---------------------

A backend defines a ``DeclarativeContainer`` with the same service names and
*wires* it into the ``scipion_bridge`` package. Wiring replaces the
``Provide`` markers in the wired modules with the container's providers, so
that ``@inject`` functions receive the services when they are called.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Service
     - Standalone
     - Ray
     - PyWorkflow
   * - ``shell_exec``
     - ``StandaloneExecProvider``
     - ``StandaloneExecProvider``
     - ``runJob`` of the protocol
   * - ``temp_file_provider``
     - system temp dir
     - system temp dir
     - protocol ``tmp`` dir
   * - ``storage_provider``
     - Arrow
     - Arrow
     - Zarr in the protocol directory
   * - ``protocol_config_provider``
     - abstract
     - ``StaticProtocolConfigurationProvider``
     - Scipion form values
   * - ``resource_provider``
     - ``DefaultResourceProvider``
     - ``RayResourceProvider``
     - ``DefaultResourceProvider``
   * - ``streaming_backend``
     - ``RayBackend`` if installed
     - ``RayBackend``
     - ``RayBackend`` if installed

The containers are defined in ``backend/standalone/container.py``
(``Container``, wired by ``configure_default_env``),
``backend/ray/container.py`` (``RayContainer``, wired by
``configure_ray_env``) and ``backend/pyworkflow/workflow_container.py``
(``configure_pyworkflow_env``, which builds a standalone ``Container`` with
overridden providers).

The standalone container leaves ``protocol_config_provider`` abstract: outside
of a runner there are no parameter values, and reading ``Field.value`` there
is an error rather than a silent ``None``.

Services in Ray workers
-----------------------

Ray workers are separate processes, which do not inherit the driver's wiring.
Every Ray actor and task therefore wires a ``RayContainer`` with the
pipeline's parameter values before it runs user code:

* stage actors and ``group_by`` routers wire it in their constructor;
* map tasks call ``_serve_parameters`` before each call, which rewires only if
  the parameters differ from those of the previous call in the same process
  (Ray reuses worker processes for CPU tasks);
* the executors of long-running compute groups wire it once, when they
  start.

The parameter values travel with the compiled pipeline (``RayBackend``'s
``parameters``), so ``Field.value`` returns the same values on the driver and
in every worker.

Overriding services
-------------------

Every provider of a container can be overridden, which is how tests replace a
service:

.. code-block:: python

    with container.shell_exec.override(exec_mock):
        ...

    with container.protocol_config_provider.override(providers.Object(values)):
        ...

Overrides apply to the process that makes them. They do not reach Ray
workers, which wire containers of their own.

Adding a service
----------------

To add a service:

1. define its interface as an abstract class in ``core/environment``;
2. declare the dependency in library code with ``Provide["name"]`` and
   ``@inject``;
3. add a provider with the same name to every container: standalone, Ray and
   pyworkflow;
4. if the service is needed in Ray workers, make sure the worker-side wiring
   (``configure_ray_env``) provides it with the right configuration;
5. test library code against a mock of the interface by overriding the
   provider.

Keep services coarse and few. A service is justified when its implementation
genuinely differs between backends.
