Developer Guide
===============

This guide is intended for developers who wish to contribute to, extend, or maintain the ``scipion_bridge`` library. It details the internal architecture, subsystems, and patterns used throughout the codebase.

Internal Architecture Overview
------------------------------

``scipion_bridge`` leverages dependency injection, graph-based type conversion, and reference counting to bridge Python environments and shell commands cleanly.

.. code-block:: text

    User Call
       │
       ├───► @proxify Decorator
       │         │
       │         ├───► Type Resolution Subsystem (Dijkstra Shortest Path)
       │         │
       │         └───► ARC Manager (File Reference Counting)
       │
       └───► @shell_command Decorator
                 │
                 ├───► AST Verification (pass body check)
                 │
                 └───► Shell Executor (dependency-injected subprocess runner)

1. Dependency Injection Container
---------------------------------

The codebase uses the ``dependency-injector`` package to manage service dependencies, defined in ``scipion_bridge/backend/standalone/container.py``. The primary services injected are:

* **``shell_exec``**: The implementation executing terminal processes. In production, this runs actual terminal commands. For testing, it is overridden with mock providers (e.g., using `pytest-mock` overrides).
* **``temp_file_provider``**: Generates and deletes temporary file paths on the filesystem.

Injecting these components decouples the library logic from direct OS and shell interactions, allowing unit tests to validate generated command structures without invoking subprocesses.

2. Type Resolution Subsystem (Graph-Based)
------------------------------------------

Located in ``scipion_bridge/core/typed/resolve.py``:

* **Registry & NetworkX**: All decorated ``@resolver`` functions are registered as directed edges in a ``networkx.DiGraph``, where nodes represent Python types.
* **Shortest Path Resolution**: When resolving a type (e.g., ``float -> str``), the system executes a custom Dijkstra pathfinding search (in ``dijkstra.py``) to locate the shortest path of resolvers.
* **MRO & Downcasting**: The registry automatically inserts downcast edges for classes based on their Method Resolution Order (MRO). This ensures that if a resolver is defined for a base class, subclasses can resolve to target types automatically.
* **Scope Prioritization**: Pathfinding prioritizes local-scope resolvers over generic/global module resolvers if multiple candidate paths exist.

3. Automatic Reference Counting (ARC) for Files
-----------------------------------------------

Located in ``scipion_bridge/core/utils/arc.py``:

* **``FileReferenceCounter``**: Manages the lifespans of files wrapped in temporary proxies. 
* **Reference Counts**:
  * Creating a new managed file via ``manager.new_managed_file()`` initializes its reference count to ``1``.
  * Re-referencing paths calls ``manager.add_reference(path)``.
  * Dereferencing paths calls ``manager.remove_reference(path)``, decrementing the count.
* **Automatic Deletion**: Once the reference count of a managed path drops to ``0``, the counter invokes ``temp_file_provider.delete(path)`` to remove the file from the disk. Non-temporary paths generate a deprecation warning if managed, to avoid deleting persistent user files.

4. AST Verification in Decorators
---------------------------------

Located in ``scipion_bridge/core/utils/shell.py``:

* **Empty Function Enforcement**: To guarantee that forward-declared external functions do not contain Python logic, the ``@shell_command`` wrapper analyzes the function's Abstract Syntax Tree (AST) using Python's standard ``ast`` module and ``autopep8``.
* **AST Inspection**: It verifies that the function's body contains exactly one node, and that this node is an instance of ``ast.Pass``. If it contains other statements, it immediately throws a ``RuntimeError`` during import/definition time.

Testing & Mocking in Development
--------------------------------

Unit tests are located in the `tests/ <file:///home/mzoch/data/Documents/scipion-bridge/tests/>`_ directory and are executed via ``pytest``.

When writing tests for shell commands or type resolvers, you must wire the container and override its providers to prevent subprocess execution:

.. code-block:: python

    import pytest
    from scipion_bridge.backend import Container

    def test_my_command(mocker):
        container = Container()
        # Wire dependency injection into the test module
        container.wire(modules=[__name__])

        exec_mock = mocker.Mock()
        
        # Override the shell executor with the mock
        with container.shell_exec.override(exec_mock):
            # Run the command
            my_proxified_command("/some/input")

            # Validate generated terminal command
            exec_mock.assert_called_with(...)
