Scipion Bridge
==============

``scipion_bridge`` is an experimental Python library designed to expose command-line tools, external utilities, and workflows (such as XMIPP programs or wget) as standard, pythonic functions. 

It handles command-line argument mapping, temporary file allocations, and type conversions automatically behind the scenes, allowing you to write clean pipeline code.

Key Features
------------

* **Declarative Wrappers**: Use the ``@shell_command`` decorator to map Python function signatures and parameter types directly to shell executions, supporting domain prefixes (e.g. ``scipion run``) or raw direct execution.
* **Flexible Type System**: Implement custom type converters with the ``@resolver`` decorator. Arguments are automatically resolved using Dijkstra's shortest path type resolution.
* **Proxy File Abstraction**: Automatically allocate, pass, and return temporary file paths using ``Proxy`` classes and ``ResolveProxy[ProxyType]`` type hints.
* **Automatic Reference Counting (ARC)**: Tracks files wrapped in managed proxies, automatically cleaning them up from disk when they are no longer referenced in Python.
* **Scipion 3 Integration**: Decorate python pipeline classes as ``Protocol`` objects and convert them into Native Scipion 3 protocols (with generated GUI forms) via the Pyworkflow backend.

Documentation Index
-------------------

* :doc:`User Guide <scipion_bridge/index>`
  Learn how to declare shell commands, use the type resolution system, work with proxies, and build declarative protocols.

* :doc:`Jupyter Examples <examples>`
  Walk through interactive notebooks downloading maps/models from EMDB/PDB, and executing thresholding pipelines.

* :doc:`Developer Guide <developer_guide>`
  Dive into codebase internals including dependency injection, reference counting implementation, AST validation, and unit testing.

* :doc:`API Reference <autoapi/index>`
  Browse auto-generated technical module reference documentation for classes, functions, and decorators.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :hidden:
   
   scipion_bridge/index
   examples

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:
   :hidden:
   
   developer_guide
   autoapi/index
