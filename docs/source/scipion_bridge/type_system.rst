Type System
===========

The ``scipion_bridge`` module features an integrated type system that automatically converts "normal" Python types (like numbers, collections, arrays, or files) into the format expected by downstream commands and functions. 

The type system is divided into two layers:
1. **Type Resolution**: The core engine that defines type conversion rules and automatically chains them together using a shortest-path graph search.
2. **Proxies**: An extension of the type resolution system designed specifically to map python objects and file references to paths on disk, managing temporary file lifespans.

Type Resolution
---------------

By establishing a graph of type conversion rules, you can define how various python objects should be converted. The library then determines the shortest path of conversions to transform inputs automatically.

Resolvers & The ``@resolver`` Decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To define a type conversion rule, implement a function and decorate it with ``@resolver``. The function **must** have type annotations for its input parameter and its return type:

.. code-block:: python

    import scipion_bridge as B

    @B.resolver
    def resolve_float_to_int(value: float) -> int:
        # Convert a float to an integer
        return int(value)

    @B.resolver
    def resolve_int_to_string(value: int) -> str:
        # Convert an integer to a string
        return str(value)

Parameter Resolution via ``@resolve_params``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To automatically resolve arguments passed to a function, decorate the function with ``@resolve_params`` and annotate the target parameters with ``Resolve[TargetType]`` (or ``Resolve[TargetType, IntermediateType]``):

.. code-block:: python

    import scipion_bridge as B

    @B.resolve_params
    def process_data(value: B.Resolve[str]):
        # Inside the function, `value` is guaranteed to be a string
        print(f"Type: {type(value)}, Value: {value}")

    # Calling the function with an integer:
    # 1. Matches `int -> str` (built-in or user-defined)
    process_data(42)  # Prints: Type: <class 'str'>, Value: 42

    # Calling the function with a float:
    # 1. Finds the shortest path: float -> int -> str
    # 2. Applies `resolve_float_to_int` then `resolve_int_to_string`
    process_data(42.5)  # Prints: Type: <class 'str'>, Value: 42

Behind the Scenes: The Resolver Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under the hood, the type resolution system maintains a directed graph where:
* **Nodes** represent Python classes/types.
* **Edges** represent the registered resolver functions.

When a function decorated with ``@resolve_params`` is called, the library uses Dijkstra's shortest path algorithm to locate and chain together the appropriate resolver functions to transform the argument from its runtime type to the target annotation type.

Default Resolvers
^^^^^^^^^^^^^^^^^

Out of the box, ``scipion_bridge`` provides several common resolvers:
* **General string conversion**: Resolves any object to a string using ``str(value)``.
* **Collections to space-separated strings**: Resolves a `tuple` to a space-separated string by resolving and joining its elements (e.g. ``(1, 2, 3)`` resolves to ``"1 2 3"``).

Proxies
-------

Proxies extend the type resolution system to wrap file paths on disk and provide an abstraction over input and output files for command line programs, turning C-style command arguments into ergonomic, object-oriented Python code.

Using the ``@proxify`` decorator, functions that accept file path strings can automatically receive and return ``Proxy`` objects.

Overview
^^^^^^^^

With proxies, instead of manually passing temporary file paths and cleaning them up, you can chain operations cleanly:

.. code-block:: python

    # Plain command line style (requires manual path management)
    xmipp_image_resize("/path/to/input.vol", "/path/to/output.vol", dim=size)

    # Proxified style
    input_volume = ... # Can be a path or a Proxy object
    output = xmipp_image_resize(input_volume, dim=size)

When a ``Proxy`` object is passed to a ``@proxify``-decorated function, the wrapper extracts the underlying file path and passes it to the function. For outputs, you can pass an ``Output`` object; the wrapper will automatically instantiate a temporary proxy, pass its path to the function, and return the proxy object.

Defining Proxy Types
^^^^^^^^^^^^^^^^^^^^

You define custom proxy types by subclassing ``Proxy`` and implementing the ``file_ext`` classmethod:

.. code-block:: python

    import scipion_bridge as B
    from typing import Optional

    class Volume(B.Proxy):
        @classmethod
        def file_ext(cls) -> Optional[str]:
            return ".vol"

    class TextFile(B.Proxy):
        @classmethod
        def file_ext(cls) -> Optional[str]:
            return ".txt"

Alternatively, you can dynamically create a proxy class using ``namedproxy``:

.. code-block:: python

    PosFile = B.namedproxy("PosFile", file_ext=".pos")

Using @proxify
^^^^^^^^^^^^^^

To enable proxy support on a function, decorate it with ``@proxify``. Use the ``ResolveProxy[ProxyType]`` type annotation for input parameters, and set default output parameters to ``Output(ProxyType)``:

.. code-block:: python

    import scipion_bridge as B

    @B.proxify
    def benchmark_filter_and_resize(
        inputs: B.ResolveProxy[Volume],
        output: B.ResolveProxy[Volume] = B.Output(Volume)
    ) -> B.Proxy:
        # Inside the function, inputs and output are resolved to their raw file path strings
        xmipp_transform_filter(inputs, output, fourier="low_pass 0.1")
        return output

    # When calling the function, you can pass a Proxy, Path, or string:
    my_volume = Volume("/path/to/input.vol")
    result_volume = benchmark_filter_and_resize(my_volume)
    
    # result_volume is returned as a Volume proxy object wrapping a managed temporary file:
    # e.g., <Volume for /tmp/temp_file_0.vol (managed)>

Automatic File Cleanup
^^^^^^^^^^^^^^^^^^^^^^

Proxies can be either **managed** (owned) or **unmanaged** (unowned):

* **Managed**: The underlying file on disk is tracked by the library's automatic reference count manager. When the Python ``Proxy`` object is garbage collected, the file is automatically deleted from disk. Temporary files created via ``Output(...)`` are always managed.
* **Unmanaged**: The proxy wraps a path but does not delete the file when deallocated. Proxies initialized from existing paths default to unmanaged.

Type Conversion and Reassignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have an untyped or differently typed proxy, you can convert it using ``typed()``:

.. code-block:: python

    # Convert untyped proxy to a TextFile proxy (copies the file contents by default)
    typed_proxy = untyped_proxy.typed(astype=TextFile)

Handling Path-Modifying Programs
""""""""""""""""""""""""""""""""

Some programs append their own file extensions to output paths (e.g., a program might write to ``/path/to/file.vol`` when passed ``/path/to/file``). 

To handle this, you can pass an untyped proxy (which has no extension) and convert it after execution with ``copy_data=False``:

.. code-block:: python

    @B.proxify
    def path_modifying_func(outputs: B.ResolveProxy = B.Output(B.Proxy)):
        # Receives "/tmp/temp_file_0" and the program writes to "/tmp/temp_file_0.vol"
        pass

    # Call function and type the resulting proxy to locate the correct file on disk
    result = path_modifying_func().typed(astype=Volume, copy_data=False)

Using ``copy_data=False`` re-maps the proxy to reference the path with the extension (``/tmp/temp_file_0.vol``) without performing any disk copies.

Reference Counting Support
^^^^^^^^^^^^^^^^^^^^^^^^^^

``Proxy`` objects support duplicate references to the same file path. If multiple ``Proxy`` instances wrap the same path and are marked as ``managed=True``, the library's internal Automatic Reference Counting (ARC) manager tracks them. The file on disk is only deleted when all referencing ``Proxy`` instances have been garbage collected (when the reference count drops to 0).

