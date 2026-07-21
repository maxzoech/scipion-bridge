Exposing External Programs
==========================

The ``@shell_command`` decorator is used to expose command line programs and external tools as Python functions. By default, programs are called using their function name directly, but you can configure them to run under a specific execution prefix using a ``Domain``.

When exposing a command line program, you declare a Python function with the desired signature, type annotations, and a body consisting of only a single ``pass`` statement. The library validates the function structure and constructs the scaffolding to map arguments and execute the underlying command in a subprocess.

Mapping Python to Shell Commands
--------------------------------

To map an external program, define a function with the same name as the program you want to expose, and decorate it using ``@shell_command``:

.. code-block:: python

    import scipion_bridge as B

    @B.shell_command
    def my_program(i: str, o: str, *, select: str, substitute: str):
        pass

This forward-declared function has no python implementation. When you call this function, it will execute:

.. code-block:: bash

    my_program -i ... -o ... --select ... --substitute ...

Argument Translation Rules:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Positional Arguments**: Positional parameters of the Python function (which can be passed positionally or as keywords) map to single-dash options (e.g., `-i` or `-o`).
2. **Keyword-Only Arguments**: Parameters after the ``*`` operator can only be passed as keyword arguments. They map to double-dash options (e.g., ``--select``).
3. **Boolean Flags**: Parameters annotated with the ``bool`` type map to boolean flags. If the argument value is ``True``, the parameter name is added as a double-dash flag (e.g., ``--apply``). If the value is ``False``, the flag is omitted entirely. Note that positional arguments cannot be annotated as ``bool`` flags; they must be keyword-only.
4. **Empty Function Rule**: The decorated function must only contain a single ``pass`` statement. If there are other statements or return values inside the body, the library will raise a ``RuntimeError`` during definition.

Domains & Command Prefixes
--------------------------

A ``Domain`` defines a command prefix under which programs are executed. For example, XMIPP tools are invoked via the ``scipion run`` command.

You can configure a domain and use it to decorator-wrap multiple programs:

.. code-block:: python

    import scipion_bridge as B

    # Define a custom domain
    xmipp_domain = B.Domain("XMIPP", ["scipion", "run"])

    # Create a decorator bound to the domain
    xmipp_func = B.shell_command(domain=xmipp_domain)

    @xmipp_func
    def xmipp_volume_align(o: str, *, i1: str, i2: str, local: bool):
        pass

Calling ``xmipp_volume_align`` runs the following command:

.. code-block:: bash

    scipion run xmipp_volume_align -o ... --i1 ... --i2 ... --local

Executing without a Domain (e.g., ``wget``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not specify a ``domain`` parameter on ``@shell_command``, the default domain (``Domain.default()``) is used. The default domain has an empty command prefix list, meaning the command is executed directly.

For example, to execute ``wget`` directly without any domain prefix, you can write:

.. code-block:: python

    import scipion_bridge as B

    # Define a named proxy for type validation
    ModelWeights = B.namedproxy("ModelWeights", file_ext=".pth")

    @B.proxify
    @B.shell_command(url="i", output="output-file", resume="continue")
    def wget(
        url: str,
        *,
        output: B.ResolveProxy[ModelWeights] = B.Output(ModelWeights),
        resume: bool = False,
    ):
        pass

In the above example:
* Since no domain is specified, ``wget`` runs directly from the shell.
* Parameter mappings are passed as keyword arguments to ``@shell_command`` (e.g., `url` maps to `-i`, `output` maps to `--output-file`, and `resume` maps to `--continue`).
* The function utilizes the ``@proxify`` decorator to automatically allocate and manage temporary files for `output`.

Remapping Parameter Names
-------------------------

Often, external command argument names are not pythonic or conflict with Python keywords. You can specify a custom mapping by passing parameter names and their command equivalents directly to the decorator:

.. code-block:: python

    @xmipp_func(inputs="input-file", outputs="output-file")
    def xmipp_to_something(inputs: str, outputs: str, *, keyword_param: int):
        pass

Calling ``xmipp_to_something("/in", "/out", keyword_param=42)`` will run:

.. code-block:: bash

    scipion run xmipp_to_something -input-file /in -output-file /out --keyword_param 42

Custom Argument Postprocessing
------------------------------

For programs that do not follow standard argument structures, you can pass a custom callable to ``postprocess_fn``. This callable takes a list of argument pairs and returns the modified arguments:

.. code-block:: python

    # Example: strip the parameter flag name from the first argument
    def remove_output_label(args):
        return [[args[0][1]]] + args[1:]

    @xmipp_func(postprocess_fn=remove_output_label)
    def my_custom_command(outputs: str, *, value: int):
        pass

Calling ``my_custom_command("/path/to/output", value=42)`` executes:

.. code-block:: bash

    scipion run my_custom_command /path/to/output --value 42
