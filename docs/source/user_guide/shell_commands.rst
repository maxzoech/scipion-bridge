Shell Commands
==============

.. currentmodule:: scipion_bridge

Most cryo-EM software is made of command line programs. ``@B.shell_command``
exposes such a program as a typed Python function. You declare its interface,
and the library generates the command line call.

.. code-block:: python

    @B.shell_command
    def my_program(i: str, o: str, *, select: str, substitute: str):
        pass

    my_program("in.xmd", "out.xmd", select="all", substitute="x")
    # runs: my_program -i in.xmd -o out.xmd --select all --substitute x

At the call site the program looks like any other Python function. Combined with
:doc:`proxies <proxies>`, it also takes and returns Python objects instead of
paths. The result reads like C-style functions over files.

.. contents:: On this page
   :local:
   :depth: 2

Declaring a command
-------------------

The decorated function is an **interface declaration**:

* Its name is the program name (override it with ``name=``).
* Its parameters are the program's arguments.
* Its body must be exactly ``pass``. Any other body raises ``RuntimeError``
  when the function is defined, so no Python logic can hide inside a command
  declaration.

Calling the function first checks the arguments the way Python checks any call
(missing or unexpected arguments raise ``TypeError``). It then builds the
command and runs it. A non-zero exit code raises ``RuntimeError`` with the
program's ``stderr``.

Argument translation
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Command line
   * - Positional-or-keyword, e.g. ``i``
     - ``-i VALUE``
   * - Keyword-only (after ``*``), e.g. ``select``
     - ``--select VALUE``
   * - Annotated ``bool``
     - ``--flag`` if truthy, omitted otherwise. Must be keyword-only.
   * - Value ``None``
     - Omitted entirely, which makes optional arguments easy.

Values are converted with ``str()``. Arguments appear in the order of the
function signature.

Renaming arguments
^^^^^^^^^^^^^^^^^^

Program flags are often not valid or not idiomatic Python names. Pass
``python_name="flag-name"`` pairs to the decorator to rename them. The rename
keeps the prefix that the parameter kind implies:

.. code-block:: python

    @B.shell_command(inputs="input-file", outputs="output-file")
    def xmipp_to_something(inputs: str, outputs: str, *, keyword_param: int):
        pass

    xmipp_to_something("/in", "/out", keyword_param=42)
    # runs: xmipp_to_something -input-file /in -output-file /out --keyword_param 42

Custom post-processing
^^^^^^^^^^^^^^^^^^^^^^

For programs with unusual syntax, ``postprocess_fn`` receives the list of
argument groups (``[["-i", "in"], ["--value", "42"], ...]``) and returns a
modified list:

.. code-block:: python

    def positional_output(args):
        # Drop the flag name of the first argument.
        return [[args[0][1]]] + args[1:]

    @B.shell_command(postprocess_fn=positional_output)
    def my_custom_command(outputs: str, *, value: int):
        pass

    my_custom_command("/path/out", value=42)
    # runs: my_custom_command /path/out --value 42

Domains
-------

A ``Domain`` groups programs that are invoked through a common prefix, for
example XMIPP programs run through ``scipion run``:

.. code-block:: python

    xmipp = B.Domain("XMIPP", ["scipion", "run"])
    xmipp_command = B.shell_command(domain=xmipp)

    @xmipp_command
    def xmipp_volume_align(o: str, *, i1: str, i2: str, local: bool):
        pass

    xmipp_volume_align("out.vol", i1="a.vol", i2="b.vol", local=True)
    # runs: scipion run xmipp_volume_align -o out.vol --i1 a.vol --i2 b.vol --local

Calling ``shell_command`` with only keyword arguments returns a decorator
factory that keeps them, so a domain-bound decorator can be specialized further
with ``@xmipp_command(name=..., inputs="input-file")``.

Without a domain, ``Domain.default()`` is used. It has no prefix, so the
program runs directly:

.. code-block:: python

    ModelWeights = B.namedproxy("ModelWeights", file_ext=".pth")

    @B.proxify
    @B.shell_command(output="output-document", resume="continue")
    def wget(
        url: str,
        *,
        output: B.ResolveProxy[ModelWeights] = B.Output(ModelWeights),
        resume: bool = False,
    ):
        pass

    weights = wget("https://example.org/model.pth")
    # runs: wget -url https://... --output-document /tmp/....pth

A domain created with ``isolated=True`` runs its programs in a separate conda
environment when executed by the pyworkflow backend.

A complete example
------------------

This is the command used in the :doc:`2D classification tutorial
<../tutorials/2d_classification>`:

.. code-block:: python

    fm_domain = B.Domain("foundation-models", ["fm"])

    @B.proxify
    @B.shell_command(
        domain=fm_domain,
        name="inference",
        architecture="model-type",
    )
    def compute_latents(
        *,
        particles: B.ResolveProxy[spa.ParticleStackProxy],
        weights: str,
        architecture: str,
        output: B.ResolveProxy[Latents] = B.Output(Latents),
    ):
        pass

    latents = compute_latents(
        particles=batch,                      # a Set[Particle] in memory
        weights=model_type.weights,
        architecture=model_type.architecture_name,
    )

``@proxify`` writes ``batch`` to a temporary STAR/MRC stack and allocates a
temporary output path. The executed command looks like:

.. code-block:: bash

    fm inference --particles .../neyktwgq7m12hz3.star --weights cryouni-b \
        --model-type cryo-uni --output .../c1sxkw3tftk8qtf.fmlatents

Under the pyworkflow backend, the conda environment of the protocol is
activated first. ``latents`` is a ``Latents`` proxy, and its file is deleted
when the proxy is garbage collected.

How commands are executed
-------------------------

The generated function does not call ``subprocess`` itself. It delegates to the
injected ``shell_exec`` provider (see :doc:`backends`):

* **Standalone and Ray**: ``StandaloneExecProvider`` runs the command with
  ``Popen`` in a shell and raises on failure.
* **pyworkflow**: the command runs through the Scipion protocol's ``runJob``,
  so it shows up in the protocol logs.

Tests replace the provider with a mock to check the generated command without
running anything (see :doc:`../developer_guide`).
