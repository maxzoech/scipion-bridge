Contributing
============

This article summarizes the coding conventions of the library, the checks a
change must pass, and how to document it. The authoritative list of
conventions is ``AGENTS.md`` at the root of the repository.

.. contents:: On this page
   :local:
   :depth: 2

Conventions
-----------

The library is written for Python 3.11 and follows modern Python patterns and,
where applicable, the Google style guide. In particular:

**Formatting.** Code is formatted with ``black``. Function calls spanning
several lines end with a trailing comma, which makes ``black`` keep one
argument per line:

.. code-block:: python

    buffer = np.zeros((self._capacity, *stacked.shape[1:]), dtype=entry.dtype,)

**Types.** The code base is annotated and type checked with ``pyright``.
Functions with several call signatures have correct ``@overload``
declarations.

**Pattern matching.** ``match`` statements are preferred over chains of
``if`` / ``elif`` with ``isinstance`` checks:

.. code-block:: python

    match key:
        case int(index):
            ...
        case slice() as index:
            ...
        case _ if _is_int_sequence(key):
            ...

**Flat control flow.** Conditionals are nested at most two levels deep.
Guard clauses keep functions flat, and ``if`` / ``else`` branches are
explicit: code after an early ``return`` that acts as an implicit ``else`` is
avoided.

**Fail early.** No overly defensive code: instead of
``getattr(a, "bar", None)`` with a fallback, assert the type and access the
attribute. Genuine bugs crash early and consistently.

**Exceptions for exceptional cases.** Expected cases are handled with
conditionals, not with exceptions used for control flow. At the same time,
the code follows the EAFP style where an operation is expected to succeed and
failure is exceptional.

**Imports at the top.** Circular imports are avoided by the structure of the
packages, not by ``TYPE_CHECKING`` guards or imports inside functions.

**Existing abstractions.** Standard functions are used over magic attributes,
for example ``typing.get_origin`` and ``get_args`` instead of
``__origin__``.

**Comprehensions** are preferred over loops that build lists, dicts and sets.

**No special cases for tests.** Code never contains branches that exist only
for unit tests. Tests use mocks that implement the methods the code calls.

Before submitting a change
--------------------------

Run the checks that CI runs, and the tests that cover the change (see
:doc:`testing`):

.. code-block:: bash

    black .
    pyright
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    pytest -m "not ray" tests
    pytest tests/scipion_bridge/streaming/test_<affected module>.py

Commits are made on a branch, not on ``main``.

Documenting a change
--------------------

The documentation lives in ``docs/source`` and is built with Sphinx (see
:doc:`/getting_started/installation`). Docstrings use the Google style and
appear in the generated module reference.

Each kind of change has a natural place in the documentation:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Change
     - Document in
   * - New or changed public behavior
     - The concept page in :doc:`/user_guide/structs/index` and its siblings,
       and :doc:`/api` if the public namespace changes
   * - Something a protocol author must do differently
     - The step of :doc:`/integration/index` it affects
   * - A behavior that is easy to get wrong
     - :doc:`/user_guide/sharp_bits`
   * - An internal mechanism or contract
     - The article of this developer guide for that layer

The documentation is written in a neutral, technical register: it states what
the code does and why, gives complete examples, and links to the page that
covers a topic in depth rather than repeating it.
