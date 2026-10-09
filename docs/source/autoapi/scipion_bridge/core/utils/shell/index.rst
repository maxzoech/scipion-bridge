scipion_bridge.core.utils.shell
===============================

.. py:module:: scipion_bridge.core.utils.shell


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.utils.shell.F


Classes
-------

.. autoapisummary::

   scipion_bridge.core.utils.shell.ShellDecoratorProtocol


Functions
---------

.. autoapisummary::

   scipion_bridge.core.utils.shell.shell_command


Module Contents
---------------

.. py:data:: F

.. py:class:: ShellDecoratorProtocol

   Bases: :py:obj:`Protocol`


   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing).

   For example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto(Protocol[T]):
           def meth(self) -> T:
               ...


.. py:function:: shell_command(f: F, *, domain: scipion_bridge.core.environment.domain.Domain = Domain.default(), name: Optional[str] = None, postprocess_fn: Optional[Callable] = None, **args_map) -> F
                 shell_command(f: None = None, *, domain: scipion_bridge.core.environment.domain.Domain = Domain.default(), name: Optional[str] = None, postprocess_fn: Optional[Callable] = None, **args_map) -> ShellDecoratorProtocol

