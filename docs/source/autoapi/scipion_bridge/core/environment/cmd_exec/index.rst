scipion_bridge.core.environment.cmd_exec
========================================

.. py:module:: scipion_bridge.core.environment.cmd_exec


Classes
-------

.. autoapisummary::

   scipion_bridge.core.environment.cmd_exec.ShellExecProvider
   scipion_bridge.core.environment.cmd_exec.StandaloneExecProvider


Module Contents
---------------

.. py:class:: ShellExecProvider

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: run(func_name, domain, args: List[str], run_args) -> int
      :abstractmethod:



.. py:class:: StandaloneExecProvider

   Bases: :py:obj:`ShellExecProvider`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: run(func_name, domain, args: List[str], run_args)


