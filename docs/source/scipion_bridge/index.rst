User Guide
=======================

The ``scipion_bridge`` module implements an experimental library for exposing
command line tools and external programs (such as XMIPP programs or wget) as Python functions.
It implements a ``@shell_command`` decorator to scaffold command line calls
and a Proxy mechanism to simplify handling temporary files and typed file paths.


.. toctree::
   :caption: Contents:
   
   shell_commands
   type_system
   protocols