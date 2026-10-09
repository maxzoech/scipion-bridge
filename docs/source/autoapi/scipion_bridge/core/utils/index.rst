scipion_bridge.core.utils
=========================

.. py:module:: scipion_bridge.core.utils


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/scipion_bridge/core/utils/arc/index
   /autoapi/scipion_bridge/core/utils/ast/index
   /autoapi/scipion_bridge/core/utils/format/index
   /autoapi/scipion_bridge/core/utils/func_params/index
   /autoapi/scipion_bridge/core/utils/marker/index
   /autoapi/scipion_bridge/core/utils/shell/index
   /autoapi/scipion_bridge/core/utils/type_annotation/index


Classes
-------

.. autoapisummary::

   scipion_bridge.core.utils.Domain


Functions
---------

.. autoapisummary::

   scipion_bridge.core.utils.shell_command


Package Contents
----------------

.. py:function:: shell_command(f: F, *, domain: scipion_bridge.core.environment.domain.Domain = Domain.default(), name: Optional[str] = None, postprocess_fn: Optional[Callable] = None, **args_map) -> F
                 shell_command(f: None = None, *, domain: scipion_bridge.core.environment.domain.Domain = Domain.default(), name: Optional[str] = None, postprocess_fn: Optional[Callable] = None, **args_map) -> ShellDecoratorProtocol

.. py:class:: Domain

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: command
      :type:  List[str]


   .. py:attribute:: isolated
      :type:  bool
      :value: False



   .. py:method:: default() -> Domain
      :classmethod:



