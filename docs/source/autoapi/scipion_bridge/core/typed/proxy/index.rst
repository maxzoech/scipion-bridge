scipion_bridge.core.typed.proxy
===============================

.. py:module:: scipion_bridge.core.typed.proxy


Attributes
----------

.. autoapisummary::

   scipion_bridge.core.typed.proxy.Casted
   scipion_bridge.core.typed.proxy.T
   scipion_bridge.core.typed.proxy.Intermediate
   scipion_bridge.core.typed.proxy.Origin
   scipion_bridge.core.typed.proxy.P
   scipion_bridge.core.typed.proxy.R
   scipion_bridge.core.typed.proxy.ResolveProxy


Classes
-------

.. autoapisummary::

   scipion_bridge.core.typed.proxy.FuncParam
   scipion_bridge.core.typed.proxy.PathToProxyResolver
   scipion_bridge.core.typed.proxy.PathToProxyGroupResolver
   scipion_bridge.core.typed.proxy.ProxyMetaclass
   scipion_bridge.core.typed.proxy.Proxy
   scipion_bridge.core.typed.proxy.ProxyGroup
   scipion_bridge.core.typed.proxy.Output
   scipion_bridge.core.typed.proxy.ProxyProtocol


Functions
---------

.. autoapisummary::

   scipion_bridge.core.typed.proxy.namedproxy
   scipion_bridge.core.typed.proxy.proxify
   scipion_bridge.core.typed.proxy.resolve_path_to_func_param
   scipion_bridge.core.typed.proxy.resolve_str_to_func_param
   scipion_bridge.core.typed.proxy.resolve_proxy_to_func_param
   scipion_bridge.core.typed.proxy.resolve_path_to_untyped_proxy
   scipion_bridge.core.typed.proxy.resolve_output_to_proxy
   scipion_bridge.core.typed.proxy.resolve_proxy_group_to_func_param


Module Contents
---------------

.. py:data:: Casted

.. py:data:: T

.. py:data:: Intermediate

.. py:data:: Origin

.. py:data:: P

.. py:data:: R

.. py:class:: FuncParam(str_rep: str, dtype: Optional[Type[T]] = None, managed_proxy=False)

   .. py:attribute:: str_rep


   .. py:attribute:: dtype
      :value: None



   .. py:attribute:: managed_proxy
      :value: False



.. py:class:: PathToProxyResolver

   Class-based resolver transforming a Path into a single-file Proxy.


   .. py:attribute:: target_cls
      :type:  Type[Any]


   .. py:method:: forward(value: pathlib.Path) -> Any


.. py:class:: PathToProxyGroupResolver

   Bases: :py:obj:`PathToProxyResolver`


   Class-based resolver transforming a Path into a ProxyGroup.


   .. py:method:: precompute() -> None
      :classmethod:



   .. py:method:: forward(value: pathlib.Path) -> Any


.. py:class:: ProxyMetaclass

   Bases: :py:obj:`abc.ABCMeta`


   Metaclass for defining Abstract Base Classes (ABCs).

   Use this metaclass to create an ABC.  An ABC can be subclassed
   directly, and then acts as a mix-in class.  You can also register
   unrelated concrete classes (even built-in classes) and unrelated
   ABCs as 'virtual subclasses' -- these and their descendants will
   be considered subclasses of the registering ABC by the built-in
   issubclass() function, but the registering ABC won't show up in
   their MRO (Method Resolution Order) nor will method
   implementations defined by the registering ABC be callable (not
   even via super()).


.. py:class:: Proxy(path: os.PathLike, managed=False, *args, **kwargs)

   .. py:attribute:: managed
      :value: False



   .. py:property:: path
      :type: pathlib.Path



   .. py:method:: file_ext() -> Optional[str]
      :classmethod:



   .. py:method:: extensions() -> Optional[tuple[str, Ellipsis]]
      :classmethod:



   .. py:method:: prefix() -> Optional[str]
      :classmethod:



   .. py:method:: suffix() -> Optional[str]
      :classmethod:



   .. py:method:: extension() -> Optional[str]
      :classmethod:



   .. py:method:: get_referenced_paths(path_str: str) -> List[pathlib.Path]
      :classmethod:


      Return the list of filesystem paths associated with this proxy class.



   .. py:method:: from_func_param(param: FuncParam) -> Proxy
      :classmethod:


      Instantiate a proxy from a FuncParam.



   .. py:method:: from_proxy(source: Proxy, copy_data: bool = True) -> Casted
      :classmethod:


      Create a typed proxy instance from another proxy instance.



   .. py:method:: new_temporary_proxy(base_path: Optional[os.PathLike] = None, temp_file_provider: scipion_bridge.core.environment.temp_files.TemporaryFilesProvider = Provide[Container.temp_file_provider]) -> Proxy
      :classmethod:



   .. py:method:: typed(*, astype: Type[Casted], copy_data=True) -> Casted


   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated size in bytes for a single logical item contained in this proxy.

      Returns None if size cannot be determined without reading the data.


.. py:class:: ProxyGroup(base_path: os.PathLike, managed: bool = False, **kwargs: Any)

   Bases: :py:obj:`Proxy`, :py:obj:`abc.ABC`


   Abstract base class representing a grouped collection of Proxy objects.

   Subclasses must annotate child proxy fields with Proxy types and implement
   the primary_proxy abstract property.


   .. py:attribute:: base_path


   .. py:attribute:: managed
      :value: False



   .. py:method:: get_proxy_fields() -> Dict[str, Type[Proxy]]
      :classmethod:


      Return a dictionary mapping child proxy field names to their Proxy subclass types.



   .. py:method:: get_field_paths(base_path: os.PathLike) -> Dict[str, pathlib.Path]
      :classmethod:


      Return a dictionary mapping each proxy field name to its corresponding Path.



   .. py:method:: primary_proxy_type() -> Optional[Type[Proxy]]
      :classmethod:


      Return the Proxy class corresponding to primary_proxy.



   .. py:method:: extensions() -> Optional[tuple[str, Ellipsis]]
      :classmethod:



   .. py:method:: extract_base_path(primary_path: os.PathLike) -> pathlib.Path
      :classmethod:


      Extract the canonical base_path from a primary proxy path by stripping
      group/child prefixes, suffixes, and extensions.



   .. py:method:: get_referenced_paths(path_str: str) -> List[pathlib.Path]
      :classmethod:


      Return the list of child proxy paths associated with this ProxyGroup class.



   .. py:method:: from_func_param(param: FuncParam) -> ProxyGroup
      :classmethod:


      Instantiate a ProxyGroup from a FuncParam.



   .. py:method:: from_proxy(source: Proxy, copy_data: bool = True) -> Casted
      :classmethod:


      Create a typed proxy instance from another proxy instance.



   .. py:property:: primary_proxy
      :type: Proxy

      :abstractmethod:


      Return the primary proxy instance for this group.


   .. py:property:: path
      :type: pathlib.Path



   .. py:method:: new_temporary_proxy(base_path: Optional[os.PathLike] = None, temp_file_provider: scipion_bridge.core.environment.temp_files.TemporaryFilesProvider = Provide[Container.temp_file_provider]) -> ProxyGroup
      :classmethod:



   .. py:method:: keys()


   .. py:method:: values()


   .. py:method:: items()


   .. py:property:: estimated_item_nbytes
      :type: Optional[int]


      Estimated total size in bytes for a single logical item across all child proxies.

      Returns the sum of all child proxy estimates, or None if any child cannot estimate its size.


.. py:class:: Output(dtype: Type[T])

   Bases: :py:obj:`Generic`\ [\ :py:obj:`T`\ ]


   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default


   .. py:attribute:: dtype


.. py:class:: ProxyProtocol

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


   .. py:method:: file_ext() -> Optional[str]
      :classmethod:



   .. py:method:: prefix() -> Optional[str]
      :classmethod:



   .. py:method:: suffix() -> Optional[str]
      :classmethod:



.. py:function:: namedproxy(typename: str, *, file_ext: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> Type[ProxyProtocol]

.. py:type:: ResolveProxy
   :canonical: Union[Output[Intermediate], Intermediate, Origin]


.. py:function:: proxify(f: Callable[Ellipsis, Any]) -> Callable[Ellipsis, Any]

.. py:function:: resolve_path_to_func_param(value: pathlib.Path) -> FuncParam

.. py:function:: resolve_str_to_func_param(value: str) -> FuncParam

.. py:function:: resolve_proxy_to_func_param(value: Proxy) -> FuncParam

.. py:function:: resolve_path_to_untyped_proxy(value: pathlib.Path) -> Proxy

.. py:function:: resolve_output_to_proxy(value: Output) -> Proxy

.. py:function:: resolve_proxy_group_to_func_param(value: ProxyGroup) -> FuncParam

