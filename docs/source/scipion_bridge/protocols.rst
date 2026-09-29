Protocols System
================

The ``scipion_bridge`` library includes a declarative protocol system for defining, configuring, and validating data processing pipelines. By subclassing ``Protocol``, you can specify input parameters, fields/parameters, and state variables using type annotations.

Subclassing ``Protocol``
------------------------

A protocol class is defined by subclassing ``B.Protocol`` and declaring class-level attributes annotated as ``Input``, ``Field``, or standard types. 

Every protocol must implement an abstract ``run`` method:

.. code-block:: python

    import scipion_bridge as B

    class BasicProtocol(B.Protocol):
        # 1. Inputs: Declared using B.Input[Type]
        path: B.Input[str]
        magic_number: B.Input[int] = B.Input(default=42, optional=True, label="Magic Number")

        # 2. Parameters: Declared using B.Field[Type]
        param: B.Field[float]
        param_default: B.Field[int] = B.Field(default=42)

        # 3. States: Standard Python type annotations
        state: int = 42

        def run(self, path: str, magic_number: int):
            # Run pipeline implementation
            pass

Input, Field & State Definitions
--------------------------------

* **``B.Input[Type]``**: Represents input paths, streams, or references fed into the protocol. You can initialize it with ``B.Input(...)`` to specify metadata:
  * ``default``: The default value of the input.
  * ``optional``: Boolean indicating if it's optional (defaults to ``True`` if a default is provided, ``False`` otherwise).
  * ``label``: Human-readable label.
  * ``help``: Help tooltip/description.
* **``B.Field[Type]``**: Represents configuration parameters or fields. Like ``Input``, it supports metadata overrides via ``B.Field(...)``.
* **States**: Declared as standard Python type annotations (e.g. ``state: int = 42``). State attributes are meant for internal protocol state tracking.

Validation Constraints
----------------------

The protocol system enforces strict design rules at class definition and runtime:

1. **State Type Annotations Required**: Every class-level attribute in a protocol must be type-annotated. If a variable is declared without a type annotation (e.g. ``state = 42`` instead of ``state: int = 42``), the library will raise a ``TypeError``:

   .. code-block:: python

       # This raises: TypeError("The protocol states 'state' in ... do not have type annotations.")
       class UntypedStateProtocol(B.Protocol):
           state = 42

2. **``run()`` Method Validation**: The argument names and type hints of your ``run`` method must correspond directly to the generic types declared in your protocol's ``Input[...]`` attributes. You can validate the protocol config programmatically by calling:

   .. code-block:: python

       proto = BasicProtocol()
       proto.validate_protocol_configuration() # Raises ValidationError if mismatch

Accessing Protocol Configurations
---------------------------------

You can access the generated fields and inputs of a protocol instance via the ``.configuration`` property, which returns a ``ProtocolConfiguration`` dataclass containing ordered dictionaries of the resolved fields:

.. code-block:: python

    proto = BasicProtocol()
    config = proto.configuration

    # Access inputs
    path_input = config.inputs["path"]  # Returns Input[str](optional=False)
    
    # Access parameters
    param_field = config.parameters["param_default"]  # Returns Field[int](default=42)

Exposing Protocols to Scipion
-----------------------------

You can expose and register a declared protocol as a native Scipion 3 protocol wrapper by using the ``convert_protocol_to_scipion3_protocol`` utility. This dynamically generates a Scipion-compatible protocol class with mapped GUI input/parameter forms.

The utility is imported from the ``scipion_bridge.backend.pyworkflow`` package:

.. code-block:: python

    from scipion_bridge.backend.pyworkflow import convert_protocol_to_scipion3_protocol
    from .protocol_model_inference import FoundationModelInference

    try:
        # Instantiate and convert the protocol
        FoundationModelInference = convert_protocol_to_scipion3_protocol(
            FoundationModelInference(),
            label="embed parts",
            conda_env="cryo-foundation-models"
        )
    except Exception as e:
        # Fallback if pyworkflow dependencies are not installed
        print(e)

Under the hood:
* **GUI Form Generation**: Inputs and parameters are mapped to corresponding Scipion parameter classes (e.g., `params.IntParam`, `params.FloatParam`, `params.BooleanParam`, `params.StringParam`, `params.EnumParam`) and grouped into Scipion GUI form sections ("Input" and "Parameters").
* **Environment Execution**: The generated protocol wrapper configures a specific Conda environment (``conda_env``) under which the pipeline steps run.
* **Class Registration**: The utility overrides the returned class's ``__module__`` and ``__name__`` properties to match the original protocol instance. This allows Scipion's plugin registration system to identify and list the protocol class correctly.

