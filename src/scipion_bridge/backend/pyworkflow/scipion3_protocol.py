import os
from pathlib import Path
import importlib
import pickle
from functools import partial
from collections import Counter, defaultdict
import time
from pyinstrument import Profiler

from ...core import struct
from ...core.protocol import Protocol, Field
from ...core.typed import resolve
from ...core.streaming import Pipeline, Sink


from .workflow_container import configure_pyworkflow_env

from typing import Optional, get_args, Dict, List, Any, get_type_hints, Type, Union

from enum import Enum


def convert_protocol_to_scipion3_protocol(
    protocol: Protocol,
    *,
    label: str,
    conda_env: str,
):
    print(f"Protocol Module: {protocol.__module__}, Class: {protocol.__class__.__name__}")
    importlib.import_module(protocol.__module__, __package__)

    try:
        import pwem  # type: ignore
        from pwem.protocols import ProtProcessParticles, ProtFlexBase  # type: ignore
        from pyworkflow.protocol import ProtStreamingBase # type: ignore
        from pwem.objects import SetOfParticles, SetOfParticlesFlex, ParticleFlex, SetOfVolumes, Volume  # type: ignore
        import pyworkflow.protocol.constants as cons # type: ignore
        from pyworkflow.constants import BETA  # type: ignore
        from pyworkflow.plugin import Domain  # type: ignore
        from pwem.constants import ALIGN_PROJ, ALIGN_NONE  # type: ignore
        import pyworkflow.protocol.params as params  # type: ignore
        import pyworkflow.object as pywfobj  # type: ignore
        from .resolvers import register_pyworkflow_resolvers
        from .utils.resolve_graph import find_pointer_class, find_output_pointer_class

        register_pyworkflow_resolvers()
    except ImportError:
        raise ImportError(
            'Using scipion bridge with scipion requires pyworkflow option. Install it using pip install "scipion-bridge[pyworkflow]"'
        )

    def _get_param_type_and_kwargs(element: Field, dtype, key: Optional[str] = None):
        param_type = params.StringParam
        kwargs: Dict[str, Any] = {}

        if dtype is int:
            param_type = params.IntParam
        elif dtype is float:
            param_type = params.FloatParam
        elif dtype is bool:
            param_type = params.BooleanParam
        elif dtype is str:
            param_type = params.StringParam
        elif isinstance(dtype, type) and issubclass(dtype, Enum):
            param_type = params.EnumParam
            choices = [option.value for option in dtype]

            kwargs = {
                "choices": choices,
                "default": (
                    choices.index(element.default) if element.default in choices else 0
                ),
            }
        elif isinstance(dtype, type) and issubclass(dtype, struct.Set):
            pointer_class = find_pointer_class(dtype)

            if not pointer_class:
                item_type = dtype.item_type()
                field_str = f"'{key}' " if key else ""
                raise TypeError(
                    f"Cannot bind field {field_str}with container type '{dtype}' for the Scipion 3 backend.\n"
                    f"No resolver is registered in the type registry converting a Scipion/PyWorkflow object "
                    f"(e.g., pwem.objects.Particle or SetOfParticles) to '{item_type.__qualname__}'.\n\n"
                    f"To fix this, define and register a resolver using `@scipion_bridge.resolver`:\n\n"
                    f"    @scipion_bridge.resolver\n"
                    f"    def resolve_scipion_particle(value: pwem.objects.Particle) -> {item_type.__qualname__}:\n"
                    f"        ...\n"
                )

            param_type = params.PointerParam
            kwargs = {
                "pointerClass": pointer_class,
            }
        else:
            # Check if it matches or inherits from a Scipion object class
            raise NotImplementedError(
                f"The type {dtype} is not supported in the Scipion 3 backend."
            )

        kwargs.setdefault("default", getattr(element, "default", None))

        return param_type, kwargs

    output_types = {
        k: find_output_pointer_class(v) for k, v in protocol.outputs().items()
    }

    valid_outputs = {k: v for k, v in output_types.items() if v is not None}
    Outputs = Enum("Outputs", valid_outputs)

    class ScipionProtocolWrapper(ProtProcessParticles, ProtFlexBase, ProtStreamingBase):

        _label = label
        _devStatus = BETA
        _possibleOutputs = Outputs
        # stepsExecutionMode = cons.STEPS_PARALLEL # We want to run the steps sequentially

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.itemIdReadList = defaultdict(list)

            self.inputTypes = {
                k: get_args(v)[0] for k, v in protocol._configuration.inputs.items()
            }

        def _defineParams(self, form):

            form.addSection(label="Input")
            for key, element in protocol.configuration.inputs.items():
                dtype = get_args(protocol._configuration.inputs[key])[0]
                param, args = _get_param_type_and_kwargs(element, dtype, key=key)

                form.addParam(
                    key,
                    param,
                    label=element.label if element.label is not None else key,
                    help=element.help,
                    **args,
                )

            form.addSection(label="Parameters")
            for key, element in protocol.configuration.parameters.items():
                dtype = get_args(protocol._configuration.parameters[key])[0]
                param, args = _get_param_type_and_kwargs(element, dtype, key=key)

                form.addParam(
                    key,
                    param,
                    label=element.label if element.label is not None else key,
                    help=element.help,
                    **args,
                )

            form.addSection(label="Streaming")
            form.addParam(
                "_ScipionProtocolWrapper__scipion_bridge_param_polling_freq",  # Weird bug because we are mangaling the __module__ of the class
                params.IntParam,
                label="Polling frequency (s)",
                default=10,
                # expertLevel=cons.LEVEL_ADVANCED,
                help="Time in seconds to wait between checking for new incoming movies or data "
                "streams from the microscope/bridge.\n\n"
                "Lower values check more frequently but increase CPU/disk activity. "
                "Higher values reduce system overhead during continuous data acquisition.",
            )

            form.addParallelSection(threads=2, mpi=1)


        def _validateProtocolSetup(self):
            protocol.validate_protocol_configuration()

        def _runProtocolProlog(self):
            protocol.setup()

        def _writeOutputDataHandler(self, outputData):
            from .resolvers import PyWorkflowResolutionContext

            outputs = {}
            for key, value in outputData.items():
                pyworkflowDtype = find_output_pointer_class(type(value))
                output = resolve.current_registry().resolve(
                    value,
                    astype=pyworkflowDtype,
                    metadata=PyWorkflowResolutionContext(
                        self,
                        output_name=key,
                        append=True,
                    ),
                )

                outputs[key] = output

            # TODO: Infer data relationship here
            self._defineOutputs(**outputs)


        def _submitDataStep(self, argname: str, inputData: Union[Any, List[Any]]):
            if isinstance(inputData, struct.Set):
                args = {argname: inputData}
                self._stepsPipeline.send(**args)
            else:
                raise NotImplementedError

            print("Finished submitting data step for input:", argname)



        def _finalizeOutput(self):
            print("Finalize the output here...")
            self._stepsPipeline.flush()

        def _convertInput(self):
            print("Validate Protocol")

        def stepsGeneratorStep(self) -> None:
            import logging
            logging.basicConfig(level=logging.DEBUG)

            profilerOutput = self._getExtraPath("profiler_trace.html")
            profiler = Profiler()

            configure_pyworkflow_env(
                backend=self,
                conda_env=conda_env,
                configuration=protocol._configuration,
                modules=[__name__, type(protocol).__module__],
                packages=["scipion_bridge"],
            )

            # Create pipeline here so that the value provider is injected
            execSteps = protocol.get_pipeline()
            
            if execSteps is not None:
                execSteps = execSteps.sink(self._writeOutputDataHandler)
                self._stepsPipeline = Pipeline.from_sink(execSteps)

            else:
                self._stepsPipeline = None

            profiler.start()

            self._insertFunctionStep(self._validateProtocolSetup)
            self._insertFunctionStep(self._runProtocolProlog)

            self._insertFunctionStep(self._convertInput)

            def _isFinished(key: str, *, inputs, inputIDs) -> bool:
                inputSet = inputs[key]
                readIDs = self.itemIdReadList[key]

                return not inputSet.isStreamOpen() and Counter(readIDs) == Counter(
                    inputIDs[key]
                )

            stepDeps = []

            while True:
                
                inputs = {
                    k: getattr(self, k).get()
                    for k in self.inputTypes.keys()
                }

                with self._lock:
                    inputIDs = {
                        k: set(v.getUniqueValues("id")) for k, v in inputs.items()
                    }

                inputsAreFinished = {
                    k: _isFinished(k, inputs=inputs, inputIDs=inputIDs)
                    for k in inputs.keys()
                }

                if all(inputsAreFinished.values()):
                    self._insertFunctionStep("_finalizeOutput", prerequisites=stepDeps)
                    break

                for name, inputSet in inputs.items():
                    nonProcessedIds = inputIDs[name] - set(self.itemIdReadList[name])
                    if not nonProcessedIds:
                        continue

                    from .resolvers import PyWorkflowResolutionContext

                    ctx = PyWorkflowResolutionContext(
                        protocol=self,
                        output_name=name,
                        append=False,
                        unprocessed_ids=list(nonProcessedIds),
                    )

                    bridgeSet = resolve.resolve(inputSet, self.inputTypes[name], metadata=ctx)
                    self.itemIdReadList[name].extend(nonProcessedIds)

                    dataStep = self._insertFunctionStep("_submitDataStep", name, bridgeSet)
                    stepDeps.append(dataStep)

                time.sleep(self.__scipion_bridge_param_polling_freq)
                
                for inputSet in inputs.values():
                    if inputSet.isStreamOpen():
                        with self._lock:
                            inputSet.loadAllProperties()  # Refresh stream status

            profiler.stop()
            profiler.write_html(profilerOutput)
            

    # Copy the module and class name from the source protocol so Scipion class registration finds it
    ScipionProtocolWrapper.__module__ = protocol.__module__
    ScipionProtocolWrapper.__name__ = protocol.__class__.__name__
    ScipionProtocolWrapper.__qualname__ = protocol.__class__.__qualname__

    return ScipionProtocolWrapper
