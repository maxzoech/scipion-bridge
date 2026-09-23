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

try:
    import pwem.objects as emobj  # type: ignore
    from pwem.objects import (  # type: ignore
        SetOfParticles,
        SetOfParticlesFlex,
        ParticleFlex,
        SetOfVolumes,
        Volume,
        SetOfClasses2D,
        Class2D,
    )
    import pyworkflow.object as pywfobj  # type: ignore

    HAS_PWEM = True
except ImportError:
    HAS_PWEM = False


def reduce_minibatch_to_persistent_output(
    protocol: Any,
    key: str,
    minibatch_obj: Any,
) -> Any:
    """Reduce a stateless minibatch Scipion object into the protocol's persistent on-disk output set."""
    if not HAS_PWEM:
        raise ImportError(
            'Using scipion bridge with scipion requires pyworkflow option. Install it using pip install "scipion-bridge[pyworkflow]"'
        )

    match minibatch_obj:
        case emobj.SetOfClasses2D(): # type: ignore 
            persistent_set = getattr(protocol, key, None)
            match persistent_set:
                case None:
                    persistent_set = protocol._createSetOfClasses2D(
                        suffix=f"_{key}",
                    )
                    persistent_set.enableAppend()
                    existing_classes = {}
                case emobj.SetOfClasses2D(): # type: ignore 
                    persistent_set.enableAppend()
                    existing_classes = persistent_set._getExistingItems()
                    if existing_classes:
                        first_item = next(iter(existing_classes.values()))
                        persistent_set._getMapper().db.setupCommands(
                            first_item.getObjDict(includeClass=True),
                        )
                case _:
                    raise TypeError(
                        f"Unexpected persistent output type for {key}: {type(persistent_set)}"
                    )

            for mb_cls in minibatch_obj:
                cid = mb_cls.getObjId()
                match existing_classes.get(cid):
                    case emobj.Class2D() as target_cls:  # type: ignore 
                        if mb_cls.hasRepresentative():
                            target_cls.setRepresentative(mb_cls.getRepresentative())
                        target_cls.enableAppend()
                        for p in mb_cls:
                            item = p.clone()
                            item.setObjId(None)
                            target_cls.append(item)
                        target_cls.write()
                        target_cls._getMapper().commit()
                        persistent_set.update(target_cls)
                    case None:
                        new_cls = emobj.Class2D()  # type: ignore 
                        new_cls.setObjId(cid)
                        new_cls.copyInfo(persistent_set)
                        if mb_cls.hasRepresentative():
                            new_cls.setRepresentative(mb_cls.getRepresentative())
                        else:
                            new_cls.setRepresentative(emobj.Particle())  # type: ignore 
                        persistent_set.append(new_cls)
                        for p in mb_cls:
                            item = p.clone()
                            item.setObjId(None)
                            new_cls.append(item)
                        new_cls.write()
                        new_cls._getMapper().commit()
                        persistent_set.update(new_cls)
                        existing_classes[cid] = new_cls

            persistent_set.write()
            persistent_set._getMapper().commit()

            if hasattr(protocol, "_updateOutputSet"):
                protocol._updateOutputSet(
                    key,
                    persistent_set,
                    state=pywfobj.Set.STREAM_OPEN,  # type: ignore 
                )
            else:
                setattr(protocol, key, persistent_set)

            minibatch_obj.close()
            return persistent_set

        case emobj.SetOfParticlesFlex():  # type: ignore 
            persistent_set = getattr(protocol, key, None)
            match persistent_set:
                case None:
                    from . import resolvers

                    persistent_set = protocol._createSetOfParticlesFlex(
                        suffix=f"_{key}",
                        progName=resolvers.PROG_NAME,
                    )
                    persistent_set.getFlexInfo().setProgName(resolvers.PROG_NAME)
                    persistent_set.enableAppend()
                case emobj.SetOfParticlesFlex():  # type: ignore 
                    persistent_set.enableAppend()
                case _:
                    raise TypeError(
                        f"Unexpected persistent output type for {key}: {type(persistent_set)}"
                    )

            for p in minibatch_obj:
                item = p.clone()
                item.setObjId(None)
                persistent_set.append(item)

            persistent_set.write()
            persistent_set._getMapper().commit()

            if hasattr(protocol, "_updateOutputSet"):
                protocol._updateOutputSet(
                    key,
                    persistent_set,
                    state=pywfobj.Set.STREAM_OPEN,  # type: ignore 
                )
            else:
                setattr(protocol, key, persistent_set)

            minibatch_obj.close()
            return persistent_set

        case emobj.SetOfParticles():  # type: ignore 
            persistent_set = getattr(protocol, key, None)
            match persistent_set:
                case None:
                    persistent_set = protocol._createSetOfParticles(
                        suffix=f"_{key}",
                    )
                    persistent_set.enableAppend()
                case emobj.SetOfParticles():
                    persistent_set.enableAppend()
                case _:
                    raise TypeError(
                        f"Unexpected persistent output type for {key}: {type(persistent_set)}"
                    )

            for p in minibatch_obj:
                item = p.clone()
                item.setObjId(None)
                persistent_set.append(item)

            persistent_set.write()
            persistent_set._getMapper().commit()

            if hasattr(protocol, "_updateOutputSet"):
                protocol._updateOutputSet(
                    key,
                    persistent_set,
                    state=pywfobj.Set.STREAM_OPEN,
                )
            else:
                setattr(protocol, key, persistent_set)

            minibatch_obj.close()
            return persistent_set

        case pywfobj.Set():  # type: ignore 
            if hasattr(protocol, "_updateOutputSet"):
                protocol._updateOutputSet(
                    key,
                    minibatch_obj,
                    state=pywfobj.Set.STREAM_OPEN,  # type: ignore 
                )
            else:
                setattr(protocol, key, minibatch_obj)
            return minibatch_obj

        case _:
            if hasattr(protocol, "_defineOutputs"):
                protocol._defineOutputs(**{key: minibatch_obj})
            if hasattr(protocol, "_store"):
                protocol._store(minibatch_obj)
            setattr(protocol, key, minibatch_obj)
            return minibatch_obj


def convert_protocol_to_scipion3_protocol(
    protocol: Protocol,
    *,
    label: str,
    conda_env: str,
):
    importlib.import_module(protocol.__module__, __package__)

    try:
        import pwem  # type: ignore
        from pwem.protocols import ProtProcessParticles, ProtFlexBase  # type: ignore
        from pyworkflow.protocol import ProtStreamingBase  # type: ignore
        import pwem.objects as emobj  # type: ignore
        from pwem.objects import (  # type: ignore
            SetOfParticles,
            SetOfParticlesFlex,
            ParticleFlex,
            SetOfVolumes,
            Volume,
            SetOfClasses2D,
        )
        import pyworkflow.protocol.constants as cons  # type: ignore
        from pyworkflow.constants import BETA  # type: ignore
        from pyworkflow.plugin import Domain  # type: ignore
        from pwem.constants import ALIGN_PROJ, ALIGN_NONE  # type: ignore
        import pyworkflow.protocol.params as params  # type: ignore
        import pyworkflow.object as pywfobj  # type: ignore
        from . import resolvers
        from .utils.resolve_graph import find_pointer_class, find_output_pointer_class

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
        elif isinstance(dtype, type) and issubclass(
            dtype, (struct.Set, struct.Collection)
        ):
            pointer_class = find_pointer_class(dtype)

            if not pointer_class:
                item_type = dtype.item_type()
                item_name = (
                    item_type.__qualname__ if item_type is not None else str(dtype)
                )
                field_str = f"'{key}' " if key else ""
                raise TypeError(
                    f"Cannot bind field {field_str}with container type '{dtype}' for the Scipion 3 backend.\n"
                    f"No resolver is registered in the type registry converting a Scipion/PyWorkflow object "
                    f"(e.g., pwem.objects.Particle or SetOfParticles) to '{item_name}'.\n\n"
                    f"To fix this, define and register a resolver using `@scipion_bridge.resolver`:\n\n"
                    f"    @scipion_bridge.resolver\n"
                    f"    def resolve_scipion_particle(value: pwem.objects.Particle) -> {item_name}:\n"
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
    Outputs = Enum("Outputs", valid_outputs)  # type: ignore[misc]

    class ScipionProtocolWrapper(ProtProcessParticles, ProtFlexBase, ProtStreamingBase):

        _label = label
        _devStatus = BETA
        _possibleOutputs = Outputs
        stepsExecutionMode = cons.STEPS_PARALLEL

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
            for key, param_element in protocol.configuration.parameters.items():
                dtype = get_args(protocol._configuration.parameters[key])[0]
                param, args = _get_param_type_and_kwargs(param_element, dtype, key=key)

                form.addParam(
                    key,
                    param,
                    label=(
                        param_element.label if param_element.label is not None else key
                    ),
                    help=param_element.help,
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

        def _createSetOfClasses2D(self, imgSet=None, suffix=""):
            classes = self._EMProtocol__createSet(
                emobj.SetOfClasses2D,
                "classes2D%s.sqlite",
                suffix,
            )
            if imgSet is not None:
                classes.setImages(imgSet)
            return classes

        def _reduce_minibatch_to_persistent_output(
            self, key: str, minibatch_obj: Any
        ) -> Any:
            return reduce_minibatch_to_persistent_output(self, key, minibatch_obj)

        def _writeOutputDataHandler(self, outputData):
            from .resolvers import PyWorkflowResolutionContext

            with self._lock:
                for key, value in outputData.items():
                    pyworkflowDtype = find_output_pointer_class(type(value))
                    if pyworkflowDtype is None:
                        raise TypeError(
                            f"No Scipion output pointer class found for type {type(value)}"
                        )

                    output = resolve.current_registry().resolve(
                        value,
                        astype=pyworkflowDtype,
                        metadata=PyWorkflowResolutionContext(
                            self,
                            output_name=key,
                        ),
                    )

                    persistent_output = self._reduce_minibatch_to_persistent_output(
                        key,
                        output,
                    )

                    for input_name in self.inputTypes:
                        source = getattr(self, input_name, None)
                        if (
                            source
                            and source.hasValue()
                            and persistent_output is not None
                        ):
                            self._defineSourceRelation(source, persistent_output)

        def _submitDataStep(
            self, argname: str, inputSet: Any, unprocessed_ids: List[Any]
        ):
            from .resolvers import PyWorkflowResolutionContext

            ctx = PyWorkflowResolutionContext(
                protocol=self,
                output_name=argname,
                append=False,
                unprocessed_ids=unprocessed_ids,
            )

            with self._lock:
                bridgeSet = resolve.resolve(
                    inputSet, self.inputTypes[argname], metadata=ctx
                )

            if isinstance(bridgeSet, struct.Set):
                args = {argname: bridgeSet}
                if self._stepsPipeline is not None:
                    self._stepsPipeline.send(**args)
            else:
                raise NotImplementedError

        def _finalizeOutput(self):
            if self._stepsPipeline is not None:
                self._stepsPipeline.flush()

            with self._lock:
                self._closeOutputSet()

        def _convertInput(self):
            pass

        def stepsGeneratorStep(self) -> None:
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
                pipeline_sink = execSteps.sink(self._writeOutputDataHandler)
                self._stepsPipeline = Pipeline.from_sink(pipeline_sink)

            else:
                self._stepsPipeline = None

            profiler.start()

            validateStep = self._insertFunctionStep(
                self._validateProtocolSetup,
                prerequisites=[],
            )
            prologStep = self._insertFunctionStep(
                self._runProtocolProlog,
                prerequisites=[],
            )

            convertStep = self._insertFunctionStep(
                self._convertInput,
                prerequisites=[],
            )

            def _isFinished(key: str, *, inputs, inputIDs) -> bool:
                inputSet = inputs[key]
                readIDs = self.itemIdReadList[key]

                if inputSet is None:
                    is_finished = False
                else:
                    is_closed = (
                        inputSet.isStreamClosed()
                        if hasattr(inputSet, "isStreamClosed")
                        else False
                    )
                    is_finished = is_closed and Counter(readIDs) == Counter(
                        inputIDs[key]
                    )

                return is_finished

            previousDataStepDeps = [
                validateStep,
                prologStep,
                convertStep,
            ]  # The first step has setup as dependency
            stepDeps: List[Any] = []
            iteration = 0

            while True:
                iteration += 1

                inputs = {}
                for k in self.inputTypes.keys():
                    param_attr = getattr(self, k, None)
                    param_val = param_attr.get() if param_attr is not None else None
                    inputs[k] = param_val

                with self._lock:
                    inputIDs = {}
                    for k, v in inputs.items():
                        if v is not None and hasattr(v, "getUniqueValues"):
                            inputIDs[k] = set(v.getUniqueValues("id"))
                        else:
                            inputIDs[k] = set()

                inputsAreFinished = {
                    k: _isFinished(k, inputs=inputs, inputIDs=inputIDs)
                    for k in inputs.keys()
                }

                if all(inputsAreFinished.values()):
                    self._insertFunctionStep(
                        self._finalizeOutput, prerequisites=stepDeps
                    )
                    break

                for name, inputSet in inputs.items():
                    if inputSet is None:
                        continue

                    nonProcessedIds = inputIDs[name] - set(self.itemIdReadList[name])

                    if not nonProcessedIds:
                        continue

                    unprocessed_ids = list(nonProcessedIds)
                    self.itemIdReadList[name].extend(nonProcessedIds)

                    dataStep = self._insertFunctionStep(
                        self._submitDataStep,
                        name,
                        inputSet,
                        unprocessed_ids,
                        prerequisites=previousDataStepDeps,
                    )

                    previousDataStepDeps.append(dataStep)
                    stepDeps.append(dataStep)

                time.sleep(self.__scipion_bridge_param_polling_freq.get())

                for inputSet in inputs.values():
                    if (
                        inputSet is not None
                        and hasattr(inputSet, "isStreamOpen")
                        and inputSet.isStreamOpen()
                    ):
                        with self._lock:
                            inputSet.loadAllProperties()  # Refresh stream status

            profiler.stop()
            profiler.write_html(profilerOutput)

    # Copy the module and class name from the source protocol so Scipion class registration finds it
    ScipionProtocolWrapper.__module__ = protocol.__module__
    ScipionProtocolWrapper.__name__ = protocol.__class__.__name__
    ScipionProtocolWrapper.__qualname__ = protocol.__class__.__qualname__

    return ScipionProtocolWrapper
