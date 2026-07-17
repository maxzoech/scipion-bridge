import os
from pathlib import Path
import pickle
from functools import partial

from ...core.protocol import Protocol, Field
from typing import Type, get_origin, get_args


def convert_protocol_to_scipion3_protocol(
    protocol: Protocol,
    *,
    label: str,
    conda_env: str,    
):
    try:
        import pwem # type: ignore
        from pwem.protocols import ProtProcessParticles, ProtFlexBase # type: ignore
        from pwem.objects import SetOfParticles, SetOfParticlesFlex, ParticleFlex, SetOfVolumes, Volume # type: ignore
        from pyworkflow.constants import BETA, Enum # type: ignore
        from pyworkflow.plugin import Domain # type: ignore
        from pwem.constants import ALIGN_PROJ, ALIGN_NONE # type: ignore
        import pyworkflow.protocol.params as params # type: ignore
    except ImportError:
        raise ImportError("Using scipion bridge with scipion requires pyworkflow option. Install it using pip install \"scipion-bridge[pyworkflow]\"")

    def _get_param_type_and_kwargs(dtype):
        param_type = params.StringParam
        kwargs = {}

        if dtype is int:
            param_type = params.IntParam
        elif dtype is float:
            param_type = params.FloatParam
        elif dtype is bool:
            param_type = params.BooleanParam
        elif dtype is str:
            param_type = params.StringParam
        else:
            # Check if it matches or inherits from a Scipion object class
            raise NotImplementedError
            
        return param_type, kwargs

    class ScipionProtocolWrapper(ProtProcessParticles, ProtFlexBase):

        _label = label
        _devStatus = BETA
        # _possibleOutputs = Outputs

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _defineParams(self, form):

            form.addSection(label="Input")
            for key, element in protocol.configuration.inputs.items():
                dtype = get_args(protocol._configuration.inputs[key])[0]
                param, args = _get_param_type_and_kwargs(dtype)
            
                form.addParam(
                    key,
                    param,
                    default=element.default,
                    label=element.label if element.label is not None else key,
                    help=element.help,
                    **args
                )

            form.addSection(label="Parameters")
            for key, element in protocol.configuration.parameters.items():
                dtype = get_args(protocol._configuration.parameters[key])[0]
                param, args = _get_param_type_and_kwargs(dtype)

                form.addParam(
                    key,
                    params.IntParam,
                    default=element.default,
                    label=element.label if element.label is not None else key,
                    help=element.help,
                    **args,
                )
            

        def _insertAllSteps(self):
            import logging
            logging.basicConfig(level=logging.DEBUG)

            print("Start inserting steps here")

            # configure_pyworkflow_env(
            #     backend=self,
            #     conda_env="foundation-models",
            #     modules=[__name__],
            #     packages=["scipion_bridge"],
            # )

    # Copy the module from the source protocol so Scipion class registration finds it
    ScipionProtocolWrapper.__module__ = protocol.__module__

    return ScipionProtocolWrapper