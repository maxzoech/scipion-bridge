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

    class ScipionProtocolWrapper(ProtProcessParticles, ProtFlexBase):

        _label = label
        _devStatus = BETA
        # _possibleOutputs = Outputs

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _defineParams(self, form):

            form.addSection(label="Input")
            for (name, item) in type(protocol)._exec_info.inputs.items():
                try:
                    value = protocol.__getattribute__(name)
                except AttributeError:
                    value = item(label=name)

                assert isinstance(value, Field)
                from typing import get_args
                print("args: ", get_args(value))

                form.addParam(
                    name,
                    params.IntParam,
                    label=value.label,
                    default=value.default,
                    help=value.help,
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