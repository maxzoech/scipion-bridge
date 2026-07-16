import os
from pathlib import Path
import pickle
from functools import partial

import pwem # type: ignore
from pwem.protocols import ProtProcessParticles, ProtFlexBase # type: ignore
from pwem.objects import SetOfParticles, SetOfParticlesFlex, ParticleFlex, SetOfVolumes, Volume # type: ignore
from pyworkflow.constants import BETA, Enum # type: ignore
from pyworkflow.plugin import Domain # type: ignore
from pwem.constants import ALIGN_PROJ, ALIGN_NONE # type: ignore
import pyworkflow.protocol.params as params # type: ignore

from ...core.protocol import Protocol
from typing import Type


def convert_protocol_to_scipion3_protocol(
    protocol: Type[Protocol],
    *,
    label: str,
    conda_env: str,    
):
    class ScipionProtocolWrapper(ProtProcessParticles, ProtFlexBase):

        _label = label
        _devStatus = BETA
        # _possibleOutputs = Outputs

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _defineParams(self, form):

            form.addSection(label="Input")
            

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