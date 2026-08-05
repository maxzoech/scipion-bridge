"""Type resolvers converting PyWorkflow/pwem objects (Particle, etc.) to scipion-bridge types."""

import os
import sys
import uuid
from typing import Dict
import numpy as np
import mrcfile
import time

from dataclasses import dataclass
from typing import Optional, Any

from tqdm import tqdm

try:
    import pwem.objects as emobj  # type: ignore
    from pwem.protocols import ProtFlexBase
    from pwem.emlib.image import ImageHandler  # type: ignore

    HAS_PWEM = True
except ImportError:
    HAS_PWEM = False
    ProtFlexBase = Any

PROG_NAME = "scipion_bridge"


@dataclass
class PyWorkflowResolutionContext:

    protocol: ProtFlexBase
    output_name: Optional[str]
    append: bool


def register_pyworkflow_resolvers():
    """Register all PyWorkflow object resolvers in the type resolution graph."""
    if not HAS_PWEM:
        return

    from ...core.typed.resolve import resolver, lift_resolvers
    from ...core import struct
    from ... import single_particle as spa

    @resolver
    def resolve_scipion_particle_to_bridge_particle(
        value: emobj.Particle,
    ) -> spa.Particle:
        """Convert a Scipion/pwem Particle object to a scipion-bridge Particle struct."""
        if value.getFileName():
            ih = ImageHandler()
            img = ih.read(value)
            pixel_data = img.getData().astype(np.float32)
        else:
            raise ValueError(
                "Could not convert Scipion 3 particle to Scipion Bridge particle"
            )

        return spa.Particle(pixels=pixel_data)

    @resolver
    def resolve_embeddings_to_flex_particles(
        value: struct.Set[spa.FlexParticle],
        metadata: PyWorkflowResolutionContext,
    ) -> emobj.SetOfParticlesFlex:
        if metadata is None:
            raise ValueError(
                "The Scipion Protocol is required as context to resolve Embeddings to SetOfParticlesFlex."
            )

        def _get_exisiting_set() -> Optional[emobj.SetOfParticlesFlex]:
            if metadata.output_name is not None:
                return getattr(metadata.protocol, metadata.output_name, None)
            else:
                return None

        # Create the Scipion 3 SetOfParticlesFlex
        outImgSet = _get_exisiting_set()
        output_name = metadata.output_name or uuid.uuid4().hex

        if outImgSet is not None and metadata.append == True:
            start_index = len(outImgSet)
            value = value[start_index:]
        else:
            outImgSet = metadata.protocol._createSetOfParticlesFlex(
                suffix=f"_{output_name}", progName=PROG_NAME
            )
            outImgSet.getFlexInfo().setProgName(PROG_NAME)
            start_index = 0

        stack_uuid = uuid.uuid4().hex
        stack_path = metadata.protocol._getExtraPath(
            f"output_{output_name}_{stack_uuid}.mrcs"
        )

        t0 = time.perf_counter()
        pixels_arr = np.array(value["pixels"], dtype=np.float32)
        t_array_conv = time.perf_counter() - t0

        t0 = time.perf_counter()
        mrcfile.write(
            stack_path,
            pixels_arr,
            overwrite=False,
        )
        
        t_mrc_write = time.perf_counter() - t0
        t_loop_start = time.perf_counter()

        # Micro-breakdown of operations inside the loop
        t_tolist_total = 0.0

        for i, particle in enumerate(value, start=1):
            outParticle = emobj.ParticleFlex(progName=PROG_NAME)
            outParticle.getFlexInfo().setProgName(PROG_NAME)
            outParticle.setLocation(i, stack_path)
            
            # Measure list conversion specifically if embeddings is a heavy NumPy array
            t_conv_0 = time.perf_counter()
            z_flex_list = particle.embeddings.tolist()
            t_tolist_total += time.perf_counter() - t_conv_0
            
            outParticle.setZFlex(z_flex_list)
            outImgSet.append(outParticle)

        t_loop_total = time.perf_counter() - t_loop_start

        print(f"\n--- Execution Benchmark ---")
        print(f"NumPy Array Conversion : {t_array_conv * 1000:8.3f} ms")
        print(f"MRC File Write (Disk) : {t_mrc_write * 1000:8.3f} ms")
        print(f"Particle Loop Total    : {t_loop_total * 1000:8.3f} ms")
        print(f"  └─ .tolist() portion : {t_tolist_total * 1000:8.3f} ms")
        print(f"Total Time             : {(t_array_conv + t_mrc_write + t_loop_total) * 1000:8.3f} ms\n")

        return outImgSet

    target_mod = sys.modules.get("scipion_bridge.backend.pyworkflow")
    assert target_mod is not None

    lift_resolvers(sys.modules[__name__], target=target_mod)


register_pyworkflow_resolvers()
