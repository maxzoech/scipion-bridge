"""Type resolvers converting PyWorkflow/pwem objects (Particle, etc.) to scipion-bridge types."""

import sys
import numpy as np

try:
    import pwem.objects as emobj  # type: ignore
    from pwem.emlib.image import ImageHandler  # type: ignore
    HAS_PWEM = True
except ImportError:
    HAS_PWEM = False


def register_pyworkflow_resolvers():
    """Register all PyWorkflow object resolvers in the type resolution graph."""
    if not HAS_PWEM:
        return

    from ...core.typed.resolve import resolver, lift_resolvers
    from ... import single_particle as spa

    @resolver
    def resolve_scipion_particle_to_bridge_particle(value: emobj.Particle) -> spa.Particle:
        """Convert a Scipion/pwem Particle object to a scipion-bridge Particle struct."""
        if value.getFileName():
            ih = ImageHandler()
            img = ih.read(value)
            pixel_data = img.getData().astype(np.float32)
            print(pixel_data.shape)
        else:
            raise ValueError("Could not convert Scipion 3 particle to Scipion Bridge particle")

        return spa.Particle(pixels=pixel_data)

    target_mod = sys.modules.get("scipion_bridge.backend.pyworkflow")
    assert target_mod is not None

    lift_resolvers(sys.modules[__name__], target=target_mod)


register_pyworkflow_resolvers()