

from .proxies import ParticleStackProxy
from .particle import Particle
from ..core.typed.resolve import resolver
from ..core import struct

@resolver
def resolve_particle_stack_proxy(value: struct.Set[Particle]) -> ParticleStackProxy:
    """
    Resolve a ParticleStackProxy from a Particle object.
    """
    import pandas as pd
    import mrcfile
    import starfile

    new_proxy = ParticleStackProxy.new_temporary_proxy()

    num_el = len(value)
    pixel_data = value["pixels"]

    mrcfile.write(new_proxy.particle_stack.path, pixel_data, overwrite=True)

    mrc_name = new_proxy.particle_stack.path.name
    df_data = {
        "rlnImageName": [f"{i + 1:06d}@{mrc_name}" for i in range(num_el)],
    }

    df = pd.DataFrame(df_data)
    starfile.write(df, new_proxy.metadata.path, overwrite=True)

    return new_proxy
