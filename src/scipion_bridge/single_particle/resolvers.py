

from .proxies import ParticleStackProxy
from .particle import Particle
from ..core.typed.resolve import resolver
from ..core import struct

@resolver
def resolve_particle_stack_proxy(value: struct.Set[Particle]) -> ParticleStackProxy:
    """
    Resolve a ParticleStackProxy from a Particle object.
    """

    new_proxy = ParticleStackProxy.new_temporary_proxy()

    raise NotImplementedError("This function is not yet implemented. It should resolve a ParticleStackProxy from a Particle object.")