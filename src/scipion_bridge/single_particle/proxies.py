

from ..core.typed.proxy import Proxy, ProxyGroup


class StarfileProxy(Proxy):

    @classmethod
    def file_ext(cls):
        return ".star"


class MRCStackProxy(Proxy):

    @classmethod
    def file_ext(cls):
        return ".mrcs"


class ParticleStackProxy(ProxyGroup):
    _primary_field = "metadata"

    metadata: StarfileProxy
    particle_stack: MRCStackProxy