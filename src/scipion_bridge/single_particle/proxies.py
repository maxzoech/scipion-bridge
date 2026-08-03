

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
    metadata: StarfileProxy
    particle_stack: MRCStackProxy

    @property
    def primary_proxy(self) -> Proxy:
        return self.metadata