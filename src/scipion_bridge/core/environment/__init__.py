from .container import Container

def configure_default_env():

    container = Container()
    container.wire(
        modules=[__name__, "scipion_bridge"], packages=["scipion_bridge"]
    )


__all__ = [
    "Container",
    "configure_default_env"
]