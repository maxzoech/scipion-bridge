def configure_default_env():
    from .container import Container

    container = Container()
    container.wire(
        modules=[__name__, "scipion_bridge"], packages=["scipion_bridge"]
    )
