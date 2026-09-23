import typing
from .base import Visualizer

visualizers: typing.Dict[str, Visualizer] = {}


def register(visualizer: Visualizer):
    types = {v.datatype for v in visualizers.values()}
    assert visualizer.datatype not in types

    visualizers[visualizer.datatype] = visualizer
