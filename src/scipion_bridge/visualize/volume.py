import os
import numpy as np
from . import base


class VolumeVisualizer(base.Visualizer):

    def show(self, path: os.PathLike):
        import k3d  # type: ignore
        import xmippLib # type: ignore

        volume = xmippLib.Image(path)
        plt_volume = k3d.volume(volume.getData().astype(np.float32))

        plot = k3d.plot()
        plot += plt_volume
        plot.display()

    @property
    def datatype(self):
        return "vol"
