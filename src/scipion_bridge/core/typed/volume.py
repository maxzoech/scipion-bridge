import numpy as np

from .proxy import Proxy
from .array import ArrayConvertable

from xmipp_metadata.image_handler import ImageSpider as _BackendSpiderImage # type: ignore


class VolumeVisualizeable:

    def get_volume_data(self) -> np.ndarray:
        raise NotImplementedError

    def _ipython_display_(self):
        import k3d  # type: ignore
        import xmippLib  # type: ignore

        volume = self.get_volume_data()
        plt_volume = k3d.volume(volume.astype(np.float32))

        plot = k3d.plot()
        plot += plt_volume
        plot.display()


class SpiderFile(Proxy, ArrayConvertable, VolumeVisualizeable):

    @classmethod
    def file_ext(cls):
        return ".vol"

    @classmethod
    def from_numpy(cls, data: np.ndarray):
        import xmippLib  # type: ignore

        new_proxy = cls.new_temporary_proxy()

        volume = _BackendSpiderImage()
        volume.write(data, filename=str(new_proxy.path))

        return new_proxy

    def to_numpy(self):
        import xmippLib  # type: ignore

        volume = xmippLib.Image(str(self.path))
        return np.array(volume.getData().astype(np.float32), copy=True)

    def get_volume_data(self) -> np.ndarray:
        return self.to_numpy()


if __name__ == "__main__":
    from pathlib import Path
    from .resolve import current_registry

    SpiderFile(Path("/path/to/file"))
    current_registry()._plot_graph()
