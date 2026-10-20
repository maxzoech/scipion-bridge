from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import starfile
import mrcfile
import mrcfile.utils

from ..core.typed.proxy import Proxy, ProxyGroup
from .mrc import read_stack


def _contiguous_range(length: int, selection: Optional[slice]) -> range:
    """The indices of ``selection`` (all by default) among ``length`` items."""
    indices = range(length)[selection if selection is not None else slice(None)]
    assert indices.step == 1, f"Expected a contiguous selection, got {selection}."
    return indices


def _read_records(
    path: Path,
    offset: int,
    count: int,
    record_nbytes: int,
) -> np.ndarray:
    """``count`` records of ``record_nbytes`` bytes from ``offset``, read sequentially.

    Copying from a memory map faults the pages in one by one, which reads
    stacks on spinning disks at a fraction of their sequential speed.
    """
    records = np.empty(count * record_nbytes, dtype=np.uint8)
    buffer = records.data
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        # A single read returns at most ~2 GiB on Linux
        filled = 0
        while filled < len(buffer):
            read = f.readinto(buffer[filled:])
            if read == 0:
                raise EOFError(
                    f"'{path}' ends before record {count} at offset {offset}.",
                )
            filled += read

    return records.reshape(count, record_nbytes)


class StarfileProxy(Proxy):
    _cached_data: Optional[Any] = None

    def read(self) -> Any:
        if self._cached_data is None:
            self._cached_data = cast(Any, starfile.read(self.path))
        return self._cached_data

    def __len__(self) -> int:
        star_data = self.read()
        match star_data:
            case pd.DataFrame():
                return len(star_data)
            case dict():
                raw = star_data.get("particles")
                if isinstance(raw, pd.DataFrame):
                    return len(raw)
                for block in star_data.values():
                    if isinstance(block, pd.DataFrame):
                        return len(block)
                return 0
            case _:
                return 0

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".star"


class MRCStackProxy(Proxy):
    metadata_path: Optional[Path]

    def __init__(
        self,
        path: os.PathLike,
        managed: bool = False,
        metadata_path: Optional[os.PathLike] = None,
        *args,
        **kwargs,
    ) -> None:
        raw_path = Path(path)
        resolved_metadata: Optional[Path] = (
            Path(metadata_path) if metadata_path is not None else None
        )

        match raw_path.suffix:
            case ".star":
                actual_path = self.find_mrc_path_from_star(raw_path)
                resolved_metadata = raw_path
            case _:
                actual_path = raw_path

        super().__init__(actual_path, managed=managed, *args, **kwargs)
        self.metadata_path = resolved_metadata

    def __len__(self) -> int:
        with mrcfile.open(self.path, header_only=True) as mrc:
            assert mrc.header is not None
            nz = int(mrc.header.nz)
            return nz if nz > 0 else 1

    def read_images(self, selection: Optional[slice] = None) -> np.ndarray:
        """The images ``selection`` (all by default) as float32, read
        sequentially (``mrc.read_stack``, which rejects volumes)."""
        return read_stack(self.path, selection)

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        if not self.path.exists():
            return None
        with mrcfile.open(self.path, header_only=True) as mrc:
            assert mrc.header is not None
            dtype = mrcfile.utils.data_dtype_from_header(mrc.header)
            nx = int(mrc.header.nx)
            ny = int(mrc.header.ny)
            return nx * ny * dtype.itemsize

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".mrcs"

    @classmethod
    def extensions(cls) -> Optional[tuple[str, ...]]:
        return (".mrcs", ".mrc", ".star")

    @classmethod
    def find_mrc_path_from_star(cls, star_path: Path) -> Path:
        """Find the MRC data file referenced by a .star file."""
        star_path = Path(star_path)
        if not star_path.exists():
            raise FileNotFoundError(f"STAR file not found: {star_path}")

        try:
            star_data = cast(Any, starfile.read(star_path))
        except Exception as e:
            raise ValueError(f"Could not read STAR file '{star_path}': {e}") from e

        df: Optional[pd.DataFrame] = None
        match star_data:
            case pd.DataFrame():
                df = star_data
            case dict():
                raw = star_data.get("particles")
                if isinstance(raw, pd.DataFrame):
                    df = raw
                else:
                    for block in star_data.values():
                        if (
                            isinstance(block, pd.DataFrame)
                            and "rlnImageName" in block.columns
                        ):
                            df = block
                            break
            case _:
                df = None

        if df is None or "rlnImageName" not in df.columns or len(df) == 0:
            raise ValueError(
                f"Could not find 'rlnImageName' column in STAR file '{star_path}'.",
            )

        first_entry = str(df["rlnImageName"].iloc[0]).strip()
        # In RELION, format is '000001@path/to/stack.mrcs' or 'path/to/stack.mrcs'
        rel_mrc_str = first_entry.split("@")[-1]
        rel_mrc_path = Path(rel_mrc_str)

        if rel_mrc_path.is_absolute() and rel_mrc_path.exists():
            return rel_mrc_path

        # Check relative to the .star file directory
        candidate = star_path.parent / rel_mrc_path
        if candidate.exists():
            return candidate

        # Check relative to current working directory
        if rel_mrc_path.exists():
            return rel_mrc_path.resolve()

        raise FileNotFoundError(
            f"Referenced MRC file '{rel_mrc_path}' from STAR file '{star_path}' not found at '{candidate}'.",
        )


@dataclass(frozen=True)
class _SpiderLayout:
    """Layout of a SPIDER stack: a header, then a header and the pixels per image."""

    header_nbytes: int
    nrow: int
    nsam: int
    count: int

    @property
    def image_nbytes(self) -> int:
        return self.nrow * self.nsam * 4

    @property
    def record_nbytes(self) -> int:
        return self.header_nbytes + self.image_nbytes


class SpiderStackProxy(Proxy):
    """A SPIDER stack of 2D images (``.stk``), as written by Xmipp.

    Only little-endian float32 stacks are supported.
    """

    def _layout(self) -> _SpiderLayout:
        # Header words, 1-based: 2 nrow, 5 iform, 12 nsam, 22 labbyt,
        # 24 istack, 26 maxim
        header = np.fromfile(self.path, dtype="<f4", count=26)
        nrow, iform, nsam, labbyt, istack, maxim = (
            int(header[word - 1]) for word in (2, 5, 12, 22, 24, 26)
        )
        layout = _SpiderLayout(
            header_nbytes=labbyt,
            nrow=nrow,
            nsam=nsam,
            count=maxim,
        )

        size = self.path.stat().st_size
        expected = labbyt + maxim * layout.record_nbytes
        if istack <= 0 or iform != 1 or size != expected:
            raise ValueError(
                f"'{self.path}' is not a little-endian SPIDER stack of 2D images: "
                f"iform={iform}, istack={istack}, {size} bytes for {maxim} images "
                f"of {nrow}x{nsam} (expected {expected} bytes).",
            )

        return layout

    def __len__(self) -> int:
        return self._layout().count

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        return self._layout().image_nbytes

    def read_images(self, selection: Optional[slice] = None) -> np.ndarray:
        """The images ``selection`` (all by default) as float32, read sequentially."""
        layout = self._layout()
        indices = _contiguous_range(layout.count, selection)
        records = _read_records(
            self.path,
            layout.header_nbytes + indices.start * layout.record_nbytes,
            len(indices),
            layout.record_nbytes,
        )
        pixels = records.view("<f4")[:, layout.header_nbytes // 4 :]
        images = pixels.reshape(len(indices), layout.nrow, layout.nsam)
        return np.ascontiguousarray(images, dtype=np.float32)

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".stk"


class ParticleStackProxy(ProxyGroup):
    metadata: StarfileProxy
    particle_stack: MRCStackProxy

    @property
    def primary_proxy(self) -> Proxy:
        return self.metadata

    @property
    def num_particles(self) -> int:
        return len(self)

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        return self.particle_stack.estimated_item_nbytes


class XmdProxy(StarfileProxy):
    """Xmipp metadata: a STAR file with Xmipp labels (``_ctfDefocusU``, ...)."""

    @classmethod
    def file_ext(cls) -> Optional[str]:
        return ".xmd"


class XmippParticleStackProxy(ProxyGroup):
    """Particles extracted by Xmipp: ``<stem>.xmd`` next to ``<stem>.mrcs``.

    The ``_image`` column of the metadata names the stack relative to the
    Scipion project, so the stack is found by its stem instead.
    """

    metadata: XmdProxy
    particle_stack: MRCStackProxy

    @property
    def primary_proxy(self) -> Proxy:
        return self.metadata

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        return self.particle_stack.estimated_item_nbytes


class XmippSpiderParticleStackProxy(ProxyGroup):
    """Particles extracted by Xmipp: ``<stem>.xmd`` next to the SPIDER ``<stem>.stk``."""

    metadata: XmdProxy
    particle_stack: SpiderStackProxy

    @property
    def primary_proxy(self) -> Proxy:
        return self.metadata

    @property
    def estimated_item_nbytes(self) -> Optional[int]:
        return self.particle_stack.estimated_item_nbytes
