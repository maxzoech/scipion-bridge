import os
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
import starfile

from ..core.typed.proxy import Proxy, ProxyGroup


class StarfileProxy(Proxy):

    @classmethod
    def file_ext(cls):
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


class ParticleStackProxy(ProxyGroup):
    metadata: StarfileProxy
    particle_stack: MRCStackProxy

    @property
    def primary_proxy(self) -> Proxy:
        return self.metadata
