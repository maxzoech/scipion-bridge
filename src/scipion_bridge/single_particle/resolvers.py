from pathlib import Path
from typing import Any, Optional, cast
import mrcfile
import numpy as np
import pandas as pd
import starfile

from .proxies import ParticleStackProxy, MRCStackProxy, StarfileProxy
from .particle import Particle
from ..core.typed.resolve import resolver
from ..core import struct


@resolver
def resolve_mrc_stack_proxy(value: struct.Set[Particle]) -> MRCStackProxy:
    """
    Resolve an MRCStackProxy from a Particle Set.
    """
    new_proxy = MRCStackProxy.new_temporary_proxy()
    assert isinstance(new_proxy, MRCStackProxy)

    pixel_data = np.asarray(value["pixels"])
    mrcfile.write(new_proxy.path, pixel_data, overwrite=True)

    return new_proxy


@resolver
def resolve_starfile_to_mrc_stack(value: StarfileProxy) -> MRCStackProxy:
    """
    Resolve an MRCStackProxy from a StarfileProxy.
    """
    return MRCStackProxy(value.path, managed=value.managed)


@resolver
def resolve_starfile_to_particle_stack(value: StarfileProxy) -> ParticleStackProxy:
    """
    Resolve a ParticleStackProxy from a StarfileProxy.
    """
    mrc_proxy = resolve_starfile_to_mrc_stack(value)
    base_path = ParticleStackProxy.extract_base_path(value.path)
    while base_path.suffix:
        base_path = base_path.with_suffix("")
    return ParticleStackProxy(
        base_path,
        managed=value.managed,
        metadata=value,
        particle_stack=mrc_proxy,
    )


@resolver
def resolve_particle_stack_proxy(value: struct.Set[Particle]) -> ParticleStackProxy:
    """
    Resolve a ParticleStackProxy from a Particle object.
    """
    new_proxy = ParticleStackProxy.new_temporary_proxy()
    assert isinstance(new_proxy, ParticleStackProxy)

    num_el = len(value)
    pixel_data = np.asarray(value["pixels"])

    mrcfile.write(new_proxy.particle_stack.path, pixel_data, overwrite=True)

    mrc_name = new_proxy.particle_stack.path.name
    df_data: dict[str, Any] = {
        "rlnImageName": [f"{i + 1:06d}@{mrc_name}" for i in range(num_el)],
    }

    if value._storage.root.append("sampling_rate") in value._storage:
        df_data["rlnDetectorPixelSize"] = np.asarray(value["sampling_rate"]).flatten()

    coord_root = value._storage.root.append("coordinate")
    if coord_root.append("x") in value._storage:
        df_data["rlnCoordinateX"] = np.asarray(value["coordinate"]["x"]).flatten()
    if coord_root.append("y") in value._storage:
        df_data["rlnCoordinateY"] = np.asarray(value["coordinate"]["y"]).flatten()

    ctf_root = value._storage.root.append("ctf")
    ctf_export_map = {
        "defocus_u": "rlnDefocusU",
        "defocus_v": "rlnDefocusV",
        "defocus_angle": "rlnDefocusAngle",
        "phase_shift": "rlnPhaseShift",
        "resolution": "rlnCtfMaxResolution",
        "fit_quality": "rlnCtfFigureOfMerit",
    }
    for field_name, star_col in ctf_export_map.items():
        if ctf_root.append(field_name) in value._storage:
            df_data[star_col] = np.asarray(value["ctf"][field_name]).flatten()

    acq_root = value._storage.root.append("acquisition")
    acq_export_map = {
        "magnification": "rlnMagnification",
        "voltage": "rlnVoltage",
        "spherical_aberration": "rlnSphericalAberration",
        "amplitude_contrast": "rlnAmplitudeContrast",
        "dose_initial": "rlnDoseInitial",
        "dose_per_frame": "rlnDosePerFrame",
    }
    for field_name, star_col in acq_export_map.items():
        if acq_root.append(field_name) in value._storage:
            df_data[star_col] = np.asarray(value["acquisition"][field_name]).flatten()

    df = pd.DataFrame(df_data)
    starfile.write(df, new_proxy.metadata.path, overwrite=True)

    return new_proxy


@resolver
class resolve_particle_stack_to_particles:
    """Resolve a Set[Particle] from a ParticleStackProxy.

    When used in iterating resolution, the instance is created once before the
    chunk loop, so the parsed STAR metadata is cached and reused across chunks.
    """

    def __init__(self) -> None:
        self._star_cache: dict[Path, Any] = {}

    def _get_star_data(self, path: Path) -> Any:
        if path not in self._star_cache:
            self._star_cache[path] = cast(Any, starfile.read(path))
        return self._star_cache[path]

    @staticmethod
    def _parse_star_blocks(
        star_data: Any,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        df_particles: Optional[pd.DataFrame] = None
        df_optics: Optional[pd.DataFrame] = None

        match star_data:
            case pd.DataFrame():
                df_particles = star_data
            case dict():
                raw_particles = star_data.get("particles")
                if isinstance(raw_particles, pd.DataFrame):
                    df_particles = raw_particles
                else:
                    for block in star_data.values():
                        if isinstance(block, pd.DataFrame):
                            df_particles = block
                            break

                raw_optics = star_data.get("optics")
                if isinstance(raw_optics, pd.DataFrame):
                    df_optics = raw_optics
            case _:
                df_particles = None

        return df_particles, df_optics

    def forward(
        self,
        value: ParticleStackProxy,
        *,
        slice: Optional[slice] = None,
    ) -> struct.Set[Particle]:
        with mrcfile.mmap(value.particle_stack.path, mode="r") as mrc:
            assert mrc.data is not None
            raw = mrc.data
            if raw.ndim == 2:
                raw = raw[np.newaxis, ...]
            elif raw.ndim != 3:
                raise ValueError(
                    f"Expected 2D or 3D image data in MRC stack '{value.particle_stack.path}', got shape {raw.shape}",
                )

            sliced_raw = raw[slice] if slice is not None else raw
            data = np.asarray(sliced_raw, dtype=np.float32)

        particle_set = struct.Set[Particle](capacity=len(data))
        particle_set["pixels"] = data

        if not value.metadata.path.exists():
            return particle_set

        star_data = (
            value.metadata.read()
            if hasattr(value.metadata, "read")
            else self._get_star_data(value.metadata.path)
        )
        df_particles, df_optics = self._parse_star_blocks(star_data)

        n = len(data)

        if df_particles is not None:
            df_sliced = (
                df_particles.iloc[slice] if slice is not None else df_particles.iloc[:n]
            )

            for col in ("rlnDetectorPixelSize", "rlnImagePixelSize", "rlnPixelSize"):
                if col in df_sliced.columns:
                    particle_set["sampling_rate"] = np.asarray(
                        df_sliced[col][:n],
                        dtype=np.float64,
                    )
                    break

            if "rlnCoordinateX" in df_sliced.columns:
                particle_set["coordinate"]["x"] = np.asarray(
                    df_sliced["rlnCoordinateX"][:n],
                    dtype=np.float64,
                )
            if "rlnCoordinateY" in df_sliced.columns:
                particle_set["coordinate"]["y"] = np.asarray(
                    df_sliced["rlnCoordinateY"][:n],
                    dtype=np.float64,
                )

            ctf_mappings = {
                "rlnDefocusU": "defocus_u",
                "rlnDefocusV": "defocus_v",
                "rlnDefocusAngle": "defocus_angle",
                "rlnPhaseShift": "phase_shift",
                "rlnCtfMaxResolution": "resolution",
                "rlnCtfResolution": "resolution",
                "rlnCtfFigureOfMerit": "fit_quality",
                "rlnCtfFitQuality": "fit_quality",
            }
            for star_col, ctf_field in ctf_mappings.items():
                if star_col in df_sliced.columns:
                    particle_set["ctf"][ctf_field] = np.asarray(
                        df_sliced[star_col][:n],
                        dtype=np.float64,
                    )

            acq_mappings = {
                "rlnMagnification": "magnification",
                "rlnVoltage": "voltage",
                "rlnSphericalAberration": "spherical_aberration",
                "rlnAmplitudeContrast": "amplitude_contrast",
                "rlnDoseInitial": "dose_initial",
                "rlnDosePerFrame": "dose_per_frame",
            }
            for star_col, acq_field in acq_mappings.items():
                if star_col in df_sliced.columns:
                    particle_set["acquisition"][acq_field] = np.asarray(
                        df_sliced[star_col][:n],
                        dtype=np.float64,
                    )

        if df_optics is not None and len(df_optics) > 0:
            optics_row = df_optics.iloc[0]
            if (
                particle_set._storage.root.append("sampling_rate")
                not in particle_set._storage
            ):
                for col in ("rlnImagePixelSize", "rlnDetectorPixelSize"):
                    if col in optics_row:
                        particle_set["sampling_rate"] = np.full(
                            n,
                            float(optics_row[col]),
                            dtype=np.float64,
                        )
                        break

            acq_root = particle_set._storage.root.append("acquisition")
            optics_acq_map = {
                "rlnVoltage": "voltage",
                "rlnSphericalAberration": "spherical_aberration",
                "rlnAmplitudeContrast": "amplitude_contrast",
                "rlnMagnification": "magnification",
            }
            for star_col, acq_field in optics_acq_map.items():
                if (
                    star_col in optics_row
                    and acq_root.append(acq_field) not in particle_set._storage
                ):
                    particle_set["acquisition"][acq_field] = np.full(
                        n,
                        float(optics_row[star_col]),
                        dtype=np.float64,
                    )

        return particle_set
