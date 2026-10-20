import builtins
from functools import reduce
from pathlib import Path
from typing import Any, Mapping, Optional, Union, cast
import numpy as np
import pandas as pd
import starfile

from .mrc import write_stack
from .proxies import (
    ParticleStackProxy,
    MRCStackProxy,
    SpiderStackProxy,
    StarfileProxy,
    XmdProxy,
    XmippParticleStackProxy,
    XmippSpiderParticleStackProxy,
)
from .particle import Particle
from ..core.typed.resolve import resolver
from ..core import struct

# STAR columns and the Particle fields they fill. When several columns fill
# the same field, the last one present wins.
RELION_LABELS: Mapping[str, tuple[str, ...]] = {
    "rlnCoordinateX": ("coordinate", "x"),
    "rlnCoordinateY": ("coordinate", "y"),
    "rlnDefocusU": ("ctf", "defocus_u"),
    "rlnDefocusV": ("ctf", "defocus_v"),
    "rlnDefocusAngle": ("ctf", "defocus_angle"),
    "rlnPhaseShift": ("ctf", "phase_shift"),
    "rlnCtfMaxResolution": ("ctf", "resolution"),
    "rlnCtfResolution": ("ctf", "resolution"),
    "rlnCtfFigureOfMerit": ("ctf", "fit_quality"),
    "rlnCtfFitQuality": ("ctf", "fit_quality"),
    "rlnMagnification": ("acquisition", "magnification"),
    "rlnVoltage": ("acquisition", "voltage"),
    "rlnSphericalAberration": ("acquisition", "spherical_aberration"),
    "rlnAmplitudeContrast": ("acquisition", "amplitude_contrast"),
    "rlnDoseInitial": ("acquisition", "dose_initial"),
    "rlnDosePerFrame": ("acquisition", "dose_per_frame"),
}

# Xmipp labels as read by starfile, which strips the leading underscore.
XMIPP_LABELS: Mapping[str, tuple[str, ...]] = {
    "xcoor": ("coordinate", "x"),
    "ycoor": ("coordinate", "y"),
    "ctfDefocusU": ("ctf", "defocus_u"),
    "ctfDefocusV": ("ctf", "defocus_v"),
    "ctfDefocusAngle": ("ctf", "defocus_angle"),
    "ctfPhaseShift": ("ctf", "phase_shift"),
    "ctfCritMaxFreq": ("ctf", "resolution"),
    "ctfCritFitting": ("ctf", "fit_quality"),
    "magnification": ("acquisition", "magnification"),
    "ctfVoltage": ("acquisition", "voltage"),
    "ctfSphericalAberration": ("acquisition", "spherical_aberration"),
    "ctfQ0": ("acquisition", "amplitude_contrast"),
    "samplingRate": ("sampling_rate",),
    "ctfSamplingRate": ("sampling_rate",),
}


def _apply_labels(
    particle_set: struct.Set[Particle],
    df: pd.DataFrame,
    labels: Mapping[str, tuple[str, ...]],
) -> None:
    """Fill the fields of ``particle_set`` from the columns of ``df`` in ``labels``."""
    for column, (*parents, field) in labels.items():
        if column not in df.columns:
            continue

        target = reduce(lambda s, key: s[key], parents, particle_set)
        target[field] = np.asarray(df[column], dtype=np.float64)


@resolver
def resolve_mrc_stack_proxy(value: struct.Set[Particle]) -> MRCStackProxy:
    """
    Resolve an MRCStackProxy from a Particle Set.
    """
    new_proxy = MRCStackProxy.new_temporary_proxy()
    assert isinstance(new_proxy, MRCStackProxy)

    pixel_data = np.asarray(value["pixels"])
    write_stack(new_proxy.path, pixel_data, overwrite=True)

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

    write_stack(new_proxy.particle_stack.path, pixel_data, overwrite=True)

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
        data = value.particle_stack.read_images(slice)

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
                df_particles.iloc[slice] if slice is not None else df_particles
            ).iloc[:n]

            for col in ("rlnDetectorPixelSize", "rlnImagePixelSize", "rlnPixelSize"):
                if col in df_sliced.columns:
                    particle_set["sampling_rate"] = np.asarray(
                        df_sliced[col],
                        dtype=np.float64,
                    )
                    break

            _apply_labels(particle_set, df_sliced, RELION_LABELS)

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


@resolver
def resolve_xmd_to_xmipp_particle_stack(value: XmdProxy) -> XmippParticleStackProxy:
    """Resolve an XmippParticleStackProxy from its Xmipp metadata file."""
    return XmippParticleStackProxy(
        XmippParticleStackProxy.extract_base_path(value.path),
        managed=value.managed,
        metadata=value,
    )


def _xmipp_particles(
    metadata: XmdProxy,
    stack: Union[MRCStackProxy, SpiderStackProxy],
    slice: Optional[slice],
) -> struct.Set[Particle]:
    """The particles ``slice`` of an Xmipp stack, without the disabled ones.

    Fields without a label in the metadata, for example the sampling rate of
    stacks extracted in Scipion, are left unset.
    """
    df = metadata.read()
    assert isinstance(df, pd.DataFrame)

    rows = slice if slice is not None else builtins.slice(None)
    df_sliced = df.iloc[rows]

    expected = np.arange(1, len(df) + 1)[rows]
    indices = df_sliced["image"].str.split("@").str[0].astype(int).to_numpy()
    if not np.array_equal(indices, expected):
        raise ValueError(
            f"The images of '{metadata.path}' are not in stack order: "
            f"expected indices {expected.tolist()}, got {indices.tolist()}.",
        )

    enabled = (
        df_sliced["enabled"].to_numpy() == 1
        if "enabled" in df_sliced.columns
        else np.ones(len(df_sliced), dtype=bool)
    )
    data = stack.read_images(slice)[enabled]

    particle_set = struct.Set[Particle](capacity=len(data))
    particle_set["pixels"] = data
    _apply_labels(particle_set, df_sliced[enabled], XMIPP_LABELS)

    return particle_set


# Xmipp stacks are read only when requested as ``intermediate``: the weight
# keeps ``Path -> Set[Particle]`` on the RELION route by default.
@resolver(weight=1)
def resolve_xmipp_stack_to_particles(
    value: XmippParticleStackProxy,
    *,
    slice: Optional[slice] = None,
) -> struct.Set[Particle]:
    """Resolve a Set[Particle] from an Xmipp stack of MRC images."""
    return _xmipp_particles(value.metadata, value.particle_stack, slice)


@resolver
def resolve_xmd_to_xmipp_spider_particle_stack(
    value: XmdProxy,
) -> XmippSpiderParticleStackProxy:
    """Resolve an XmippSpiderParticleStackProxy from its Xmipp metadata file."""
    return XmippSpiderParticleStackProxy(
        XmippSpiderParticleStackProxy.extract_base_path(value.path),
        managed=value.managed,
        metadata=value,
    )


@resolver(weight=1)
def resolve_xmipp_spider_stack_to_particles(
    value: XmippSpiderParticleStackProxy,
    *,
    slice: Optional[slice] = None,
) -> struct.Set[Particle]:
    """Resolve a Set[Particle] from an Xmipp stack of SPIDER images."""
    return _xmipp_particles(value.metadata, value.particle_stack, slice)
