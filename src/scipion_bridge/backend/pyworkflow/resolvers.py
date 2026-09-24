import os
import sys
import uuid
import numpy as np
import mrcfile

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple, Sequence, cast

try:
    import pwem.objects as emobj  # type: ignore
    from pwem.objects import (  # type: ignore
        Particle,
        SetOfParticles,
        SetOfParticlesFlex,
        ParticleFlex,
        SetOfClasses2D,
        Class2D,
        CTFModel,
        Coordinate,
    )
    from pwem.protocols import ProtFlexBase  # type: ignore
    from pwem.emlib.image import ImageHandler  # type: ignore

    HAS_PWEM = True
except ImportError:
    HAS_PWEM = False

    class _DynamicStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def __getattr__(self, item: str) -> Any:
            return None

        def __len__(self) -> int:
            return 0

    class Particle(_DynamicStub):
        pass  # type: ignore

    class SetOfParticles(_DynamicStub):
        pass  # type: ignore

    class SetOfParticlesFlex(_DynamicStub):
        pass  # type: ignore

    class ParticleFlex(_DynamicStub):
        pass  # type: ignore

    class SetOfClasses2D(_DynamicStub):
        pass  # type: ignore

    class Class2D(_DynamicStub):
        pass  # type: ignore

    class CTFModel(_DynamicStub):
        pass  # type: ignore

    class Coordinate(_DynamicStub):
        pass  # type: ignore

    class ProtFlexBase(_DynamicStub):
        pass  # type: ignore

    class ImageHandler(_DynamicStub):
        pass  # type: ignore


PROG_NAME = "scipion_bridge"


@dataclass
class PyWorkflowResolutionContext:

    protocol: ProtFlexBase  # type: ignore
    output_name: Optional[str] = None
    append: bool = False
    unprocessed_ids: Optional[Sequence[int]] = None


if HAS_PWEM:
    from ...core.typed.resolve import resolver, lift_resolvers
    from ...core import struct
    from ...core.struct.exceptions import UninitializedFieldError
    from ... import single_particle as spa

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _build_id_where_clause(ids: Sequence[int]) -> str:
        """Build a fast SQL WHERE clause (BETWEEN for contiguous ranges, IN for arbitrary IDs)."""
        lo, hi = min(ids), max(ids)
        if len(ids) == (hi - lo + 1):
            return f"id BETWEEN {lo} AND {hi}"
        return f"id IN ({','.join(map(str, ids))})"

    def _extract_particles_base(
        value: Any,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> Tuple[List[Any], np.ndarray, set]:
        """Helper extracting raw database rows, pre-loaded pixels array, and row column keys."""
        if hasattr(value, "loadAllProperties"):
            try:
                value.loadAllProperties()
            except Exception:
                pass

        ids = metadata.unprocessed_ids if metadata else None
        where_clause = _build_id_where_clause(ids) if ids else None

        db = value._getMapper().db
        raw_rows = db.selectAll(where=where_clause, iterate=False)

        num_particles = len(raw_rows)
        if num_particles == 0:
            return raw_rows, np.zeros((0, 0, 0), dtype=np.float32), set()

        row_keys = set(raw_rows[0].keys()) if hasattr(raw_rows[0], "keys") else set()
        filename_key = next(
            (k for k in (db._getRealCol("_filename"), "_filename") if k in row_keys),
            None,
        )
        idx_key = next(
            (k for k in (db._getRealCol("_index"), "_index") if k in row_keys), None
        )

        pixels: Optional[np.ndarray] = None
        ih = ImageHandler()

        for pos, row in enumerate(raw_rows):
            filename = row[filename_key] if filename_key else None
            idx = row[idx_key] if idx_key else 1
            if filename:
                data = cast(Any, ih.read((idx, filename))).getData().astype(np.float32)
                if pixels is None:
                    pixels = np.empty((num_particles, *data.shape), dtype=np.float32)
                pixels[pos] = data

        if pixels is None:
            pixels = np.zeros((num_particles, 0, 0), dtype=np.float32)

        return raw_rows, pixels, row_keys

    def _col(db: Any, attr: str) -> Optional[str]:
        """Return the real SQLite column name for a given attribute, or None if absent."""
        try:
            return db._getRealCol(attr)
        except Exception:
            return None

    def _fill_ctf_columns(
        particle_set: "struct.Set[spa.Particle]",
        raw_rows: List[Any],
        db: Any,
    ) -> None:
        """Populate CTF sub-columns on *particle_set* from *raw_rows* if the columns exist."""
        col_map = {
            "defocus_u": _col(db, "_ctfModel._defocusU"),
            "defocus_v": _col(db, "_ctfModel._defocusV"),
            "defocus_angle": _col(db, "_ctfModel._defocusAngle"),
            "phase_shift": _col(db, "_ctfModel._phaseShift"),
            "resolution": _col(db, "_ctfModel._resolution"),
            "fit_quality": _col(db, "_ctfModel._fitQuality"),
        }
        row_keys = set(raw_rows[0].keys()) if raw_rows else set()
        present = {
            field: col for field, col in col_map.items() if col and col in row_keys
        }
        if not present:
            return

        n = len(raw_rows)
        for field, col in present.items():
            raw_vals = [row[col] for row in raw_rows]
            if any(v is None for v in raw_vals):
                continue
            particle_set["ctf"][field] = np.array(
                raw_vals,
                dtype=np.float64,
            ).reshape(n, 1)

    def _fill_coordinate_columns(
        particle_set: "struct.Set[spa.Particle]",
        raw_rows: List[Any],
        db: Any,
    ) -> None:
        """Populate coordinate sub-columns on *particle_set* from *raw_rows* if the columns exist."""
        col_x = _col(db, "_coordinate._x")
        col_y = _col(db, "_coordinate._y")
        row_keys = set(raw_rows[0].keys()) if raw_rows else set()
        if not (col_x and col_x in row_keys and col_y and col_y in row_keys):
            return

        xs = [row[col_x] for row in raw_rows]
        ys = [row[col_y] for row in raw_rows]
        if any(x is None for x in xs) or any(y is None for y in ys):
            return

        n = len(raw_rows)
        particle_set["coordinate"]["x"] = np.array(
            xs,
            dtype=np.float64,
        ).reshape(n, 1)
        particle_set["coordinate"]["y"] = np.array(
            ys,
            dtype=np.float64,
        ).reshape(n, 1)

    def _fill_sampling_rate(
        particle_set: "struct.Set[spa.Particle]",
        raw_rows: List[Any],
        db: Any,
        container: Any = None,
    ) -> None:
        """Populate sampling_rate on *particle_set* from *raw_rows* with container fallback."""
        n = len(raw_rows)
        if n == 0:
            return

        col = _col(db, "_samplingRate")
        row_keys = set(raw_rows[0].keys()) if raw_rows else set()
        vals = None
        if col and col in row_keys:
            raw_vals = [row[col] for row in raw_rows]
            if not any(v is None for v in raw_vals):
                vals = [float(v) for v in raw_vals]

        if (
            vals is None
            and container is not None
            and hasattr(container, "getSamplingRate")
        ):
            sr = container.getSamplingRate()
            if sr is not None and float(sr) > 0:
                vals = [float(sr)] * n
        elif (
            vals is None
            and container is not None
            and hasattr(container, "getImages")
            and container.getImages() is not None
        ):
            sr = container.getImages().getSamplingRate()
            if sr is not None and float(sr) > 0:
                vals = [float(sr)] * n

        if vals is not None:
            particle_set["sampling_rate"] = np.array(
                vals,
                dtype=np.float64,
            ).reshape(n, 1)

    def _has_any_acquisition(obj: Any) -> bool:
        if hasattr(obj, "hasAcquisition") and obj.hasAcquisition():
            return True
        if hasattr(obj, "getAcquisition"):
            acq = obj.getAcquisition()
            if acq is not None and hasattr(acq, "equalAttributes"):
                return not acq.equalAttributes(emobj.Acquisition())
        return False

    def _fill_acquisition_columns(
        particle_set: "struct.Set[spa.Particle]",
        raw_rows: List[Any],
        db: Any,
        container: Any = None,
    ) -> None:
        """Populate acquisition sub-columns on *particle_set* from *raw_rows* with container fallback."""
        n = len(raw_rows)
        if n == 0:
            return

        field_to_prop = {
            "magnification": "_acquisition._magnification",
            "voltage": "_acquisition._voltage",
            "spherical_aberration": "_acquisition._sphericalAberration",
            "amplitude_contrast": "_acquisition._amplitudeContrast",
            "dose_initial": "_acquisition._doseInitial",
            "dose_per_frame": "_acquisition._dosePerFrame",
        }

        prop_to_getter = {
            "magnification": "getMagnification",
            "voltage": "getVoltage",
            "spherical_aberration": "getSphericalAberration",
            "amplitude_contrast": "getAmplitudeContrast",
            "dose_initial": "getDoseInitial",
            "dose_per_frame": "getDosePerFrame",
        }

        container_acq = None
        if container is not None and _has_any_acquisition(container):
            container_acq = container.getAcquisition()
        elif (
            container is not None
            and hasattr(container, "getImages")
            and container.getImages() is not None
            and _has_any_acquisition(container.getImages())
        ):
            container_acq = container.getImages().getAcquisition()

        row_keys = set(raw_rows[0].keys()) if raw_rows else set()

        for field, prop in field_to_prop.items():
            col = _col(db, prop)
            vals = None
            if col and col in row_keys:
                raw_vals = [row[col] for row in raw_rows]
                if not any(v is None for v in raw_vals):
                    vals = [float(v) for v in raw_vals]

            if vals is None and container_acq is not None:
                getter_name = prop_to_getter[field]
                getter = getattr(container_acq, getter_name, None)
                if getter is not None:
                    c_val = getter()
                    if c_val is not None:
                        vals = [float(c_val)] * n

            if vals is not None:
                particle_set["acquisition"][field] = np.array(
                    vals,
                    dtype=np.float64,
                ).reshape(n, 1)

    # ---------------------------------------------------------------------------
    # Single-particle resolvers
    # ---------------------------------------------------------------------------

    @resolver
    def resolve_scipion_particle_to_bridge_particle(
        value: Particle,
    ) -> spa.Particle:
        """Convert a Scipion/pwem Particle object to a scipion-bridge Particle struct."""
        if not value.getFileName():
            raise ValueError(
                "Could not convert Scipion 3 particle to Scipion Bridge particle",
            )

        ih = ImageHandler()
        img: Any = ih.read(value)
        pixel_data = img.getData().astype(np.float32)

        particle = spa.Particle(pixels=pixel_data)

        if value.getSamplingRate():
            particle.sampling_rate = float(value.getSamplingRate())

        if value.hasCTF():
            ctf_model = value.getCTF()
            particle.ctf.defocus_u = float(ctf_model.getDefocusU())  # type: ignore
            particle.ctf.defocus_v = float(ctf_model.getDefocusV())  # type: ignore
            particle.ctf.defocus_angle = float(ctf_model.getDefocusAngle())  # type: ignore
            particle.ctf.phase_shift = float(ctf_model.getPhaseShift() or 0.0)  # type: ignore
            particle.ctf.resolution = float(ctf_model.getResolution() or 0.0)  # type: ignore
            particle.ctf.fit_quality = float(ctf_model.getFitQuality() or 0.0)  # type: ignore

        if value.hasCoordinate():
            coord = value.getCoordinate()
            particle.coordinate.x = float(coord.getX())  # type: ignore
            particle.coordinate.y = float(coord.getY())  # type: ignore

        if value.hasAcquisition():
            acq = value.getAcquisition()
            if acq is not None:
                if acq.getMagnification() is not None:
                    particle.acquisition.magnification = float(acq.getMagnification())
                if acq.getVoltage() is not None:
                    particle.acquisition.voltage = float(acq.getVoltage())
                if acq.getSphericalAberration() is not None:
                    particle.acquisition.spherical_aberration = float(
                        acq.getSphericalAberration()
                    )
                if acq.getAmplitudeContrast() is not None:
                    particle.acquisition.amplitude_contrast = float(
                        acq.getAmplitudeContrast()
                    )
                if acq.getDoseInitial() is not None:
                    particle.acquisition.dose_initial = float(acq.getDoseInitial())
                if acq.getDosePerFrame() is not None:
                    particle.acquisition.dose_per_frame = float(acq.getDosePerFrame())

        return particle

    def _populate_acquisition_from_struct(
        target: Any,
        acq: spa.Acquisition,
    ) -> None:
        """Populate Scipion Acquisition on target (Particle, Class2D, or SetOfParticles) from Acquisition struct."""
        if not isinstance(acq, spa.Acquisition):
            return

        acq_obj = emobj.Acquisition()  # type: ignore
        has_acq = False

        if acq.is_initialized("magnification"):
            acq_obj.setMagnification(float(acq.magnification))
            has_acq = True
        if acq.is_initialized("voltage"):
            acq_obj.setVoltage(float(acq.voltage))
            has_acq = True
        if acq.is_initialized("spherical_aberration"):
            acq_obj.setSphericalAberration(float(acq.spherical_aberration))
            has_acq = True
        if acq.is_initialized("amplitude_contrast"):
            acq_obj.setAmplitudeContrast(float(acq.amplitude_contrast))
            has_acq = True
        if acq.is_initialized("dose_initial"):
            acq_obj.setDoseInitial(float(acq.dose_initial))
            has_acq = True
        if acq.is_initialized("dose_per_frame"):
            acq_obj.setDosePerFrame(float(acq.dose_per_frame))
            has_acq = True

        if has_acq and hasattr(target, "setAcquisition"):
            target.setAcquisition(acq_obj)

    def _populate_ctf_from_struct(particle: Any, ctf: spa.CTF) -> None:
        ctf_model = emobj.CTFModel()  # type: ignore
        has_ctf = False
        for attr, setter in (
            ("defocus_u", ctf_model.setDefocusU),
            ("defocus_v", ctf_model.setDefocusV),
            ("defocus_angle", ctf_model.setDefocusAngle),
            ("phase_shift", ctf_model.setPhaseShift),
            ("resolution", ctf_model.setResolution),
            ("fit_quality", ctf_model.setFitQuality),
        ):
            if ctf.is_initialized(attr):
                setter(float(getattr(ctf, attr)))
                has_ctf = True
        if has_ctf:
            particle.setCTF(ctf_model)

    def _populate_coord_from_struct(particle: Any, coord: spa.Coordinate) -> None:
        coord_model = emobj.Coordinate()  # type: ignore
        has_coord = False
        if coord.is_initialized("x"):
            coord_model.setX(int(coord.x))
            has_coord = True
        if coord.is_initialized("y"):
            coord_model.setY(int(coord.y))
            has_coord = True
        if has_coord:
            particle.setCoordinate(coord_model)

    @resolver
    def resolve_bridge_particle_to_scipion_particle(
        value: spa.Particle,
    ) -> Particle:
        """Convert a scipion-bridge Particle struct to a Scipion/pwem Particle object."""
        is_flex = isinstance(value, spa.FlexParticle) and value.is_initialized(
            "embeddings",
        )
        particle = ParticleFlex(progName=PROG_NAME) if is_flex else emobj.Particle()  # type: ignore
        if is_flex:
            particle.getFlexInfo().setProgName(PROG_NAME)
            embeddings = np.asarray(value.embeddings)
            particle.setZFlex(embeddings.tolist())

        if value.is_initialized("sampling_rate"):
            particle.setSamplingRate(float(value.sampling_rate))

        if value.is_initialized("ctf"):
            _populate_ctf_from_struct(particle, value.ctf)

        if value.is_initialized("coordinate"):
            _populate_coord_from_struct(particle, value.coordinate)

        if value.is_initialized("acquisition"):
            _populate_acquisition_from_struct(particle, value.acquisition)

        return particle

    def _is_column_initialized(s: "struct.Set", *path: str) -> bool:
        """Return True if the given nested key path has been written into *s*."""
        kp = s._storage.root
        for part in path:
            kp = kp.append(part)
        return kp in s._storage

    @resolver
    def resolve_set_of_particles_to_bridge_particles(
        value: SetOfParticles,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> "struct.Set[spa.Particle]":
        """Fast resolver converting Scipion SetOfParticles directly to scipion-bridge Set[Particle]."""
        raw_rows, pixels, _ = _extract_particles_base(value, metadata)
        n = len(raw_rows)
        particle_set = struct.Set[spa.Particle](capacity=n)

        if n == 0:
            return particle_set

        db = value._getMapper().db
        zflex_col = _col(db, "_zFlex")
        row_keys = set(raw_rows[0].keys()) if raw_rows else set()
        if zflex_col and zflex_col in row_keys:
            flex_set: struct.Set[spa.FlexParticle] = struct.Set[spa.FlexParticle](
                capacity=n,
            )
            flex_set["pixels"] = pixels
            _fill_sampling_rate(flex_set, raw_rows, db, container=value)
            _fill_ctf_columns(flex_set, raw_rows, db)
            _fill_coordinate_columns(flex_set, raw_rows, db)
            _fill_acquisition_columns(flex_set, raw_rows, db, container=value)
            embeddings = np.array(
                [np.fromstring(row[zflex_col], sep=",") for row in raw_rows],
                dtype=np.float32,
            )
            flex_set["embeddings"] = embeddings
            return flex_set  # type: ignore[return-value]

        particle_set = struct.Set[spa.Particle](capacity=n)
        particle_set["pixels"] = pixels

        db = value._getMapper().db
        _fill_sampling_rate(particle_set, raw_rows, db, container=value)
        _fill_ctf_columns(particle_set, raw_rows, db)
        _fill_coordinate_columns(particle_set, raw_rows, db)
        _fill_acquisition_columns(particle_set, raw_rows, db, container=value)

        return particle_set

    def _find_input_particles(
        metadata: Optional[PyWorkflowResolutionContext],
    ) -> Optional[Any]:
        if metadata is None or metadata.protocol is None:
            return None
        proto = metadata.protocol
        if hasattr(proto, "inputTypes"):
            for input_name in proto.inputTypes:
                source = getattr(proto, input_name, None)
                if (
                    source is not None
                    and hasattr(source, "hasValue")
                    and source.hasValue()
                ):
                    val = source.get()
                    if isinstance(
                        val, (emobj.SetOfParticles, emobj.SetOfParticlesFlex)
                    ):
                        return val
        return None

    @resolver
    def resolve_bridge_particles_to_set_of_particles(
        value: "struct.Set[spa.Particle]",
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> SetOfParticles:
        """Convert a scipion-bridge Set[Particle] to a Scipion SetOfParticles."""
        assert (
            metadata is not None
        ), "PyWorkflowResolutionContext is required to resolve Set[Particle] to SetOfParticles."

        output_name = metadata.output_name or uuid.uuid4().hex
        unique_id = uuid.uuid4().hex
        out_set: SetOfParticles = metadata.protocol._createSetOfParticles(
            suffix=f"_{output_name}_{unique_id}",
        )

        input_particles = _find_input_particles(metadata)
        if input_particles is not None:
            out_set.copyInfo(input_particles)

        if len(value) == 0:
            out_set.write()
            out_set._getMapper().commit()
            return out_set

        stack_path = metadata.protocol._getExtraPath(
            f"output_{output_name}_{unique_id}.mrcs",
        )
        Path(stack_path).parent.mkdir(parents=True, exist_ok=True)

        pixels_arr = np.array(value["pixels"], dtype=np.float32)
        mrcfile.write(stack_path, pixels_arr, overwrite=False)

        if pixels_arr.ndim >= 3:
            out_set.setDim((int(pixels_arr.shape[-1]), int(pixels_arr.shape[-2]), 1))

        # Determine which optional columns are initialized
        has_sr = _is_column_initialized(value, "sampling_rate")
        has_ctf = _is_column_initialized(value, "ctf", "defocus_u")
        has_coord = _is_column_initialized(value, "coordinate", "x")
        has_acq = _is_column_initialized(value, "acquisition", "voltage")

        out_set.enableAppend()  # type: ignore
        mapper = out_set._getMapper()  # type: ignore

        for i in range(len(value)):
            p = emobj.Particle()  # type: ignore
            p.setLocation(i + 1, stack_path)

            if has_sr:
                p.setSamplingRate(float(value[i].sampling_rate))
            elif input_particles is not None and input_particles.getSamplingRate():
                p.setSamplingRate(float(input_particles.getSamplingRate()))

            if has_ctf:
                ctf_model = emobj.CTFModel()  # type: ignore
                ctf_model.setDefocusU(float(value[i].ctf.defocus_u))
                ctf_model.setDefocusV(float(value[i].ctf.defocus_v))
                ctf_model.setDefocusAngle(float(value[i].ctf.defocus_angle))
                ctf_model.setPhaseShift(float(value[i].ctf.phase_shift))
                p.setCTF(ctf_model)

            if has_coord:
                coord = emobj.Coordinate()  # type: ignore
                coord.setX(int(value[i].coordinate.x))
                coord.setY(int(value[i].coordinate.y))
                p.setCoordinate(coord)

            if has_acq:
                _populate_acquisition_from_struct(p, value[i].acquisition)
            elif (
                input_particles is not None
                and hasattr(input_particles, "hasAcquisition")
                and input_particles.hasAcquisition()
            ):
                p.setAcquisition(input_particles.getAcquisition().clone())

            out_set.append(p)  # type: ignore

        if len(value) > 0:
            if has_sr:
                out_set.setSamplingRate(float(value[0].sampling_rate))
            if has_acq:
                _populate_acquisition_from_struct(out_set, value[0].acquisition)

        out_set.write()  # type: ignore
        mapper.commit()

        return out_set  # type: ignore

    @resolver
    def resolve_set_of_particles_flex_to_bridge_particles(
        value: SetOfParticlesFlex,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> "struct.Set[spa.FlexParticle]":
        """Fast resolver converting Scipion SetOfParticlesFlex directly to scipion-bridge Set[FlexParticle]."""
        raw_rows, pixels, _ = _extract_particles_base(value, metadata)
        num_particles = len(raw_rows)

        if num_particles == 0:
            return struct.Set[spa.FlexParticle](capacity=0)

        db = value._getMapper().db
        zflex_key = db._getRealCol("_zFlex")

        embeddings = np.array(
            [np.fromstring(row[zflex_key], sep=",") for row in raw_rows],
            dtype=np.float32,
        )

        particle_set = struct.Set[spa.FlexParticle](capacity=num_particles)
        particle_set["pixels"] = pixels
        particle_set["embeddings"] = embeddings

        _fill_sampling_rate(particle_set, raw_rows, db, container=value)
        _fill_acquisition_columns(particle_set, raw_rows, db, container=value)
        _fill_ctf_columns(particle_set, raw_rows, db)
        _fill_coordinate_columns(particle_set, raw_rows, db)

        return particle_set

    @resolver
    def resolve_embeddings_to_flex_particles(
        value: "struct.Set[spa.FlexParticle]",
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> SetOfParticlesFlex:
        assert (
            metadata is not None
        ), "PyWorkflowResolutionContext is required to resolve FlexParticle to SetOfParticlesFlex."

        output_name = metadata.output_name or uuid.uuid4().hex
        unique_id = uuid.uuid4().hex

        out_img_set: SetOfParticlesFlex = metadata.protocol._createSetOfParticlesFlex(
            suffix=f"_{output_name}_{unique_id}",
            progName=PROG_NAME,
        )
        out_img_set.getFlexInfo().setProgName(PROG_NAME)

        input_particles = _find_input_particles(metadata)
        if input_particles is not None:
            out_img_set.copyInfo(input_particles)

        if len(value) == 0:
            out_img_set.write()
            out_img_set._getMapper().commit()
            return out_img_set

        stack_path = metadata.protocol._getExtraPath(
            f"output_{output_name}_{unique_id}.mrcs",
        )
        Path(stack_path).parent.mkdir(parents=True, exist_ok=True)

        pixels_arr = np.array(value["pixels"], dtype=np.float32)
        mrcfile.write(
            stack_path,
            pixels_arr,
            overwrite=False,
        )

        if pixels_arr.ndim >= 3:
            out_img_set.setDim(
                (int(pixels_arr.shape[-1]), int(pixels_arr.shape[-2]), 1)
            )

        embeddings_list = np.array(value["embeddings"]).tolist()

        has_sr = _is_column_initialized(value, "sampling_rate")
        has_ctf = _is_column_initialized(value, "ctf", "defocus_u")
        has_coord = _is_column_initialized(value, "coordinate", "x")
        has_acq = _is_column_initialized(value, "acquisition", "voltage")

        out_img_set.enableAppend()
        mapper = out_img_set._getMapper()

        for i, z_flex_list in enumerate(embeddings_list, start=1):
            out_particle = ParticleFlex(progName=PROG_NAME)
            out_particle.getFlexInfo().setProgName(PROG_NAME)
            out_particle.setLocation(i, stack_path)
            out_particle.setZFlex(z_flex_list)

            if has_sr:
                out_particle.setSamplingRate(float(value[i - 1].sampling_rate))
            elif input_particles is not None and input_particles.getSamplingRate():
                out_particle.setSamplingRate(float(input_particles.getSamplingRate()))

            if has_ctf:
                _populate_ctf_from_struct(out_particle, value[i - 1].ctf)

            if has_coord:
                _populate_coord_from_struct(out_particle, value[i - 1].coordinate)

            if has_acq:
                _populate_acquisition_from_struct(
                    out_particle, value[i - 1].acquisition
                )
            elif (
                input_particles is not None
                and hasattr(input_particles, "hasAcquisition")
                and input_particles.hasAcquisition()
            ):
                out_particle.setAcquisition(input_particles.getAcquisition().clone())

            out_img_set.append(out_particle)

        if len(value) > 0:
            if has_sr:
                out_img_set.setSamplingRate(float(value[0].sampling_rate))
            if has_acq:
                _populate_acquisition_from_struct(out_img_set, value[0].acquisition)

        out_img_set.write()
        mapper.commit()

        return out_img_set

    # ---------------------------------------------------------------------------
    # Class2D resolvers
    # ---------------------------------------------------------------------------

    def _resolve_class2d_to_bridge(
        cls2d: Class2D,
        class_id: Optional[int] = None,
    ) -> spa.Class2D:
        """Convert a single Scipion Class2D to a bridge spa.Class2D struct."""
        particles_bridge = resolve_set_of_particles_to_bridge_particles(cls2d)
        cid = int(cls2d.getObjId() or 1)
        bid = class_id if class_id is not None else cid - 1
        bridge_cls = spa.Class2D(
            particles=particles_bridge,
            class_id=bid,
        )

        if cls2d.hasRepresentative():
            rep = cls2d.getRepresentative()
            if rep is not None and rep.getFileName():
                bridge_cls.representative = resolve_scipion_particle_to_bridge_particle(
                    rep,
                )

        return bridge_cls

    @resolver
    def resolve_set_of_classes2d_to_collection_classes2d(
        value: SetOfClasses2D,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> struct.Collection[spa.Class2D]:
        """Convert a Scipion SetOfClasses2D to a scipion-bridge Collection[Class2D]."""
        items: list[tuple[int, spa.Class2D]] = []
        for cls2d in value:  # type: ignore
            cid = int(cls2d.getObjId() or 1)
            idx = cid - 1
            bridge_cls = _resolve_class2d_to_bridge(cls2d, class_id=idx)
            items.append((idx, bridge_cls))

        if not items:
            return struct.Collection[spa.Class2D](size=1)

        max_idx = max(idx for idx, _ in items)
        coll = struct.Collection[spa.Class2D](size=max_idx + 1)
        for idx, bridge_cls in items:
            coll[idx] = bridge_cls

        return coll

    def _resolve_bridge_to_scipion_class2d(
        bridge_cls: spa.Class2D,
        out_classes: SetOfClasses2D,
        class_id: Optional[int] = None,
        metadata: Optional[PyWorkflowResolutionContext] = None,
        output_name: Optional[str] = None,
        unique_id: Optional[str] = None,
    ) -> Class2D:
        """Convert a single bridge spa.Class2D to a Scipion Class2D and append its particles."""
        scipion_cls = Class2D()
        match class_id:
            case int():
                cid = class_id
            case _ if bridge_cls.is_initialized("class_id"):
                cid = int(bridge_cls.class_id) + 1
            case _:
                cid = 1

        scipion_cls.setObjId(cid)
        scipion_cls.copyInfo(out_classes)

        if bridge_cls.is_initialized("representative"):
            rep_bridge = bridge_cls.representative
            rep = resolve_bridge_particle_to_scipion_particle(rep_bridge)
            rep.setClassId(cid)
            if (
                metadata is not None
                and metadata.protocol is not None
                and hasattr(metadata.protocol, "_getExtraPath")
                and rep_bridge.is_initialized("pixels")
            ):
                rep_path = os.path.join(
                    metadata.protocol._getExtraPath(),
                    f"output_{output_name}_{unique_id}_class_{cid}_rep.mrc",
                )
                Path(rep_path).parent.mkdir(parents=True, exist_ok=True)
                rep_pixels = np.array(rep_bridge.pixels, dtype=np.float32)
                mrcfile.write(rep_path, rep_pixels, overwrite=False)
                rep.setLocation(1, rep_path)
                if rep_pixels.ndim >= 2:
                    scipion_cls.setDim(
                        (int(rep_pixels.shape[-1]), int(rep_pixels.shape[-2]), 1),
                    )

            if rep_bridge.is_initialized("sampling_rate"):
                sr = float(rep_bridge.sampling_rate)
                rep.setSamplingRate(sr)
                scipion_cls.setSamplingRate(sr)

            if rep_bridge.is_initialized("acquisition"):
                _populate_acquisition_from_struct(rep, rep_bridge.acquisition)
                _populate_acquisition_from_struct(scipion_cls, rep_bridge.acquisition)

            scipion_cls.setRepresentative(rep)
        else:
            rep = emobj.Particle()
            rep.setClassId(cid)
            scipion_cls.setRepresentative(rep)

        out_classes.append(scipion_cls)

        if bridge_cls.is_initialized("particles"):
            particles_bridge: struct.Set[spa.Particle] = bridge_cls.particles
            n_particles = len(particles_bridge)
            if n_particles > 0:
                has_pixels = (
                    metadata is not None
                    and metadata.protocol is not None
                    and hasattr(metadata.protocol, "_getExtraPath")
                    and _is_column_initialized(particles_bridge, "pixels")
                )
                stack_path = None
                if has_pixels:
                    stack_path = os.path.join(
                        metadata.protocol._getExtraPath(),
                        f"output_{output_name}_{unique_id}_class_{cid}_particles.mrcs",
                    )
                    Path(stack_path).parent.mkdir(parents=True, exist_ok=True)
                    pixels_arr = np.array(particles_bridge["pixels"], dtype=np.float32)
                    mrcfile.write(stack_path, pixels_arr, overwrite=False)
                    if scipion_cls.getDim() is None and pixels_arr.ndim >= 3:
                        scipion_cls.setDim(
                            (int(pixels_arr.shape[-1]), int(pixels_arr.shape[-2]), 1),
                        )

                has_sr = _is_column_initialized(particles_bridge, "sampling_rate")
                has_ctf = _is_column_initialized(particles_bridge, "ctf", "defocus_u")
                has_coord = _is_column_initialized(particles_bridge, "coordinate", "x")
                has_acq = _is_column_initialized(
                    particles_bridge, "acquisition", "voltage"
                )

                if has_acq and not scipion_cls.hasAcquisition():
                    _populate_acquisition_from_struct(
                        scipion_cls, particles_bridge[0].acquisition
                    )

                for i in range(n_particles):
                    p = emobj.Particle()
                    p.setClassId(cid)

                    if stack_path is not None:
                        p.setLocation(i + 1, stack_path)

                    if has_sr:
                        p.setSamplingRate(float(particles_bridge[i].sampling_rate))

                    if has_ctf:
                        _populate_ctf_from_struct(p, particles_bridge[i].ctf)

                    if has_coord:
                        _populate_coord_from_struct(p, particles_bridge[i].coordinate)

                    if has_acq:
                        _populate_acquisition_from_struct(
                            p, particles_bridge[i].acquisition
                        )

                    scipion_cls.append(p)

                scipion_cls.write()
                scipion_cls._getMapper().commit()

        out_classes.update(scipion_cls)
        return scipion_cls

    @resolver
    def resolve_collection_classes2d_to_set_of_classes2d(
        value: struct.Collection[spa.Class2D],
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> SetOfClasses2D:
        """Convert a scipion-bridge Collection[Class2D] to a Scipion SetOfClasses2D."""
        assert (
            metadata is not None
        ), "PyWorkflowResolutionContext is required to resolve Collection[Class2D] to SetOfClasses2D."

        output_name = metadata.output_name or uuid.uuid4().hex
        unique_id = uuid.uuid4().hex

        out_classes: SetOfClasses2D = metadata.protocol._createSetOfClasses2D(
            suffix=f"_{output_name}_{unique_id}",
        )
        out_classes.enableAppend()

        input_particles = _find_input_particles(metadata)
        if input_particles is not None:
            out_classes.setImages(input_particles)

        first_cls_sr = None
        first_cls_acq = None
        for idx in value.initialized_indices():
            bridge_cls = value[idx]
            sc_cls = _resolve_bridge_to_scipion_class2d(
                bridge_cls,
                out_classes,
                class_id=idx + 1,
                metadata=metadata,
                output_name=output_name,
                unique_id=unique_id,
            )
            if first_cls_sr is None and sc_cls.getSamplingRate():
                first_cls_sr = sc_cls.getSamplingRate()

            if first_cls_acq is None:
                if bridge_cls.is_initialized(
                    "representative"
                ) and bridge_cls.representative.is_initialized("acquisition"):
                    first_cls_acq = bridge_cls.representative.acquisition
                elif (
                    bridge_cls.is_initialized("particles")
                    and len(bridge_cls.particles) > 0
                    and _is_column_initialized(
                        bridge_cls.particles, "acquisition", "voltage"
                    )
                ):
                    first_cls_acq = bridge_cls.particles[0].acquisition

        if out_classes.getImages() is None:
            img_set = emobj.SetOfParticles(filename=":memory:")
            if first_cls_sr is not None:
                img_set.setSamplingRate(first_cls_sr)
            if first_cls_acq is not None:
                _populate_acquisition_from_struct(img_set, first_cls_acq)
            out_classes.setImages(img_set)
        else:
            if first_cls_sr is not None and not out_classes.getSamplingRate():
                out_classes.getImages().setSamplingRate(first_cls_sr)
            if (
                first_cls_acq is not None
                and hasattr(out_classes.getImages(), "hasAcquisition")
                and not out_classes.getImages().hasAcquisition()
            ):
                _populate_acquisition_from_struct(
                    out_classes.getImages(), first_cls_acq
                )

        out_classes.write()
        out_classes._getMapper().commit()

        return out_classes

    target_mod = sys.modules.get("scipion_bridge.backend.pyworkflow")
    assert target_mod is not None

    lift_resolvers(sys.modules[__name__], target=target_mod)
