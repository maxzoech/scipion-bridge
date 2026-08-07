"""Type resolvers converting PyWorkflow/pwem objects (Particle, etc.) to scipion-bridge types."""

import os
import sys
import uuid
from typing import Dict
import numpy as np
import mrcfile
import time

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Any, Dict, List, Tuple, Sequence

try:
    import pwem.objects as emobj  # type: ignore
    from pwem.protocols import ProtFlexBase  # type: ignore
    from pwem.emlib.image import ImageHandler  # type: ignore

    HAS_PWEM = True
except ImportError:
    HAS_PWEM = False
    ProtFlexBase = Any  # type: ignore

PROG_NAME = "scipion_bridge"


@dataclass
class PyWorkflowResolutionContext:

    protocol: ProtFlexBase # type: ignore
    output_name: Optional[str]
    append: bool
    unprocessed_ids: Optional[Sequence[int]] = None


def register_pyworkflow_resolvers():
    """Register all PyWorkflow object resolvers in the type resolution graph."""
    if not HAS_PWEM:
        return

    from ...core.typed.resolve import resolver, lift_resolvers
    from ...core import struct
    from ... import single_particle as spa

    @resolver
    def resolve_scipion_particle_to_bridge_particle(
        value: emobj.Particle,
    ) -> spa.Particle:
        """Convert a Scipion/pwem Particle object to a scipion-bridge Particle struct."""
        if value.getFileName():
            ih = ImageHandler()
            img = ih.read(value)
            pixel_data = img.getData().astype(np.float32)
        else:
            raise ValueError(
                "Could not convert Scipion 3 particle to Scipion Bridge particle"
            )

        return spa.Particle(pixels=pixel_data)

    def _build_id_where_clause(ids: Sequence[int]) -> str:
        """Build a fast SQL WHERE clause (BETWEEN for contiguous ranges, IN for arbitrary IDs)."""
        lo, hi = min(ids), max(ids)
        if len(ids) == (hi - lo + 1):
            return f"id BETWEEN {lo} AND {hi}"
        return f"id IN ({','.join(map(str, ids))})"

    def _extract_particles_base(
        value: emobj.SetOfParticles,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> Tuple[List[Any], np.ndarray, set]:
        """Helper extracting raw database rows, pre-loaded pixels array, and row column keys."""
        ids = metadata.unprocessed_ids if metadata else None
        where_clause = _build_id_where_clause(ids) if ids else None

        db = value._getMapper().db
        raw_rows = db.selectAll(where=where_clause, iterate=False)

        num_particles = len(raw_rows)
        if num_particles == 0:
            return raw_rows, np.zeros((0, 128, 128), dtype=np.float32), set()

        row_keys = set(raw_rows[0].keys()) if hasattr(raw_rows[0], "keys") else set()
        filename_key = next((k for k in (db._getRealCol("_filename"), "_filename") if k in row_keys), None)
        idx_key = next((k for k in (db._getRealCol("_index"), "_index") if k in row_keys), None)

        pixels = np.empty((num_particles, 128, 128), dtype=np.float32)
        ih = ImageHandler()

        for pos, row in enumerate(raw_rows):
            filename = row[filename_key] if filename_key else None
            idx = row[idx_key] if idx_key else 1
            if filename:
                pixels[pos] = ih.read((idx, filename)).getData()

        return raw_rows, pixels, row_keys

    @resolver
    def resolve_set_of_particles_to_bridge_particles(
        value: emobj.SetOfParticles,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> struct.Set[spa.Particle]:
        """Fast resolver converting Scipion SetOfParticles directly to scipion-bridge Set[Particle]."""
        raw_rows, pixels, _ = _extract_particles_base(value, metadata)
        particle_set = struct.Set[spa.Particle](capacity=len(raw_rows))
        if len(raw_rows) > 0:
            particle_set["pixels"] = pixels
        return particle_set

    @resolver
    def resolve_set_of_particles_flex_to_bridge_particles(
        value: emobj.SetOfParticlesFlex,
        metadata: Optional[PyWorkflowResolutionContext] = None,
    ) -> struct.Set[spa.FlexParticle]:
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

        return particle_set

    @resolver
    def resolve_embeddings_to_flex_particles(
        value: struct.Set[spa.FlexParticle],
        metadata: PyWorkflowResolutionContext,
    ) -> emobj.SetOfParticlesFlex:
        if metadata is None:
            raise ValueError(
                "The Scipion Protocol is required as context to resolve Embeddings to SetOfParticlesFlex."
            )

        def _get_exisiting_set() -> Optional[emobj.SetOfParticlesFlex]:
            if metadata.output_name is not None:
                return getattr(metadata.protocol, metadata.output_name, None)
            else:
                return None

        # Create the Scipion 3 SetOfParticlesFlex
        outImgSet = _get_exisiting_set()
        output_name = metadata.output_name or uuid.uuid4().hex

        if outImgSet is not None and metadata.append == True:
            start_index = len(outImgSet)
            value = value[start_index:]
        else:
            outImgSet = metadata.protocol._createSetOfParticlesFlex(
                suffix=f"_{output_name}", progName=PROG_NAME
            )
            outImgSet.getFlexInfo().setProgName(PROG_NAME)
            start_index = 0

        stack_uuid = uuid.uuid4().hex
        stack_path = metadata.protocol._getExtraPath(
            f"output_{output_name}_{stack_uuid}.mrcs"
        )

        pixels_arr = np.array(value["pixels"], dtype=np.float32)

        mrcfile.write(
            stack_path,
            pixels_arr,
            overwrite=False,
        )

        embeddings_list = np.array(value["embeddings"]).tolist()

        outImgSet.enableAppend()
        mapper = outImgSet._getMapper()

        for i, z_flex_list in enumerate(embeddings_list, start=1):
            outParticle = emobj.ParticleFlex(progName=PROG_NAME)
            outParticle.getFlexInfo().setProgName(PROG_NAME)
            outParticle.setLocation(i, stack_path)
            outParticle.setZFlex(z_flex_list)
            outImgSet.append(outParticle)

        outImgSet.write()
        mapper.commit()

        return outImgSet

    target_mod = sys.modules.get("scipion_bridge.backend.pyworkflow")
    assert target_mod is not None

    lift_resolvers(sys.modules[__name__], target=target_mod)


register_pyworkflow_resolvers()
