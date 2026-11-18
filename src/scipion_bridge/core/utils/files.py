"""Writing files that readers never see partially written."""

import os
import uuid
from pathlib import Path


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in its directory,
    renamed into place; creates the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
