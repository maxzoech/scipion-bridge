"""Previews of items kept at checkpoints: their structure and thumbnails.

A preview reads only what it shows: the scalar fields and the 2D image
fields of Structs (e.g. the representative of a class), never the columns
of Sets, whose rows may reference input files (e.g. the particles of a
class).
"""

from __future__ import annotations

import base64
import struct
import zlib
from collections.abc import Mapping
from typing import Any, Dict

import numpy as np

from ...struct import Collection, Set, Struct
from ...struct.schema import ArrayEntryBase, SetEntryBase

# Thumbnails a preview shows at most, and their size in pixels at most.
MAX_THUMBNAILS = 64
THUMBNAIL_SIZE = 128
# Entries of mappings, sequences and Collections a preview lists at most.
MAX_ENTRIES = 64


def preview_of(item: Any) -> Dict[str, Any]:
    """The structure of ``item`` as JSON values, with PNG thumbnails (data
    URIs) of up to ``MAX_THUMBNAILS`` 2D image fields."""
    return _Previewer().preview(item)


class _Previewer:
    def __init__(self) -> None:
        self.thumbnails = 0

    def preview(self, obj: Any) -> Dict[str, Any]:
        match obj:
            case Collection():
                return {
                    "kind": f"Collection[{obj.item_type().__name__}]",
                    "length": len(obj),
                    "entries": {
                        str(key): self.preview(value)
                        for key, value in list(obj.items())[:MAX_ENTRIES]
                    },
                }
            case Set():
                assert obj.dtype is not None
                return {
                    "kind": f"Set[{obj.dtype.__name__}]",
                    "length": len(obj),
                    "fields": [
                        _field(path, entry, (len(obj), *entry.shape))
                        for path, entry in obj.schema().tree_iter()
                    ],
                }
            case Struct():
                return {
                    "kind": type(obj).__name__,
                    "fields": [
                        self._struct_field(obj, path, entry)
                        for path, entry in obj.schema().tree_iter()
                    ],
                }
            case Mapping():
                return {
                    "kind": "dict",
                    "length": len(obj),
                    "entries": {
                        str(key): self.preview(value)
                        for key, value in list(obj.items())[:MAX_ENTRIES]
                    },
                }
            case list() | tuple():
                return {
                    "kind": type(obj).__name__,
                    "length": len(obj),
                    "entries": {
                        str(i): self.preview(value)
                        for i, value in enumerate(obj[:MAX_ENTRIES])
                    },
                }
            case np.ndarray():
                return {"kind": "array", "dtype": str(obj.dtype), "shape": obj.shape}
            case _:
                return {"kind": type(obj).__name__, "value": repr(obj)[:200]}

    def _struct_field(self, obj: Struct, path: Any, entry: Any) -> Dict[str, Any]:
        """A field of a Struct, with its value if small or its thumbnail if
        it is an image."""
        assert isinstance(entry, ArrayEntryBase)
        if isinstance(entry, SetEntryBase):
            return _field(path, entry, (None, *entry.shape))
        key = obj.storage.root.extend(path)
        if not obj.storage.is_initialized(key):
            return {**_field(path, entry, entry.shape), "unset": True}
        value = np.asarray(obj.storage.read(key, entry))
        field = _field(path, entry, value.shape)
        if value.size <= 8:
            field["value"] = value.tolist()
        elif _is_image(value) and self.thumbnails < MAX_THUMBNAILS:
            self.thumbnails += 1
            field["thumbnail"] = png_data_uri(value)
        return field


def _field(path: Any, entry: Any, shape: Any) -> Dict[str, Any]:
    return {
        "name": "/".join(str(p) for p in path.path),
        "dtype": str(entry.dtype),
        "shape": list(shape),
    }


def _is_image(value: np.ndarray) -> bool:
    return value.ndim == 2 and min(value.shape) >= 4 and value.dtype.kind in "fiu"


def png_data_uri(image: np.ndarray) -> str:
    """``image`` as a grayscale PNG data URI, scaled to its range and to
    ``THUMBNAIL_SIZE`` pixels at most."""
    step = max(1, -(-max(image.shape) // THUMBNAIL_SIZE))
    pixels = np.asarray(image[::step, ::step], dtype=np.float64)
    low, high = np.nanmin(pixels), np.nanmax(pixels)
    scaled = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)
    gray = np.nan_to_num(scaled * 255).astype(np.uint8)
    return "data:image/png;base64," + base64.b64encode(png(gray)).decode()


def png(gray: np.ndarray) -> bytes:
    """An 8-bit grayscale image as PNG bytes."""
    height, width = gray.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in gray)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", header),
            chunk(b"IDAT", zlib.compress(raw)),
            chunk(b"IEND", b""),
        ],
    )
