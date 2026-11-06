"""Fingerprints identifying what a protocol and the stages of its pipeline compute.

A fingerprint is a SHA-256 digest of the code of a protocol, its parameter
values and its sealed state. Two runs whose stages have equal fingerprints run
the same computation, which lineage records, checkpoints and profiles of
earlier runs rely on.

Values are encoded canonically where possible. Other values are encoded as
their pickle, which may differ between processes for equal values: the
fingerprints then differ, so that two stages are never taken for the same
computation when they are not.
"""

import hashlib
import inspect
from enum import Enum
from pathlib import PurePath
from typing import Any, Protocol, runtime_checkable

import cloudpickle  # type: ignore
import numpy as np

from ..streaming.node import Node
from ..streaming.ops import MapElementOp, MapOp


@runtime_checkable
class HasFingerprint(Protocol):
    """An object identified by a fingerprint of its own, e.g. a protocol."""

    def fingerprint(self) -> str: ...


def _join(*parts: bytes) -> bytes:
    """Concatenate ``parts`` unambiguously, each prefixed with its length."""
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def digest(*parts: bytes) -> str:
    """The hex SHA-256 digest of ``parts``."""
    return hashlib.sha256(_join(*parts)).hexdigest()


def fingerprint_value(value: Any) -> bytes:
    """Canonical encoding of ``value``: equal values have equal encodings."""
    match value:
        case HasFingerprint():
            return _join(b"fingerprint", value.fingerprint().encode())
        case Enum():
            return _join(
                b"enum",
                type(value).__qualname__.encode(),
                fingerprint_value(value.value),
            )
        case None | bool() | int() | float() | str() | bytes():
            return _join(b"scalar", repr(value).encode())
        case PurePath():
            return _join(b"path", str(value).encode())
        case tuple():
            return _join(b"tuple", *(fingerprint_value(element) for element in value))
        case frozenset():
            return _join(
                b"frozenset",
                *sorted(fingerprint_value(element) for element in value),
            )
        case np.ndarray():
            return _join(
                b"array",
                value.dtype.str.encode(),
                repr(value.shape).encode(),
                hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(),
            )
        case _:
            return _join(b"pickle", cloudpickle.dumps(value))


def class_identity(cls: type) -> bytes:
    """Encoding of the code of ``cls``: its qualified name and its source.

    A class without source on disk (defined in ``__main__``, a notebook or
    dynamically) is encoded as its pickle, which cloudpickle writes by value.
    """
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return _join(b"pickled class", cloudpickle.dumps(cls))
    else:
        return _join(
            b"class",
            f"{cls.__module__}.{cls.__qualname__}".encode(),
            source.encode(),
        )


def _stage_label(node: Node) -> str:
    match node:
        case MapOp(func=func) | MapElementOp(func=func):
            func_name = getattr(func, "__qualname__", type(func).__name__)
            return f"{type(node).__name__}({func_name})"
        case _:
            return type(node).__name__


def stage_fingerprint(protocol_fp: str, ordinal: int, node: Node) -> str:
    """Fingerprint of a stage that a protocol's pipeline creates.

    Args:
        protocol_fp: Fingerprint of the protocol.
        ordinal: Position of ``node`` in the ``lineage`` of the protocol's
            pipeline. It tells apart stages of the same kind, e.g. two
            ``chunk`` stages.
        node: The node of the stage.
    """
    return digest(
        protocol_fp.encode(),
        str(ordinal).encode(),
        _stage_label(node).encode(),
    )
