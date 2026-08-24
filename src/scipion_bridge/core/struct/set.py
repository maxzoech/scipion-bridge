"""Set container implementation.

A Set is a fixed-capacity sequence container for Struct items, stored as flattened
N-dimensional Zarr arrays across the outer capacity dimension.
"""

from __future__ import annotations
import types
from typing import (
    Type,
    Generic,
    TypeVar,
    Dict,
    Union,
    Any,
    ForwardRef,
    Tuple,
    Optional,
    Sequence,
    overload,
    TYPE_CHECKING,
)

from .struct import Struct
from .storage import SchemaArrayStorage
from ..utils.marker import Marker

T = TypeVar("T", bound=Struct)

class Set(Marker[T], SchemaArrayStorage):

    def __init__(self) -> None:
        raise NotImplementedError("Set is not yet implemented")