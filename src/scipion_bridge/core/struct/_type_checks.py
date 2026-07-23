"""Type checking utilities for the struct system.

This module exists to break circular dependencies between struct.py and
schema.py. Both modules need to check whether a type is a struct type,
but they also import from each other. By extracting this check into a
separate module with no internal dependencies, both can safely import
from here.

The check uses a sentinel attribute (_bridge_struct_marker) rather than
importing Struct directly, which would re-introduce the circular dependency.
"""

from typing import Any


def is_struct_type(dtype: Any) -> bool:
    """Check if a type is a Bridge struct type.

    Uses a structural marker attribute rather than an isinstance/issubclass
    check against Struct, to avoid circular imports.
    """
    return isinstance(dtype, type) and getattr(dtype, '_bridge_struct_marker', False) is True
