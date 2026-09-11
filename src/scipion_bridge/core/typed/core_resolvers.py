import numpy as np
from .resolve import current_registry, resolver
from ..struct.utils.arrow_utils import RaggedArrayView


@resolver
def resolve_any_to_str(value: object) -> str:
    return str(value)


@resolver
def resolve_tuple_to_str(value: tuple) -> str:
    return " ".join([current_registry().resolve(v, astype=str) for v in value])


@resolver
def resolve_ragged_view_to_ndarray(value: RaggedArrayView) -> np.ndarray:
    """Resolve a RaggedArrayView directly into a NumPy ndarray."""
    return value.to_numpy()
