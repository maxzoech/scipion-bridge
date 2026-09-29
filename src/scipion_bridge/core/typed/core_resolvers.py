import awkward as ak
import numpy as np
from .resolve import current_registry, resolver


@resolver
def resolve_any_to_str(value: object) -> str:
    return str(value)


@resolver
def resolve_tuple_to_str(value: tuple) -> str:
    return " ".join([current_registry().resolve(v, astype=str) for v in value])


@resolver
def resolve_awkward_to_ndarray(value: ak.Array) -> np.ndarray:
    """Resolve an Awkward array directly into a NumPy ndarray."""
    return ak.to_numpy(value)
