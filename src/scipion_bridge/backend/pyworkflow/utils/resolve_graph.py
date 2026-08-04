"""Scipion pointer class resolution helpers."""

import inspect
from typing import Optional
import warnings
from ....core import struct
from ....core.typed.resolve import current_registry



def _get_scipion_env_info():
    """Helper to return available modules and base set/object classes."""
    import pyworkflow.object as pywfobj  # type: ignore

    try:
        import pwem.objects as emobj  # type: ignore
        base_set = (emobj.EMSet, pywfobj.Set)
        base_obj = (emobj.EMObject, pywfobj.Object)
        modules = [emobj, pywfobj]
    except ImportError:
        base_set = (pywfobj.Set,)
        base_obj = (pywfobj.Object,)
        modules = [pywfobj]

    return modules, base_set, base_obj


def _extract_scipion_container_names(cls: type, modules, base_set, base_obj):
    """Helper to resolve a node type `cls` to candidate Scipion container class names."""
    if not isinstance(cls, type):
        return []

    # Direct Set container match
    if issubclass(cls, base_set):
        return [cls.__name__]

    # Item-level match -> find corresponding Set containers in Scipion modules
    if issubclass(cls, base_obj):
        set_candidates = []
        for mod in modules:
            for _, mod_cls in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(mod_cls, base_set)
                    and getattr(mod_cls, "ITEM_TYPE", None) == cls
                ):
                    set_candidates.append(mod_cls.__name__)

        if set_candidates:
            return set_candidates

        name = cls.__name__
        return [name if name.startswith("SetOf") else f"SetOf{name}s"]

    return []


def _resolve_candidate_with_warning(candidates, target_type: type, role: str) -> Optional[str]:
    """Helper to pick a candidate and warn if multiple options exist."""
    if not candidates:
        return None

    candidate_list = sorted(list(candidates))
    if len(candidate_list) > 1:
        warnings.warn(
            f"Multiple Scipion {role} pointer classes found for type '{target_type}': {candidate_list}. "
            f"Using '{candidate_list[0]}'.",
            UserWarning,
            stacklevel=3,
        )

    return candidate_list[0]


def find_pointer_class(target_type: type) -> Optional[str]:
    """Inspect the type resolution graph to find the appropriate Scipion input
    container class (e.g. 'SetOfParticles') for a given target type.

    Traverses predecessors (Scipion -> Bridge) in the resolution graph.
    """
    modules, base_set, base_obj = _get_scipion_env_info()
    graph = current_registry().graph
    candidates = set()

    # 1. Direct Predecessor Check (Scipion -> Bridge)
    if target_type in graph:
        for u in graph.predecessors(target_type):
            raise NotImplementedError("Exact resolution not implemented yet.")
            names = _extract_scipion_container_names(u, modules, base_set, base_obj)
            candidates.update(names)

    # 2. Item-level Predecessor Check (if target_type is a Set container)
    if isinstance(target_type, type) and issubclass(target_type, struct.Set):
        item_type = target_type.item_type()
        if item_type in graph:
            for u in graph.predecessors(item_type):
                names = _extract_scipion_container_names(u, modules, base_set, base_obj)
                candidates.update(names)

    return _resolve_candidate_with_warning(candidates, target_type, role="input")


def find_output_pointer_class(target_type: type, include_itemwise=False) -> Optional[str]:
    """Inspect the type resolution graph to find the appropriate Scipion output
    container class (e.g. 'SetOfParticlesFlex') for a given target type.

    Traverses successors (Bridge -> Scipion) in the resolution graph.
    """
    modules, base_set, base_obj = _get_scipion_env_info()
    graph = current_registry().graph
    candidates = set()

    # 1. Direct Successor Check (Bridge -> Scipion)
    if target_type in graph:
        for v in graph.successors(target_type):
            names = _extract_scipion_container_names(v, modules, base_set, base_obj)
            candidates.update(names)

    # 2. Item-level Successor Check (if target_type is a Set container)
    if isinstance(target_type, type) and issubclass(target_type, struct.Set) and include_itemwise:
        item_type = target_type.item_type()
        if item_type in graph:
            for v in graph.successors(item_type):
                names = _extract_scipion_container_names(v, modules, base_set, base_obj)
                candidates.update(names)

    return _resolve_candidate_with_warning(candidates, target_type, role="output")


def find_pyworkflow_object_for_type(target_type: type) -> Optional[str]:
    """Inspect the type resolution graph to find the appropriate PyWorkflow
    object class for a given target type (defaults to input lookup).
    """
    return find_pointer_class(target_type)