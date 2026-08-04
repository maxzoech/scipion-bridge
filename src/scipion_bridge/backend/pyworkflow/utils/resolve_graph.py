"""Scipion pointer class resolution helpers."""

import inspect
import warnings
from typing import Optional, Set as PySet, List, Tuple, Any, Type
from ....core import struct
from ....core.typed.resolve import current_registry


def _get_scipion_env_info() -> Tuple[List[Any], Tuple[Type, ...], Tuple[Type, ...]]:
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


def _extract_scipion_container_types(
    cls: Type, modules: List[Any], base_set: Tuple[Type, ...], base_obj: Tuple[Type, ...]
) -> List[Type]:
    """Helper to resolve a node type `cls` to candidate Scipion container Python classes."""
    if not isinstance(cls, type):
        return []

    # Direct Set container match
    if issubclass(cls, base_set):
        return [cls]

    # Item-level match -> find corresponding Set containers in Scipion modules
    if issubclass(cls, base_obj):
        set_candidates: List[Type] = []
        for mod in modules:
            for _, mod_cls in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(mod_cls, base_set)
                    and getattr(mod_cls, "ITEM_TYPE", None) == cls
                ):
                    set_candidates.append(mod_cls)

        if set_candidates:
            return set_candidates

        return [cls]

    return []


def _resolve_candidate_type_with_warning(
    candidates: PySet[Type], target_type: Type, role: str
) -> Optional[Type]:
    """Helper to pick a candidate type and warn if multiple options exist."""
    if not candidates:
        return None

    candidate_list = sorted(list(candidates), key=lambda c: c.__name__)
    if len(candidate_list) > 1:
        names = [c.__name__ for c in candidate_list]
        warnings.warn(
            f"Multiple Scipion {role} pointer classes found for type '{target_type}': {names}. "
            f"Using '{candidate_list[0].__name__}'.",
            UserWarning,
            stacklevel=3,
        )

    return candidate_list[0]


def find_pointer_class(target_type: Type) -> Optional[str]:
    """Inspect the type resolution graph to find the appropriate Scipion input
    container class name (e.g. 'SetOfParticles') for a given target type.

    Traverses predecessors (Scipion -> Bridge) in the resolution graph.
    """
    modules, base_set, base_obj = _get_scipion_env_info()
    graph = current_registry().graph
    candidates: PySet[Type] = set()

    # 1. Direct Predecessor Check (Scipion -> Bridge)
    if target_type in graph:
        for u in graph.predecessors(target_type):
            types = _extract_scipion_container_types(u, modules, base_set, base_obj)
            candidates.update(types)

    # 2. Item-level Predecessor Check (if target_type is a Set container)
    if isinstance(target_type, type) and issubclass(target_type, struct.Set):
        item_type = target_type.item_type()
        if item_type in graph:
            for u in graph.predecessors(item_type):
                types = _extract_scipion_container_types(u, modules, base_set, base_obj)
                candidates.update(types)

    cls = _resolve_candidate_type_with_warning(candidates, target_type, role="input")
    return cls.__name__ if cls else None


def find_output_pointer_class(
    target_type: Type, include_itemwise: bool = False
) -> Optional[Type]:
    """Inspect the type resolution graph to find the appropriate Scipion output
    container Python type (e.g. SetOfParticlesFlex) for a given target type.

    Traverses successors (Bridge -> Scipion) in the resolution graph.
    """
    modules, base_set, base_obj = _get_scipion_env_info()
    graph = current_registry().graph
    candidates: PySet[Type] = set()

    # 1. Direct Successor Check (Bridge -> Scipion)
    if target_type in graph:
        for v in graph.successors(target_type):
            types = _extract_scipion_container_types(v, modules, base_set, base_obj)
            candidates.update(types)

    # 2. Item-level Successor Check (if target_type is a Set container)
    if isinstance(target_type, type) and issubclass(target_type, struct.Set) and include_itemwise:
        item_type = target_type.item_type()
        if item_type in graph:
            for v in graph.successors(item_type):
                types = _extract_scipion_container_types(v, modules, base_set, base_obj)
                candidates.update(types)

    return _resolve_candidate_type_with_warning(candidates, target_type, role="output")


def find_pyworkflow_object_for_type(target_type: Type) -> Optional[str]:
    """Inspect the type resolution graph to find the appropriate PyWorkflow
    object class for a given target type (defaults to input lookup).
    """
    return find_pointer_class(target_type)