"""Scipion pointer class resolution helpers."""

import inspect
from typing import Optional
from ....core.struct import Set
from ....core.typed.resolve import current_registry



def find_pointer_class(target_type: type) -> Optional[str]:
    """Inspect the type resolution graph to find the appropriate Scipion
    container class (e.g. 'SetOfParticles') for a given target type.
    """

    import pwem.objects as emobj  # type: ignore
    import pyworkflow.object as pywfobj  # type: ignore
    
    graph = current_registry().graph

    # 1. Direct resolver check (e.g. SetOfParticles -> Set[spa.Particle])
    if target_type in graph:
        for u in graph.predecessors(target_type):
            if isinstance(u, type) and (
                issubclass(u, emobj.EMObject) or issubclass(u, pywfobj.Object)
            ):
                return u.__name__

    # 2. Item-level resolver check (e.g. pwem.objects.Particle -> spa.Particle)
    if isinstance(target_type, type) and issubclass(target_type, Set):
        item_type = target_type.item_type()
        if item_type in graph:
            for u in graph.predecessors(item_type):
                if isinstance(u, type) and (
                    issubclass(u, emobj.EMObject) or issubclass(u, pywfobj.Object)
                ):
                    candidates = []
                    for _, cls in inspect.getmembers(emobj, inspect.isclass):
                        if (
                            issubclass(cls, emobj.EMSet)
                            or issubclass(cls, pywfobj.Set)
                        ) and getattr(cls, "ITEM_TYPE", None) == u:
                            candidates.append(cls.__name__)

                    if candidates:
                        exact_match = f"SetOf{u.__name__}s"
                        if exact_match in candidates:
                            return exact_match
                        setof_candidates = [c for c in candidates if c.startswith("SetOf")]
                        return setof_candidates[0] if setof_candidates else candidates[0]

                    # Fallback naming convention if not explicitly in pwem.objects
                    name = u.__name__
                    return name if name.startswith("SetOf") else f"SetOf{name}s"

