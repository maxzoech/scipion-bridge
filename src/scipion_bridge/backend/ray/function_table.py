"""The map functions of a pipeline, shipped to the workers once.

A map call would otherwise carry its function as an argument, which Ray
pickles on every call: for a protocol's method, the whole protocol and, for a
class defined in ``__main__``, the class itself. A pipeline instead puts all of
its map functions in the object store once, as a ``FunctionTable``, and its
calls carry a reference to it. Every worker process unpacks a table once and
keeps it in a ``TableCache``. Pickle stores an object once per payload, so the
methods of a protocol in one table share one instance of it.

This is correct because protocols are sealed once their pipeline is built: a
copy per process behaves the same as a copy per call.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import ray

from .mailbox import Ref
from .profiling import ProfileConfig


@dataclass(frozen=True)
class FunctionTable:
    """The map functions of a pipeline, by label, and what their calls need."""

    functions: Mapping[str, Callable[[Any], Any]]
    # Values served to Field.value in the process running a call.
    parameters: Mapping[str, Any]
    profile: Optional[ProfileConfig]
    # Builds the process-scope resources of a protocol, by its compute group.
    preparers: Mapping[str, Callable[[], None]]


class TableCache:
    """The function tables a process has unpacked, least recently used first.

    Task workers are pooled and outlive pipelines, so the cache keeps only the
    last ``size`` tables. Threaded actors look tables up concurrently.
    """

    def __init__(self, load: Callable[[Any], FunctionTable], size: int) -> None:
        self._load = load
        self._size = size
        self._tables: OrderedDict[Any, FunctionTable] = OrderedDict()
        self._lock = threading.Lock()

    def table(self, functions: Ref) -> FunctionTable:
        """The table ``functions`` refers to, unpacked once per process."""
        with self._lock:
            if functions.ref not in self._tables:
                self._tables[functions.ref] = self._load(functions.ref)
                while len(self._tables) > self._size:
                    self._tables.popitem(last=False)
            self._tables.move_to_end(functions.ref)
            return self._tables[functions.ref]


# The function tables of the pipelines whose maps ran in this process: an
# executor shared by the children of a group_by gets the table of each.
_CACHED_TABLES = 16
_TABLES = TableCache(ray.get, size=_CACHED_TABLES)


def table_of(functions: Ref) -> FunctionTable:
    """The table ``functions`` refers to, from the cache of this process.

    Ray pickles actor classes by value, with the module globals their methods
    use. Reached through a function of this importable module, which is
    pickled by reference, the cache and its lock stay out of the pickle.
    """
    return _TABLES.table(functions)
