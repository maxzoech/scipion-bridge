import networkx as nx

from typing import (
    List,
    Optional,
    Any,
    Callable,
    Generic,
    TypeVar,
)

import heapq as hq

T = TypeVar("T")


class PathfindingContainer(Generic[T]):

    def __init__(self, value: T, previous: Optional[T], weight: int) -> None:

        self.value = value
        self.previous = previous
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight


def build_default_container(
    graph: nx.DiGraph, value: T, previous: Optional[T], weight: int
):
    del graph

    return PathfindingContainer(value, previous, weight)


def find_shortest_path(
    graph: nx.DiGraph,
    origin: T,
    destination: T,
    intermediate: Optional[Any] = None,
    container_builder: Callable[
        [nx.DiGraph, T, Optional[T], int], PathfindingContainer
    ] = build_default_container,
    weight: str = "weight",
):

    if intermediate is not None:

        path_1 = find_shortest_path(
            graph,
            origin,
            intermediate,
            weight=weight,
            container_builder=container_builder,
        )

        path_2 = find_shortest_path(
            graph,
            intermediate,
            destination,
            weight=weight,
            container_builder=container_builder,
        )

        return path_1 + path_2[1:]

    if origin not in graph.nodes or destination not in graph.nodes:
        raise nx.NodeNotFound()

    predecessors = {}
    heap = [container_builder(graph, origin, None, 0)]

    hq.heapify(heap)

    while heap:
        element = hq.heappop(heap)  # type: ignore

        if element.value in predecessors:
            continue  # Already encountered before

        predecessors[element.value] = element.previous

        for neighbor in graph.neighbors(element.value):
            if neighbor in predecessors:
                continue

            cost = graph.get_edge_data(element.value, neighbor)[weight]
            hq.heappush(
                heap,
                container_builder(
                    graph, neighbor, element.value, (element.weight + cost)
                ),
            )

        if element.value == destination:
            break
    else:
        raise nx.NetworkXNoPath()

    # Backtrack path
    def _find_path(path: List):
        node = path[0]
        if node == origin:
            return path
        else:
            return _find_path([predecessors[node]] + path)

    return _find_path(path=[destination])
