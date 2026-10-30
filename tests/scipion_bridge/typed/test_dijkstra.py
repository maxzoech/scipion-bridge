import json
from pathlib import Path

import pytest

import networkx as nx

from scipion_bridge.core.typed.dijkstra import find_shortest_path


def test_simple_graph():

    graph = nx.DiGraph()

    graph.add_edge("A", "B", weight=0, module="__main__")
    graph.add_edge("B", "C", weight=0, module="__main__")

    graph.add_edge("A", "Base", weight=1, module="__main__")
    graph.add_edge("Base", "C", weight=0, module="__main__")

    path = find_shortest_path(graph, origin="A", destination="C")
    print(path)

    assert path == ["A", "B", "C"]


def test_simple_graph_intermediate():

    graph = nx.DiGraph()

    graph.add_edge("A", "B", weight=0, module="__main__")
    graph.add_edge("B", "C", weight=0, module="__main__")

    graph.add_edge("A", "Base", weight=1, module="__main__")
    graph.add_edge("Base", "C", weight=0, module="__main__")

    path = find_shortest_path(graph, origin="A", destination="C", intermediate="Base")
    assert path == ["A", "Base", "C"]


def test_simple_graph_exceptions():

    graph = nx.DiGraph()

    graph.add_edge("A", "B", weight=0, module="__main__")
    graph.add_edge("B", "C", weight=0, module="__main__")

    graph.add_edge("A", "Base", weight=1, module="__main__")
    graph.add_edge("D", "Base", weight=0, module="__main__")

    graph.add_edge("Base", "C", weight=0, module="__main__")

    with pytest.raises(nx.exception.NetworkXNoPath):
        find_shortest_path(graph, origin="A", destination="D")

    with pytest.raises(nx.exception.NodeNotFound):
        find_shortest_path(graph, origin="A", destination="F")


def test_type_resolution_graph():

    graph = nx.DiGraph()

    json_path = Path("/".join([*__file__.split("/")[:-1], "resolve_graph.json"]))
    with open(json_path) as f:
        data = json.load(f)

        edges = data["links"]
        for edge in edges:
            data = {u: v for u, v in edge.items() if u not in {"source", "target"}}
            graph.add_edge(edge["source"], edge["target"], **data)

        path = find_shortest_path(
            graph,
            origin="scipion_bridge.typed.proxy.Output",
            destination="scipion_bridge.typed.proxy.FuncParam",
            # intermediate="__main__.Volume",
        )

        assert path == [
            "scipion_bridge.typed.proxy.Output",
            "__main__.Volume",
            "scipion_bridge.typed.proxy.Proxy",
            "scipion_bridge.typed.proxy.FuncParam",
        ]


if __name__ == "__main__":
    test_type_resolution_graph()
