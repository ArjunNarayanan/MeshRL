from src.polygraph import PolyGraph
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.convert import from_networkx


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3],
    ]

    graph = PolyGraph(face_loops)
    for vertex in face_loops[0]:
        graph.nodes[(vertex, "h")]["feature"] = vertex

    return graph


graph = initialize_graph()
halfedges = [(idx, "h") for idx in range(4)]
halfedge_graph = graph.subgraph(halfedges)

# torch_graph = from_networkx(graph, group_node_attrs="feature")