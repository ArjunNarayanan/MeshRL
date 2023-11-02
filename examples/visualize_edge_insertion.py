from src.polygraph import PolyGraph
from src.plot import Visualizer
import math


def initialize_hex_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5],
    ]

    graph = PolyGraph(face_loops)
    graph.insert_edge(0, 2)
    graph.insert_edge(1, 1)
    graph.insert_edge(4, 1)

    return graph


c = math.cos(math.pi / 3)
s = math.sin(math.pi / 3)
coords = [[-c, -s],
          [c, -s],
          [1, 0],
          [c, s],
          [-c, s],
          [-1, 0]]
coords = dict(zip(range(6), coords))

graph = initialize_hex_graph()
vis = Visualizer(graph, coords)
vis.plot_vertices()
vis.plot_face_centroids()
vis.plot_all_halfedges()
