from src.polygraph import PolyGraph
from src.plot import Visualizer


def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = PolyGraph(face_loops)

    return graph


graph = initialize_tri_quad_graph()
coordinates = [[0, 0],
               [1, -1],
               [2, -1],
               [2, 1],
               [1, 1]]
coords = dict(zip(range(5), coordinates))

vis = Visualizer(graph, coords)
vis.plot_vertices()
vis.plot_face_centroids()
vis.plot_all_halfedges()
