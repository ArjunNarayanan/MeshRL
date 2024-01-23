from src.tiler import Tiler


def initialize_graph():
    face_loops = [
        [11, 12, 14, 15, 0, 1],
        [3, 10, 11, 1, 2],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [11, 10, 9, 12],
        [12, 9, 13]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


graph = initialize_graph()
# nbrs = graph.knn_half_edges_with_boundary(6, 30)