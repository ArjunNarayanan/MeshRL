from src.tiler import Tiler


def initialize_triangle_graph():
    face_loops = [
        [0, 1, 2],
    ]
    graph = Tiler.from_face_loops(face_loops)
    graph.insert_vertex(1)
    return graph


graph = initialize_triangle_graph()
graph._delete_boundary_vertex(3)
