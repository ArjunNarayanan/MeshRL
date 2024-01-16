import timeit
from src.tiler import Tiler


def initialize_graph():
    face_loops = [
        [0, 1, 14, 11, 12, 13],
        [1, 2, 3, 4, 15, 14],
        [4, 5, 6, 7, 8, 15],
        [8, 9, 10, 11, 14, 15]
    ]
    graph = Tiler.from_face_loops(face_loops)

    return graph


def iterate_vertex_degree(graph):
    return [graph.vertex_degree(vidx) for vidx in range(16)]


def bunch_vertex_degree(graph):
    vertices = range(16)
    return graph.vertex_degree_of_list(vertices)


graph = initialize_graph()

setup = """
from src.tiler import Tiler


def initialize_graph():
    face_loops = [
        [0, 1, 14, 11, 12, 13],
        [1, 2, 3, 4, 15, 14],
        [4, 5, 6, 7, 8, 15],
        [8, 9, 10, 11, 14, 15]
    ]
    graph = Tiler.from_face_loops(face_loops)

    return graph


def iterate_vertex_degree(graph):
    return [graph.vertex_degree(vidx) for vidx in range(16)]
    

def bunch_vertex_degree(graph):
    vertices = [(vidx, graph.vertex_tag) for vidx in range(16)]
    return graph.degree(vertices)



graph = initialize_graph()
"""

timeit.timeit("bunch_vertex_degree(graph)", setup=setup, number=10000)
