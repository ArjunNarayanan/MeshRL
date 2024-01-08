from src.tiler import Tiler
import timeit


def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


def loop_halfedges(graph):
    halfedge = 3
    for step in range(100):
        halfedge = graph.next_half_edge(halfedge)


setup = """
from src.tiler import Tiler

def loop_halfedges(graph, numsteps=100):
    halfedge = 3
    for step in range(numsteps):
        halfedge = graph.next_half_edge(halfedge)

def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph

graph = initialize_tri_quad_graph()
"""

timeit.timeit("loop_halfedges(graph)", setup=setup, number=10000)
