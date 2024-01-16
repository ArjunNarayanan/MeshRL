from src.polygraph import PolyGraph
import timeit


def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = PolyGraph.from_face_loops(face_loops)

    return graph


def loop_halfedges(graph):
    halfedge = 3
    for step in range(100):
        halfedge = graph.next_halfedge(halfedge)


setup = """
from src.polygraph import PolyGraph

def loop_halfedges(graph, numsteps=100):
    halfedge = 3
    for step in range(numsteps):
        halfedge = graph.next_halfedge(halfedge)

def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = PolyGraph.from_face_loops(face_loops)

    return graph
    
graph = initialize_tri_quad_graph()
"""

timeit.timeit("loop_halfedges(graph)", setup=setup, number=10000)
