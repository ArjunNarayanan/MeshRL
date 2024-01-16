from src.polygraph import PolyGraph
import timeit


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3],
    ]

    graph = PolyGraph.from_face_loops(face_loops)

    return graph


def operate(graph):
    graph.insert_halfedge(0, 1)

    hidx = graph.next_halfedge(1)
    graph.insert_vertex(hidx)

    hidx = graph.previous_halfedge(0)
    graph.delete_source_vertex(hidx)

    hidx = graph.previous_halfedge(0)
    graph.delete_halfedge(hidx)


setup = """
from src.polygraph import PolyGraph

def initialize_graph():
    face_loops = [
        [0, 1, 2, 3],
    ]

    graph = PolyGraph.from_face_loops(face_loops)

    return graph


def operate(graph):
    graph.insert_halfedge(0, 1)
    
    hidx = graph.next_halfedge(1)
    graph.insert_vertex(hidx)
    
    hidx = graph.previous_halfedge(0)
    graph.delete_source_vertex(hidx)
    
    hidx = graph.previous_halfedge(0)
    graph.delete_halfedge(hidx)
    
graph = initialize_graph()
"""

timeit.timeit("operate(graph)", setup=setup, number=10000)
