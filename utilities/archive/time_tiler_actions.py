from src.tiler import Tiler
import timeit


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3],
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


def operate(graph):
    graph.insert_half_edge(0, 1)

    hidx = graph.next_half_edge(1)
    graph.insert_vertex(hidx)

    hidx = graph.previous_half_edge(0)
    graph.delete_source_vertex(hidx)

    hidx = graph.previous_half_edge(0)
    graph.delete_half_edge(hidx)


setup = """
from src.tiler import Tiler
import timeit


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3],
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


def operate(graph):
    graph.insert_half_edge(0, 1)

    hidx = graph.next_half_edge(1)
    graph.insert_vertex(hidx)

    hidx = graph.previous_half_edge(0)
    graph.delete_source_vertex(hidx)

    hidx = graph.previous_half_edge(0)
    graph.delete_half_edge(hidx)

graph = initialize_graph()
"""

timeit.timeit("operate(graph)", setup=setup, number=10000)