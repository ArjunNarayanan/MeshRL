from src.tiler import Tiler
import unittest
import numpy as np


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    coords = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


if __name__ == "__main__":
    graph = initialize_graph()
    graph.insert_half_edge(0, 2)
    graph.insert_vertex(7)
    graph.insert_half_edge(8, 1)
