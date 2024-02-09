import unittest
import numpy as np
from src.tiler import Tiler
from src.quad_global_splitter import QuadGlobalSplitter


def initialize_graph():
    coords = generate_coordinates()
    faces = [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ]
    graph = Tiler.from_face_loops(faces, vertex_coordinates=coords)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 0.1],
        [1, 0.1],
        [2, 0.1],
        [0, 2],
        [1, 2],
        [2, 2]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestAspectRatio(unittest.TestCase):
    def setUp(self):
        self.graph = initialize_graph()

    def test_aspect_ratios(self):
        quad_splitter = QuadGlobalSplitter(self.graph)
        quad_splitter.update_half_edge_aspect_ratios()

        test_half_edge_aspect_ratio = 2 * [10, 0.1, 10, 0.1, ] + 2 * [1 / 1.9, 1.9, 1 / 1.9, 1.9]
        htag = self.graph.half_edge_tag
        half_edges = [(idx, htag) for idx in range(16)]
        test_half_edge_aspect_ratio = dict(zip(half_edges, test_half_edge_aspect_ratio))

        self.assertTrue(
            all(test_half_edge_aspect_ratio[k] == v for k, v in quad_splitter.half_edge_aspect_ratios.items()))
