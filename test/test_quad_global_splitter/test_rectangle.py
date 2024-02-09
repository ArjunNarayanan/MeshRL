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
        self.quad_splitter = QuadGlobalSplitter(self.graph)

    def test_aspect_ratios(self):
        quad_splitter = self.quad_splitter
        quad_splitter.update_half_edge_aspect_ratios()

        test_half_edge_aspect_ratio = 2 * [10, 0.1, 10, 0.1, ] + 2 * [1 / 1.9, 1.9, 1 / 1.9, 1.9]
        htag = self.graph.half_edge_tag
        half_edges = [(idx, htag) for idx in range(16)]
        test_half_edge_aspect_ratio = dict(zip(half_edges, test_half_edge_aspect_ratio))

        self.assertTrue(
            all(test_half_edge_aspect_ratio[k] == v for k, v in quad_splitter.half_edge_aspect_ratios.items()))

    def test_global_line_half_edges(self):
        half_edges = self.graph._get_global_line_half_edges(0, 3)
        htag = self.graph.half_edge_tag

        test_half_edges = [(0, htag), (8, htag)]
        self.assertEqual(half_edges, test_half_edges)

    def test_refine_and_global_line(self):
        self.graph.global_split_to_boundary(0, 3)
        half_edges = self.graph._get_global_line_half_edges(11, 4)
        htag = self.graph.half_edge_tag

        test_half_edges = [(idx, htag) for idx in [11, 22, 15]]
        self.assertEqual(half_edges, test_half_edges)

    def test_quantile_score(self):
        self.quad_splitter.update_half_edge_aspect_ratios()
        score = self.quad_splitter.get_global_split_quantile_score(0)
        self.assertAlmostEqual(score, 0.5 * (10 + 1 / 1.9))

    def test_quantile_score_on_refine(self):
        self.graph.global_split_to_boundary(0, 3)
        self.quad_splitter.update_half_edge_aspect_ratios()

        score = self.quad_splitter.get_global_split_quantile_score(11)
        self.assertAlmostEqual(score, 3.8)


if __name__ == "__main__":
    unittest.main()
